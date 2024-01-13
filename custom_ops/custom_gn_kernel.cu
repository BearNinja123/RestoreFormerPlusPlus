#include <ATen/native/SharedReduceOps.h> // WelfordData/WelfordOps
#include <ATen/native/cuda/Loops.cuh>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAMathCompat.h> // rsqrt
#include <ATen/AccumulateType.h> // acc_type
#include <ATen/ATen.h>
#include <string>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <thrust/pair.h>
#include <cuda.h>
#include <vector>
#define THREADS_PER_BLOCK 512 // 512 slightly faster (~3%) than 1024 because of higher theoretical occupancy -> higher mem throughput

// Reduces a value across the y-threads of a threadblock
template <typename T, class ReduceOp>
__device__ void
y_reduce(
    T val,
    const ReduceOp& op,
    T* output_buffer
    ) {
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  output_buffer[tid] = val;
  __syncthreads();

  for (int stride = (int)(blockDim.x * blockDim.y / 2); stride >= (int)blockDim.x; stride >>= 1) {
    if (tid < stride)
      output_buffer[tid] = op.combine(output_buffer[tid], output_buffer[tid + stride]);
    __syncthreads();
  }
  // output_buffer[0..blockDim.x] will have reduced values
}

template <typename T>
__global__ void
compute_stats(
        const T* X,
        const int G,
        const int HWd,
        const float eps,
        T* means,
        T* rstds
  ) {
  using T_ACC = at::acc_type<T, true>;
  using WelfordType = at::native::WelfordData<T_ACC, int>;
  using WelfordOp = at::native::WelfordOps<T_ACC, T_ACC, int, thrust::pair<T_ACC, T_ACC>>;

  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);

  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type vals_reduced_arr[THREADS_PER_BLOCK];
  WelfordType *vals_reduced = reinterpret_cast<WelfordType*>(vals_reduced_arr);

  const int HWC = HWd * THREADS_PER_BLOCK;
#pragma unroll 8
  for (int i = 0; i < HWd; ++i) {
    int reduce_idx;
    if (THREADS_PER_BLOCK >= G)
      reduce_idx = i * THREADS_PER_BLOCK + threadIdx.y * G + threadIdx.x;
    else
      reduce_idx = i * G + blockIdx.y * THREADS_PER_BLOCK + threadIdx.x;
    T x = X[blockIdx.x * HWC + reduce_idx];
    val = welford_op.reduce(val, static_cast<T_ACC>(x), reduce_idx); // last arg isn't used in src
  }

  y_reduce(val, welford_op, vals_reduced);
  if (threadIdx.y == 0) {
    T_ACC m1, m2;
    thrust::tie(m2, m1) = welford_op.project(vals_reduced[threadIdx.x]);
    //means[blockIdx.x * G + threadIdx.x] = m1;
    //rstds[blockIdx.x * G + threadIdx.x] = c10::cuda::compat::rsqrt(m2 + static_cast<T_ACC>(eps));
    means[blockIdx.x * G + blockIdx.y * THREADS_PER_BLOCK + threadIdx.x] = m1;
    rstds[blockIdx.x * G + blockIdx.y * THREADS_PER_BLOCK + threadIdx.x] = c10::cuda::compat::rsqrt(m2 + static_cast<T_ACC>(eps));
  }
}

/*
   In the fwd, there is a (X - mean) * rstd * weight + bias which requires an add, mul, fma op on each elem on X
   The above eqn can be rewritten as:
     X * rstd * weight - mean * rstd * weight + bias ->
     X * a + b (a = rstd * weight, b = -mean * rstd * weight + bias)
    The X * a + b eqn uses only one fma per elem on X, making it faster than the original eqn since a, b doesn't have to be repeated for each elem of X
*/
template <typename T>
__global__ void
compute_scale_biases(
        T* means,  // (N, G)
        T* rstds,  // (N, G)
        const T* weight, // (C)
        const T* bias,   // (C)
        const int G,
        const int C,
        at::acc_type<T, true>* a,            // (N, C)
        at::acc_type<T, true>* b             // (N, C)
  ) {
  const int c = blockIdx.y * THREADS_PER_BLOCK + threadIdx.x;
  const int g = c % G;
  const int nc = blockIdx.x * C + c;
  const int ng = blockIdx.x * G + g;
  a[nc] = rstds[ng] * weight[c];
  b[nc] = -means[ng] * rstds[ng] * weight[c] + bias[c];
}

template <typename T>
void gn_nhwc_forward_kernel(
    const torch::Tensor& X,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const int G,
    T eps,
    torch::Tensor& Y,
    torch::Tensor& means,
    torch::Tensor& rstds) {
  using T_ACC = at::acc_type<T, true>;
  const T* X_data = X.const_data_ptr<T>();
  T* mean_data = means.mutable_data_ptr<T>();
  T* rstd_data = rstds.mutable_data_ptr<T>();
  const T* weight_data = weight.const_data_ptr<T>();
  const T* bias_data = bias.const_data_ptr<T>();

  const int N = X.size(0);
  const int H = X.size(1);
  const int W = X.size(2);
  const int C = X.size(3);
  const int D = C / G;
  int blockDimX, blockDimY, gridDimY;
  if (THREADS_PER_BLOCK >= G) {
    blockDimX = G;
    blockDimY = THREADS_PER_BLOCK / G;
    gridDimY = 1;
  }
  else {
    blockDimX = THREADS_PER_BLOCK;
    blockDimY = 1;
    gridDimY = G / THREADS_PER_BLOCK;
  }
  const int HWd = H * W * D / blockDimY;

  dim3 dimGrid(N, gridDimY);
  //const dim3 dimGrid(N);
  dim3 dimBlock(blockDimX, blockDimY);
  compute_stats<T><<<dimGrid, dimBlock>>>(
      X_data, G, HWd, eps,
      mean_data, rstd_data
  );

  const at::ScalarType kAccType =
      (X.scalar_type() == at::kHalf || X.scalar_type() == at::kBFloat16)
      ? at::kFloat
      : X.scalar_type();

  torch::Tensor a = torch::empty({N, C}, X.options().dtype(kAccType));
  torch::Tensor b = torch::empty({N, C}, X.options().dtype(kAccType));
  T_ACC* a_data = a.mutable_data_ptr<T_ACC>();
  T_ACC* b_data = b.mutable_data_ptr<T_ACC>();
  
  if (THREADS_PER_BLOCK >= C) {
    blockDimX = C;
    blockDimY = THREADS_PER_BLOCK / C;
    gridDimY = 1;
  }
  else {
    blockDimX = THREADS_PER_BLOCK;
    blockDimY = 1;
    gridDimY = C / THREADS_PER_BLOCK;
  }
  compute_scale_biases<<<dim3(N, gridDimY), dim3(blockDimX, blockDimY)>>>(
      mean_data, rstd_data,
      weight_data, bias_data,
      G, C,
      a_data, b_data);

  at::TensorIterator iter = at::TensorIteratorConfig()
    .check_all_same_dtype(std::is_same<T, T_ACC>::value) // this line relaxes requirement that all inputs/outputs are same dtype if T isn't T_ACC 
    .resize_outputs(false)
    .add_owned_output(Y.view({N, H * W, D, G}))
    .add_owned_input(X.view({N, H * W, D, G}))
    .add_owned_input(a.view({N, 1, D, G}))
    .add_owned_input(b.view({N, 1, D, G}))
    .build();

  //at::native::gpu_kernel(iter, [] GPU_LAMBDA(T x, T mean, T rstd, T weight, T bias) -> T {
  at::native::gpu_kernel(iter, [] GPU_LAMBDA(T x, T_ACC a, T_ACC b) -> T {
      //return (static_cast<T_ACC>(x) - static_cast<T_ACC>(mean)) * static_cast<T_ACC>(rstd) * static_cast<T_ACC>(weight) + static_cast<T_ACC>(bias);
      return static_cast<T_ACC>(x) * a + b;
      });
  AT_CUDA_CHECK(cudaGetLastError());
}

std::vector<torch::Tensor> gn_nhwc_cuda_forward(
    const torch::Tensor& X,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const int G,
    float eps) {
  const int N = X.size(0);

  torch::Tensor X_nhwc = X.permute({0, 2, 3, 1});
  torch::Tensor X_out = torch::empty_like(X_nhwc);
  torch::Tensor means = torch::empty({N, G}, weight.options());
  torch::Tensor rstds = torch::empty({N, G}, weight.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    X.scalar_type(),
    "group_norm_nhwc_forward", [&]() {
      gn_nhwc_forward_kernel<scalar_t>(
          X_nhwc,
          weight,
          bias,
          G,
          eps,
          X_out,
          means,
          rstds
      );
  });
  return {X_out.permute({0, 3, 1, 2}), means, rstds};
}
