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
#define THREADS_PER_BLOCK 128 // 512 slightly faster (~3%) than 1024 because of higher theoretical occupancy -> higher mem throughput

// Reduces a value across the y-threads of a threadblock
template <typename T, class ReduceOp>
__device__ void
full_reduce(
    T val,
    const ReduceOp& op,
    T* output_buffer
    ) {
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  output_buffer[tid] = val;
  __syncthreads();

  for (int stride = (int)(blockDim.x * blockDim.y / 2); stride >= 1; stride >>= 1) {
    if (tid < stride)
      output_buffer[tid] = op.combine(output_buffer[tid], output_buffer[tid + stride]);
    __syncthreads();
    }
}
template <typename T>
__global__ void
compute_stats2(
        const T* X,
        const int D,
        const int G,
        const int HWd,
        const float eps,
        T* means,
        T* rstds
  ) {
  using T_ACC = at::acc_type<T, true>;
  using WelfordType = at::native::WelfordData<T_ACC, int>;
  using WelfordOp = at::native::WelfordOps<T_ACC, T_ACC, int, thrust::pair<T_ACC, T_ACC>>;
  // griddim = N, G, blockdim = d, D

  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);

  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type vals_reduced_arr[THREADS_PER_BLOCK];
  WelfordType *vals_reduced = reinterpret_cast<WelfordType*>(vals_reduced_arr);

  const int HWC = HWd * THREADS_PER_BLOCK * G;
//#pragma unroll 8
  //for (int i = 0; i < HWd/4; ++i) {
  for (int i = 0; i < HWd; ++i) {
    //int reduce_idx = i * THREADS_PER_BLOCK * G * 4 + threadIdx.y * D * G + blockIdx.y * D + threadIdx.x * 4; // only works if THREADS_PER_BLOCK >= D but realistically this will happen all the time
    //const float4 tmp = reinterpret_cast<const float4 *>(X + (blockIdx.x * HWC + reduce_idx))[0];
    //val = welford_op.reduce(val, static_cast<T_ACC>(tmp.x), reduce_idx); // last arg isn't used in src
    //val = welford_op.reduce(val, static_cast<T_ACC>(tmp.y), reduce_idx); // last arg isn't used in src
    //val = welford_op.reduce(val, static_cast<T_ACC>(tmp.z), reduce_idx); // last arg isn't used in src
    //val = welford_op.reduce(val, static_cast<T_ACC>(tmp.w), reduce_idx); // last arg isn't used in src

    int reduce_idx = i * THREADS_PER_BLOCK * G + threadIdx.y * D * G + blockIdx.y * D + threadIdx.x; // only works if THREADS_PER_BLOCK >= D but realistically this will happen all the time
    T x = X[blockIdx.x * HWC + reduce_idx];
    val = welford_op.reduce(val, static_cast<T_ACC>(x), reduce_idx); // last arg isn't used in src
  }

  full_reduce(val, welford_op, vals_reduced);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    T_ACC m1, m2;
    thrust::tie(m2, m1) = welford_op.project(vals_reduced[threadIdx.x]);
    means[blockIdx.x * G + blockIdx.y] = m1;
    rstds[blockIdx.x * G + blockIdx.y] = c10::cuda::compat::rsqrt(m2 + static_cast<T_ACC>(eps));
  }
}

template <typename T>
__global__ void
compute_scale_biases2(
        T* means,  // (N, G)
        T* rstds,  // (N, G)
        const T* weight, // (C)
        const T* bias,   // (C)
        const int G,
        const int C,
        at::acc_type<T, true>* a,            // (N, C)
        at::acc_type<T, true>* b             // (N, C)
  ) {
  const int D = C / G;
  //const int g = blockIdx.y * blockDim.y + threadIdx.y;
  //const int c = g * D + threadIdx.x;
  const int c = threadIdx.x;
  const int g = c / D;
  const int nc = blockIdx.x * C + c;
  const int ng = blockIdx.x * G + g;
  const at::acc_type<T, true> a_nc = rstds[ng] * weight[c];
  a[nc] = a_nc;
  b[nc] = -means[ng] * a_nc + bias[c];
}

template <typename T>
void gn_nhwc_forward_kernel2(
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
  int blockDimX, blockDimY, gridDimY, gridDimZ;
  //blockDimX = D / 4;
  blockDimX = D;
  blockDimY = THREADS_PER_BLOCK / blockDimX;
  gridDimY = G;
  gridDimZ = 1;

  dim3 dimGrid(N, gridDimY, gridDimZ);
  dim3 dimBlock(blockDimX, blockDimY);

  const int HWd = H * W * D / THREADS_PER_BLOCK;
  compute_stats2<T><<<dimGrid, dimBlock>>>(
      X_data, D, G, HWd, eps,
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

  const int TPB = 128;
  gridDimY = C / TPB;
  gridDimY = gridDimY > 0 ? gridDimY : 1;
  blockDimY = TPB >= D ? TPB / D : 1;
  //compute_scale_biases2<<<dim3(N, gridDimY), dim3(D, blockDimY)>>>( // note: max(D, T) threads per block
  compute_scale_biases2<<<N, C>>>( // note: max(D, T) threads per block
      mean_data, rstd_data,
      weight_data, bias_data,
      G, C,
      a_data, b_data);

  at::TensorIterator iter = at::TensorIteratorConfig()
    .check_all_same_dtype(std::is_same<T, T_ACC>::value) // this line relaxes requirement that all inputs/outputs are same dtype if T isn't T_ACC 
    .resize_outputs(false)
    .add_owned_output(Y.view({N, H * W, G, D}))
    .add_owned_input(X.view({N, H * W, G, D}))
    .add_owned_input(a.view({N, 1, G, D}))
    .add_owned_input(b.view({N, 1, G, D}))
    .build();

  at::native::gpu_kernel(iter, [] GPU_LAMBDA(T x, T_ACC a, T_ACC b) -> T {
      return static_cast<T_ACC>(x) * a + b;
      });
  AT_CUDA_CHECK(cudaGetLastError());
}

std::vector<torch::Tensor> gn_nhwc_cuda_forward2(
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
      gn_nhwc_forward_kernel2<scalar_t>(
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
