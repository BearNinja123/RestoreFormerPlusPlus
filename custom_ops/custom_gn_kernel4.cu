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
#define THREADS_PER_BLOCK 128 // low threads per block bad because less occupancy, high threads per block bad because of smaller reduction loops -> more instruction overhead

template <typename T>
__global__ void
compute_stats4(
        const T* X,
        const int D,
        const int G,
        const int HWC,
        const int num_elems_coalesced,
        const float eps,
        T* means,
        T* rstds
  ) {
  using T_ACC = at::acc_type<T, true>;
  using WelfordType = at::native::WelfordData<T_ACC, int>;
  using WelfordOp = at::native::WelfordOps<T_ACC, T_ACC, int, thrust::pair<T_ACC, T_ACC>>;
  // griddim = N, G/g, blockdim = num_elems_coalesced (gD), c

  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);

  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type vals_reduced_arr[THREADS_PER_BLOCK];
  WelfordType *vals_reduced = reinterpret_cast<WelfordType*>(vals_reduced_arr);

  const int HWd = HWC / gridDim.y / THREADS_PER_BLOCK;
#pragma unroll 8
  for (int i = 0; i < HWd; ++i) {
    int reduce_idx = i * blockDim.y * D * G + threadIdx.y * D * G + blockIdx.y * num_elems_coalesced + threadIdx.x; // only works if THREADS_PER_BLOCK >= D but realistically this will happen all the time
    T x = X[blockIdx.x * HWC + reduce_idx];
    val = welford_op.reduce(val, static_cast<T_ACC>(x), reduce_idx); // last arg isn't used in src
  }

  // suppose vals_reduced shape is (c, g, D), we need (g,) output
  // (c,g,D) -> (D,c,g) -> (g,) (where g * D = num_elems_coalesced)
  const int g = num_elems_coalesced / D; // number of groups this thread block is processing
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int c_idx = threadIdx.y;
  const int g_idx = threadIdx.x / D;
  const int d = threadIdx.x % D;
  vals_reduced[d * blockDim.y * g + c_idx * g + g_idx] = val;
  __syncthreads();

  for (int stride = THREADS_PER_BLOCK / 2; stride >= g; stride >>= 1) {
    if (tid < stride)
      vals_reduced[tid] = welford_op.combine(vals_reduced[tid], vals_reduced[tid + stride]);
    __syncthreads();
    }

  // put reduced outputs into return buffers
  if ((int)threadIdx.x < g && threadIdx.y == 0) {
    T_ACC m1, m2;
    thrust::tie(m2, m1) = welford_op.project(vals_reduced[threadIdx.x]);
    //thrust::tie(m2, m1) = welford_op.project(vals_reduced[blockDim.y * D * threadIdx.x]);
    means[blockIdx.x * G + blockIdx.y * g + threadIdx.x] = m1;
    rstds[blockIdx.x * G + blockIdx.y * g + threadIdx.x] = c10::cuda::compat::rsqrt(m2 + static_cast<T_ACC>(eps));
  }
}

template <typename T>
__global__ void
compute_scale_biases4(
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
void gn_nhwc_forward_kernel4(
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
  const int num_elems_coalesced = 8; // reads 8 floats consecutively, will still cause uncoalesced reads for bf16 (2 bytes/float * 8 floats -> 16 bytes, 16 bytes < 32 bytes/coalesced read) but happens to work better than reading 16 floats because of larger grid size
  int blockDimX, blockDimY, gridDimY, gridDimZ;
  blockDimX = num_elems_coalesced;
  blockDimY = THREADS_PER_BLOCK / blockDimX;
  gridDimY = C / blockDimX;
  gridDimZ = 1;

  dim3 dimGrid(N, gridDimY, gridDimZ);
  dim3 dimBlock(blockDimX, blockDimY);

  const int HWC = H * W * C;
  compute_stats4<T><<<dimGrid, dimBlock>>>(
      X_data, D, G, HWC, num_elems_coalesced, eps,
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

  compute_scale_biases4<<<N, C>>>( // note: max(D, T) threads per block
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

std::vector<torch::Tensor> gn_nhwc_cuda_forward4(
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
      gn_nhwc_forward_kernel4<scalar_t>(
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
