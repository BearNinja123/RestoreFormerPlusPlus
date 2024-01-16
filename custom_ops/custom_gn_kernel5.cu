#include <ATen/native/SharedReduceOps.h> // WelfordData/WelfordOps
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAMathCompat.h> // rsqrt
#include <ATen/AccumulateType.h> // acc_type
#include "scale_shift_kernel.h" // scale_shift
#include <thrust/pair.h> // thrust::pair
#include <vector> // std::vector
#define THREADS_PER_BLOCK 512 // 512 slightly faster (~3%) than 1024 because of higher theoretical occupancy -> higher mem throughput

template <typename T>
__global__ void
compute_stats5_pt1(
        const T* X,
        const int C,
        const int G,
        const int H,
        const int W,
        at::native::WelfordData<at::acc_type<T, true>, int> *welford_data
  ) {
  using T_ACC = at::acc_type<T, true>;
  using WelfordType = at::native::WelfordData<T_ACC, int>;
  using WelfordOp = at::native::WelfordOps<T_ACC, T_ACC, int, thrust::pair<T_ACC, T_ACC>>;
  // griddim = N, H, blockdim = C, c

  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);

  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type vals_reduced_arr[THREADS_PER_BLOCK];
  WelfordType *vals_reduced = reinterpret_cast<WelfordType*>(vals_reduced_arr);

  const int Wc = W * C / THREADS_PER_BLOCK;
#pragma unroll 8
  for (int i = 0; i < Wc; ++i) {
    int reduce_idx = i * THREADS_PER_BLOCK + threadIdx.y * C + threadIdx.x; // only works if THREADS_PER_BLOCK >= D but realistically this will happen all the time
    T x = X[blockIdx.x * H * W * C + blockIdx.y * W * C + reduce_idx];
    val = welford_op.reduce(val, static_cast<T_ACC>(x), reduce_idx); // last arg isn't used in src
  }

  const int D = C / G;

  // suppose vals_reduced shape is (c, G, D), we need (G,) output
  // (c,C) -> (c,G,D) -> (D,c,G) -> (G,)
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int c_idx = threadIdx.y;
  const int g = threadIdx.x / D;
  const int d = threadIdx.x % D;
  vals_reduced[d * blockDim.y * G + c_idx * G + g] = val;
  __syncthreads();

  for (int stride = THREADS_PER_BLOCK / 2; stride >= G; stride >>= 1) {
    if (tid < stride)
      vals_reduced[tid] = welford_op.combine(vals_reduced[tid], vals_reduced[tid + stride]);
    __syncthreads();
    }

  // put reduced outputs into return buffers
  if ((int)threadIdx.x < G && threadIdx.y == 0) {
    welford_data[blockIdx.x * H * G + blockIdx.y * G + threadIdx.x] = vals_reduced[threadIdx.x];
  }
}

template <typename T>
__global__ void
compute_stats5_pt2(
        at::native::WelfordData<at::acc_type<T, true>, int> *welford_data,
        const int G,
        const int H,
        const float eps,
        T* means,
        T* rstds
  ) {
  using T_ACC = at::acc_type<T, true>;
  using WelfordType = at::native::WelfordData<T_ACC, int>;
  using WelfordOp = at::native::WelfordOps<T_ACC, T_ACC, int, thrust::pair<T_ACC, T_ACC>>;
  // griddim = N, blockdim = G, g
  // (H, G) -> (G,)

  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);

  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type vals_reduced_arr[THREADS_PER_BLOCK];
  WelfordType *vals_reduced = reinterpret_cast<WelfordType*>(vals_reduced_arr);
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;

#pragma unroll 8
  for (int i = 0; i < H / (int)blockDim.y; ++i) {
    WelfordType x = welford_data[blockIdx.x * H * G + i * THREADS_PER_BLOCK + tid];
    val = welford_op.combine(val, x);
  }

  // (g, G) -> (G,)
  vals_reduced[tid] = val;
  __syncthreads();

  for (int stride = THREADS_PER_BLOCK / 2; stride >= G; stride >>= 1) {
    if (tid < stride)
      vals_reduced[tid] = welford_op.combine(vals_reduced[tid], vals_reduced[tid + stride]);
    __syncthreads();
    }

  // put reduced outputs into return buffers
  if (threadIdx.y == 0) {
    T_ACC m1, m2;
    thrust::tie(m2, m1) = welford_op.project(vals_reduced[threadIdx.x]);
    means[blockIdx.x * G + threadIdx.x] = m1;
    rstds[blockIdx.x * G + threadIdx.x] = c10::cuda::compat::rsqrt(m2 + static_cast<T_ACC>(eps));
  }
}

template <typename T>
void gn_nhwc_forward_kernel5(
    const torch::Tensor& X,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const int G,
    T eps,
    torch::Tensor& Y,
    torch::Tensor& means,
    torch::Tensor& rstds) {
  const T* X_data = X.const_data_ptr<T>();
  T* mean_data = means.mutable_data_ptr<T>();
  T* rstd_data = rstds.mutable_data_ptr<T>();

  const int N = X.size(0);
  const int H = X.size(1);
  const int W = X.size(2);
  const int C = X.size(3);
  int blockDimX, blockDimY, gridDimY, gridDimZ;
  blockDimX = C;
  blockDimY = THREADS_PER_BLOCK / blockDimX;
  gridDimY = H;
  gridDimZ = 1;

  dim3 dimGrid(N, gridDimY, gridDimZ);
  dim3 dimBlock(blockDimX, blockDimY);

  using WelfordType = at::native::WelfordData<at::acc_type<T, true>, int>;
  torch::Tensor welford_tensor = torch::empty({N, H, G, sizeof(WelfordType)}, X.options().dtype(torch::kByte));
  WelfordType *welford_data = reinterpret_cast<WelfordType *>(welford_tensor.mutable_data_ptr());
  
  compute_stats5_pt1<<<dimGrid, dimBlock>>>(
      X_data, C, G, H, W, 
      welford_data
  );

  blockDimX = G;
  blockDimY = THREADS_PER_BLOCK / blockDimX;
  gridDimY = 1;
  gridDimZ = 1;

  dimGrid = dim3(N, gridDimY, gridDimZ);
  dimBlock = dim3(blockDimX, blockDimY);
  compute_stats5_pt2<<<dimGrid, dimBlock>>>(
          welford_data,
          G, H, eps,
          mean_data, rstd_data
    );

  scale_shift<T>(X, weight, bias, G, Y, means, rstds);
  AT_CUDA_CHECK(cudaGetLastError());
}

std::vector<torch::Tensor> gn_nhwc_cuda_forward5(
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
      gn_nhwc_forward_kernel5<scalar_t>(
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