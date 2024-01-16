#include <ATen/native/SharedReduceOps.h> // WelfordData/WelfordOps
#include <c10/cuda/CUDAMathCompat.h> // rsqrt
#include <ATen/AccumulateType.h> // acc_type
#include "scale_shift_kernel.h" // scale_shift
#include <thrust/pair.h> // thrust::pair
#include <vector> // std::vector
#define THREADS_PER_BLOCK 128 // low threads per block bad because less occupancy, high threads per block bad because of smaller reduction loops -> more instruction overhead

template <typename T>
__global__ void
compute_stats2(
        const T* X,
        //const int D,
        const int H,
        const int W,
        const int C,
        const int G,
        //const int HWd,
        const float eps,
        T* means,
        T* rstds
  ) {
  using T_ACC = at::acc_type<T, true>;
  using WelfordType = at::native::WelfordData<T_ACC, int>;
  using WelfordOp = at::native::WelfordOps<T_ACC, T_ACC, int, thrust::pair<T_ACC, T_ACC>>;
  /*
     D >= 8:
       griddim: (N, G); blockdim: (D, d)
        d = num. spatial elements (from HW dimension) each thread-block processes in parallel
        Dd = THREADS_PER_BLOCK
       X shape: (N, H, W, G, D) -> (N, HW/d, d, G, D); X stride: (HWC, dGD, GD, D, 1)
       thread-block reduction: (H, W, D) -> 1
     D < 8:
       griddim: (N, G/g); blockdim: (e, d)
        g = num. groups each block computes in parallel
        d = num. spatial elements (from HW dimension) each thread-block processes in parallel
        e = num. elements loaded in one coalesced read
        Dg = e
        ed = Ddg = THREADS_PER_BLOCK
       X shape: (N, H, W, G, D) -> (N, HW/d, d, G/g, e); X stride: (HWC, dGD, GD, e, 1)
       thread-block reduction: (H, W, e) -> g
   */

  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);

  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type vals_reduced_arr[THREADS_PER_BLOCK];
  WelfordType *vals_reduced = reinterpret_cast<WelfordType*>(vals_reduced_arr);

  //const int HWC = HWd * THREADS_PER_BLOCK * G;
  const int D = C / G;
  const int e = blockDim.x;
#pragma unroll 8
  //for (int i = 0; i < HWd; ++i) {
  for (int i = 0; i < H * W / (int)blockDim.y; ++i) {
    int reduce_idx = 0;
    reduce_idx += blockIdx.x * H * W * C; // dim 0, HWGD stride
    reduce_idx += i * blockDim.y * G * D; // dim 1, dGD stride
    if (D == e)
      reduce_idx += blockIdx.y * D; // dim 3, D stride
    else
      reduce_idx += blockIdx.y * e; // dim 3, e stride
    reduce_idx += threadIdx.x; // dim 4, 1 stride
    //int reduce_idx = (i * THREADS_PER_BLOCK * G) + (threadIdx.y * G * D) + (blockIdx.y * D) + threadIdx.x; // only works if THREADS_PER_BLOCK >= D but realistically this will happen all the time
    //T x = X[blockIdx.x * HWC + reduce_idx];
    //T x = X[(blockIdx.x * H * W * C) + reduce_idx];
    T x = X[reduce_idx];
    val = welford_op.reduce(val, static_cast<T_ACC>(x), reduce_idx); // last arg isn't used in src
  }

  // reduce (D,) -> single welford data
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  vals_reduced[tid] = val;
  __syncthreads();

  for (int stride = (int)(blockDim.x * blockDim.y / 2); stride >= 1; stride >>= 1) {
    if (tid < stride)
      vals_reduced[tid] = welford_op.combine(vals_reduced[tid], vals_reduced[tid + stride]);
    __syncthreads();
    }

  // place value into output buffer
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    T_ACC m1, m2;
    thrust::tie(m2, m1) = welford_op.project(vals_reduced[threadIdx.x]);
    means[blockIdx.x * G + blockIdx.y] = m1;
    rstds[blockIdx.x * G + blockIdx.y] = c10::cuda::compat::rsqrt(m2 + static_cast<T_ACC>(eps));
  }
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
  const T* X_data = X.const_data_ptr<T>();
  T* mean_data = means.mutable_data_ptr<T>();
  T* rstd_data = rstds.mutable_data_ptr<T>();

  const int N = X.size(0);
  const int H = X.size(1);
  const int W = X.size(2);
  const int C = X.size(3);
  const int D = C / G;
  int blockDimX, blockDimY, gridDimY;
  blockDimX = D;
  blockDimY = THREADS_PER_BLOCK / blockDimX;
  gridDimY = G;

  dim3 dimGrid(N, gridDimY);
  dim3 dimBlock(blockDimX, blockDimY);

  compute_stats2<T><<<dimGrid, dimBlock>>>(
      X_data, H, W, C, G, eps,
      mean_data, rstd_data
  );

  scale_shift<T>(X, weight, bias, G, Y, means, rstds);
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
