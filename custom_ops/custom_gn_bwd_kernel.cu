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

// let output_buffer be size THREADS_PER_BLOCK and can be reshaped to (THREADS_PER_BLOCK/num_elems, num_elems); output will be sum(output_buffer, dim=0)
template <typename T>
__device__ void
sum_reduce(
    T val,
    T* output_buffer,
    const int num_elems
    ) {
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  output_buffer[tid] = val;
  __syncthreads();
  for (int stride = (int)(blockDim.x * blockDim.y / 2); stride >= num_elems; stride >>= 1) {
    if (tid < stride)
      output_buffer[tid] = output_buffer[tid] + output_buffer[tid + stride];
    __syncthreads();
  }
  // output_buffer[0..blockDim.x] will have reduced values
}

template <typename T>
__global__ void
compute_backward_params_weight_bias(
        const T* dy,    // N x HWD x G
        const T* X,     // N x HWD x G
        const T* means, // N x G
        const T* rstds, // N x G
        const T* weight, // C (AKA DG)
        const int HW,
        const int C,
        const int G,
        at::acc_type<T, true>* coef2,
        at::acc_type<T, true>* coef3,
        at::acc_type<T, true>* dweight,
        at::acc_type<T, true>* dbias
        ) {
  using T_ACC = at::acc_type<T, true>;
  const int HWD = HW * C / G;
  //const int c_idx = THREADS_PER_BLOCK >= C ? threadIdx.x : blockIdx.y * THREADS_PER_BLOCK + threadIdx.x;
  const int c_idx = blockIdx.y * THREADS_PER_BLOCK + threadIdx.x;
  const int g = c_idx % G;
  const int ng = blockIdx.x * G + g;
  const int Hw = HW / blockDim.y;
  T_ACC dweight_sum = 0;
  T_ACC dbias_sum = 0;
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;

#pragma unroll 8
  for (int i = 0; i < Hw; ++i) {
    int index;
    if (THREADS_PER_BLOCK >= C)
      index = blockIdx.x * HW*C + i * THREADS_PER_BLOCK + threadIdx.y * C + c_idx; // [bix][i][ty][tx]
    else
      index = blockIdx.x * HW*C + i * C + c_idx; // [bix][i][ty][tx]

    T_ACC gamma_v = static_cast<T_ACC>(weight[c_idx]);
    T_ACC dy_elem = static_cast<T_ACC>(dy[index]);
    T_ACC X_elem = static_cast<T_ACC>(X[index]);
    sum1 += dy_elem * X_elem * gamma_v; // sum1 = sum(X * gamma * dy)
    sum2 += dy_elem * gamma_v; // sum1 = sum(X * gamma * dy)

    dweight_sum += dy_elem * (static_cast<T_ACC>(X[index]) - static_cast<T_ACC>(means[ng])) * static_cast<T_ACC>(rstds[ng]);
    dbias_sum += dy_elem;
  }

  __shared__ T_ACC reduced_sum1[THREADS_PER_BLOCK];
  __shared__ T_ACC reduced_sum2[THREADS_PER_BLOCK];
  int num_elems = THREADS_PER_BLOCK >= G ? G : THREADS_PER_BLOCK;
  sum_reduce(sum1, reduced_sum1, num_elems);
  sum_reduce(sum2, reduced_sum2, num_elems);

  if (threadIdx.y == 0 && (int)threadIdx.x < G) {
    const int ntg = blockIdx.x * G * gridDim.y + blockIdx.y * G + g;
    sum1 = reduced_sum1[g];
    sum2 = reduced_sum2[g];

    T_ACC mean = static_cast<T_ACC>(means[ng]);
    T_ACC rstd = static_cast<T_ACC>(rstds[ng]);
    T_ACC x = (sum2 * static_cast<T_ACC>(means[ng]) - sum1) * rstd * rstd * rstd / HWD; // AKA -sum(norm(X[ng])[i] * gamma[g] * dy[ngi], i) / std[ng]^2 / D
    //coef2[ng] = x;
    coef2[ntg] = x;
    float c3a = -x * mean; // AKA sum(norm(X[ngi]) * gamma[g] * dy[ngi] * mean[ng], i)
    float c3b = -sum2 * rstd / HWD; // AKA -gamma * sum(dy) / std[ng] / D
    //coef3[ng] = c3a + c3b;
    coef3[ntg] = c3a + c3b;
  }

  // new aliases for shmem arrays for different reductions
  T_ACC* reduced_dweight = reduced_sum1;
  T_ACC* reduced_dbias = reduced_sum2;
  num_elems = THREADS_PER_BLOCK >= C ? C : THREADS_PER_BLOCK;
  sum_reduce(dweight_sum, reduced_dweight, num_elems);
  sum_reduce(dbias_sum, reduced_dbias, num_elems);

  if (threadIdx.y == 0) {
    dweight[blockIdx.x * C + blockIdx.y * THREADS_PER_BLOCK + threadIdx.x] = reduced_dweight[threadIdx.x];
    dbias[blockIdx.x * C + blockIdx.y * THREADS_PER_BLOCK + threadIdx.x] = reduced_dbias[threadIdx.x];
  }
}

template <typename T>
void gn_nhwc_backward_kernel(
    const torch::Tensor& dy_nhwc,
    const torch::Tensor& X_nhwc,
    const torch::Tensor& weight,
    const torch::Tensor& means,
    const torch::Tensor& rstds,
    const int G,
    torch::Tensor& dx,
    torch::Tensor& dweight,
    torch::Tensor& dbias) {
  using T_ACC = at::acc_type<T, true>;
  const T* dy_data = dy_nhwc.const_data_ptr<T>();
  const T* X_data = X_nhwc.const_data_ptr<T>();
  const T* weight_data = weight.const_data_ptr<T>();
  const T* mean_data = means.const_data_ptr<T>();
  const T* rstd_data = rstds.const_data_ptr<T>();

  const int N = X_nhwc.size(0);
  const int H = X_nhwc.size(1);
  const int W = X_nhwc.size(2);
  const int C = X_nhwc.size(3);
  const int D = C / G;

  const at::ScalarType kAccType =
      (X_nhwc.scalar_type() == at::kHalf || X_nhwc.scalar_type() == at::kBFloat16)
      ? at::kFloat
      : X_nhwc.scalar_type();

  torch::Tensor dweight_tmp = torch::empty({N, C}, dweight.options().dtype(kAccType));
  torch::Tensor dbias_tmp = torch::empty({N, C}, dbias.options().dtype(kAccType));
  T_ACC* dweight_tmpdata = dweight_tmp.mutable_data_ptr<T_ACC>();
  T_ACC* dbias_tmpdata = dbias_tmp.mutable_data_ptr<T_ACC>();

  int blockDimX, blockDimY, gridDimY;
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

  torch::Tensor coef1 = torch::empty({N, C}, X_nhwc.options().dtype(kAccType));
  torch::Tensor coef2_tmp = torch::empty({N, gridDimY, G}, X_nhwc.options().dtype(kAccType));
  torch::Tensor coef3_tmp = torch::empty({N, gridDimY, G}, X_nhwc.options().dtype(kAccType));
  T_ACC* coef2_tmpdata = coef2_tmp.mutable_data_ptr<T_ACC>();
  T_ACC* coef3_tmpdata = coef3_tmp.mutable_data_ptr<T_ACC>();

  compute_backward_params_weight_bias<<<dim3(N, gridDimY), dim3(blockDimX, blockDimY)>>>(
  //int c = THREADS_PER_BLOCK / C;
  //compute_backward_params_weight_bias<<<N, dim3(C, c)>>>(
          dy_data, X_data,
          mean_data, rstd_data, weight_data,
          H * W, C, G,
          coef2_tmpdata, coef3_tmpdata,
          dweight_tmpdata, dbias_tmpdata);

  torch::Tensor coef2 = coef2_tmp.sum(1);
  torch::Tensor coef3 = coef3_tmp.sum(1);

  at::TensorIterator coef1_iter = at::TensorIteratorConfig()
    .check_all_same_dtype(std::is_same<T, T_ACC>::value)
    .add_owned_output(coef1.view({N, D, G}))
    .add_owned_input(rstds.view({N, 1, G}))
    .add_owned_input(weight.view({1, D, G}))
    .build();

  at::native::gpu_kernel(coef1_iter, [] GPU_LAMBDA(T rstd, T gamma) -> T_ACC {
    return static_cast<T_ACC>(rstd) * static_cast<T_ACC>(gamma);
  });

  at::TensorIterator iter = at::TensorIteratorConfig()
    .resize_outputs(false)
    .check_all_same_dtype(std::is_same<T, T_ACC>::value)
    .add_owned_output(dx.view({N, H * W, D, G}))
    .add_owned_input(dy_nhwc.view({N, H * W, D, G}))
    .add_owned_input(X_nhwc.view({N, H * W, D, G}))
    .add_owned_input(coef1.view({N, 1, D, G}))
    .add_owned_input(coef2.view({N, 1, 1, G}))
    .add_owned_input(coef3.view({N, 1, 1, G}))
    .build();

  at::native::gpu_kernel(iter, [] GPU_LAMBDA(T dy, T x, T_ACC coef1, T_ACC coef2, T_ACC coef3) -> T {
      T_ACC result1 = (coef2 * x) + coef3; // fma
      return (coef1 * dy) + result1; // fma
    });

  dweight = dweight_tmp.sum(0).to(dweight.scalar_type());
  dbias = dbias_tmp.sum(0).to(dweight.scalar_type());

  AT_CUDA_CHECK(cudaGetLastError());
}

std::vector<torch::Tensor> gn_nhwc_cuda_backward(
    const torch::Tensor& dy,
    const torch::Tensor& X,
    const torch::Tensor& weight,
    const torch::Tensor& means,
    const torch::Tensor& rstds,
    const int G) {
  //const int N = X.size(0);
  const int C = X.size(1);
  const torch::Tensor X_nhwc = X.permute({0, 2, 3, 1});
  const torch::Tensor dy_nhwc = dy.permute({0, 2, 3, 1});
  torch::Tensor dx = torch::empty_like(dy_nhwc);
  torch::Tensor dgamma = torch::empty({C}, dy.options());
  torch::Tensor dbeta = torch::empty({C}, dy.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    X.scalar_type(),
    "group_norm_nhwc_backward", [&]() {
      gn_nhwc_backward_kernel<scalar_t>(
          dy_nhwc,
          X_nhwc,
          weight,
          means,
          rstds,
          G,
          dx,
          dgamma,
          dbeta
      );
  });

  return {dx.permute({0, 3, 1, 2}), dgamma, dbeta};
}
