#include <ATen/native/SharedReduceOps.h> // WelfordData/WelfordOps
#include <ATen/native/cuda/Loops.cuh>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAMathCompat.h> // rsqrt
#include <ATen/AccumulateType.h> // acc_type
#include <string>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <thrust/pair.h>
#include <cuda.h>
#include <vector>
#define THREADS_PER_BLOCK 1024

// Reduces a value across the y-threads of a threadblock
template <typename T, class ReduceOp>
__device__ void
y_reduce(
    T val,
    const ReduceOp& op,
    //const T& identity_element,
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
  using WelfordType = at::native::WelfordData<T_ACC, int64_t>;
  using WelfordOp = at::native::WelfordOps<T_ACC, T_ACC, int64_t, thrust::pair<T_ACC, T_ACC>>;

  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);

  const int g = blockDim.y;
  for (int i = 0; i < HWd; ++i) {
    T x = X[blockIdx.x * HWd * G + i * g * G + threadIdx.y * G + threadIdx.x];
    val = welford_op.reduce(val, static_cast<T_ACC>(x), i); // check last arg, probably wrong
  }

  __shared__ typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type vals_reduced_arr[THREADS_PER_BLOCK];
  WelfordType *vals_reduced = reinterpret_cast<WelfordType*>(vals_reduced_arr);
  y_reduce(val, welford_op, vals_reduced);
  if (threadIdx.y == 0) {
    T_ACC m1, m2;
    thrust::tie(m2, m1) = welford_op.project(vals_reduced[threadIdx.x]);
    means[blockIdx.x * G + threadIdx.x] = m1;
    rstds[blockIdx.x * G + threadIdx.x] = c10::cuda::compat::rsqrt(m2 + static_cast<T_ACC>(eps));
  }
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

  const int N = X.size(0);
  const int H = X.size(1);
  const int W = X.size(2);
  const int C = X.size(3);
  const int D = C / G;
  const int g = THREADS_PER_BLOCK / G;
  const int d = D / g;

  const dim3 dimGrid(N);
  const dim3 dimBlock(G, g);
  compute_stats<T><<<dimGrid, dimBlock>>>(
      X_data, G, H * W * d, eps,
      mean_data, rstd_data
  );

  at::TensorIterator iter = at::TensorIteratorConfig()
    .resize_outputs(false)
    .add_owned_output(Y.view({N, H * W, D, G}))
    .add_owned_input(X.view({N, H * W, D, G}))
    .add_owned_input(means.view({N, 1, 1, G}))
    .add_owned_input(rstds.view({N, 1, 1, G}))
    .add_owned_input(weight.view({1, 1, D, G}))
    .add_owned_input(bias.view({1, 1, D, G}))
    .build();

  at::native::gpu_kernel(iter, [] GPU_LAMBDA(T x, T mean, T rstd, T weight, T bias) -> T {
      return (static_cast<T_ACC>(x) - static_cast<T_ACC>(mean)) * static_cast<T_ACC>(rstd) * static_cast<T_ACC>(weight) + static_cast<T_ACC>(bias);
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
  torch::Tensor X_out = torch::zeros_like(X_nhwc);
  torch::Tensor means = torch::zeros({N, G}).to(X.device());
  torch::Tensor rstds = torch::zeros({N, G}).to(X.device());

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

// Reduces a value across the y-threads of a threadblock
template <typename T>
__device__ void
y_sum_reduce(
    T val,
    T* output_buffer
    ) {
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  output_buffer[tid] = val;
  __syncthreads();
  for (int stride = (int)(blockDim.x * blockDim.y / 2); stride >= (int)blockDim.x; stride >>= 1) {
    if (tid < stride)
      output_buffer[tid] = output_buffer[tid] + output_buffer[tid + stride];
    __syncthreads();
  }
  // output_buffer[0..blockDim.x] will have reduced values
}

template <typename T>
__global__ void
compute_backward_params(
        const T* dy,
        const T* X,
        const T* means,
        const T* rstds,
        const T* weight,
        const int HWD,
        const int G,
        at::acc_type<T, true>* coef2,
        at::acc_type<T, true>* coef3) {
  using T_ACC = at::acc_type<T, true>;
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;
  const int g = THREADS_PER_BLOCK / G;
  const int HWd = HWD / g;
  const int ng = blockIdx.x * G + threadIdx.x; // [bix][tx]

  for (int i = 0; i < HWd; ++i) {
    int index = blockIdx.x * HWD*G + i * g*G + threadIdx.y * G + threadIdx.x; // [bix][i][ty][tx]
    T_ACC gamma_v = static_cast<T_ACC>(weight[ng]);
    sum1 += static_cast<T_ACC>(dy[index]) * static_cast<T_ACC>(X[index]) * gamma_v; // sum1 = sum(X * gamma * dy)
    sum2 += static_cast<T_ACC>(dy[index]) * gamma_v; // sum1 = sum(X * gamma * dy)
  }

  __shared__ T_ACC reduced_sum1[THREADS_PER_BLOCK];
  __shared__ T_ACC reduced_sum2[THREADS_PER_BLOCK];
  y_sum_reduce(sum1, reduced_sum1);
  y_sum_reduce(sum2, reduced_sum2);

  if (threadIdx.y == 0) {
    T_ACC sum1 = reduced_sum1[threadIdx.x];
    T_ACC sum2 = reduced_sum2[threadIdx.x];

    T_ACC mean = static_cast<T_ACC>(means[ng]);
    T_ACC rstd = static_cast<T_ACC>(rstds[ng]);
    T_ACC x = (sum2 * static_cast<T_ACC>(means[ng]) - sum1) * rstd * rstd * rstd / HWD; // AKA -sum(norm(X[ng])[i] * gamma[g] * dy[ngi], i) / std[ng]^2 / D
    coef2[ng] = x;
    float c3a = -x * mean; // AKA sum(norm(X[ngi]) * gamma[g] * dy[ngi] * mean[ng], i)
    float c3b = -sum2 * rstd / HWD; // AKA -gamma * sum(dy) / std[ng] / D
    coef3[ng] = c3a + c3b;
  }
}

template <typename T>
__global__ void
backward_weight_bias(
        const T* dy,
        const T* X,
        const T* means,
        const T* rstds,
        const int HW,
        const int C,
        const int G,
        T* dweight,
        T* dbias) {
  using T_ACC = at::acc_type<T, true>;
  T_ACC dweight_sum = 0;
  T_ACC dbias_sum = 0;

  const int c = THREADS_PER_BLOCK / C;
  const int Hw = HW / c;
  const int ng = blockIdx.x * G + threadIdx.x % G; // [bix][i][ty][tx]

  for (int i = 0; i < Hw; ++i) {
  //for (int i = 0; i < 1; ++i) {
    int index = blockIdx.x * HW*C + i * THREADS_PER_BLOCK + threadIdx.y * C + threadIdx.x; // [bix][i][ty][tx]
    //int index = threadIdx.x; // [bix][i][ty][tx]
    T_ACC dy_elem = static_cast<T_ACC>(dy[index]);
    dweight_sum += dy_elem * (static_cast<T_ACC>(X[index]) - static_cast<T_ACC>(means[ng])) * static_cast<T_ACC>(rstds[ng]);
    dbias_sum += dy_elem;
  }

  __shared__ T_ACC reduced_dweight[THREADS_PER_BLOCK];
  __shared__ T_ACC reduced_dbias[THREADS_PER_BLOCK];
  y_sum_reduce(dweight_sum, reduced_dweight);
  y_sum_reduce(dbias_sum, reduced_dbias);

  if (threadIdx.y == 0) {
    dweight[threadIdx.x] = reduced_dweight[threadIdx.x];
    dbias[threadIdx.x] = reduced_dbias[threadIdx.x];
    //dweight[threadIdx.x] = dweight_sum;
    //dbias[threadIdx.x] = dbias_sum;
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
  T* dweight_data = dweight.mutable_data_ptr<T>();
  T* dbias_data = dbias.mutable_data_ptr<T>();

  const int N = X_nhwc.size(0);
  const int H = X_nhwc.size(1);
  const int W = X_nhwc.size(2);
  const int C = X_nhwc.size(3);
  const int D = C / G;
  const int g = THREADS_PER_BLOCK / G;

  torch::Tensor coef2 = at::empty({N, G}, X_nhwc.options());
  torch::Tensor coef3 = at::empty({N, G}, X_nhwc.options());
  T_ACC* coef2_data = coef2.mutable_data_ptr<T_ACC>();
  T_ACC* coef3_data = coef3.mutable_data_ptr<T_ACC>();

  const dim3 dimGrid(N);
  const dim3 dimBlock(G, g);
  compute_backward_params<T><<<dimGrid, dimBlock>>>(
      dy_data, X_data, mean_data, rstd_data, weight_data,
      H * W * D, G,
      coef2_data, coef3_data
  );

  at::TensorIterator iter = at::TensorIteratorConfig()
    .resize_outputs(false)
    .add_owned_output(dx.view({N, H * W, D, G}))
    .add_owned_input(dy_nhwc.view({N, H * W, D, G}))
    .add_owned_input(X_nhwc.view({N, H * W, D, G}))
    .add_owned_input(weight.view({1, 1, D, G}))
    .add_owned_input(rstds.view({N, 1, 1, G}))
    .add_owned_input(coef2.view({N, 1, 1, G}))
    .add_owned_input(coef3.view({N, 1, 1, G}))
    .build();

  at::native::gpu_kernel(iter, [] GPU_LAMBDA(T dy, T x, T weight, T rstd, T_ACC coef2, T_ACC coef3) -> T {
      const T_ACC c1 = static_cast<T_ACC>(rstd) * static_cast<T_ACC>(weight);
      return (c1 * static_cast<T_ACC>(dy)) + (coef2 * static_cast<T_ACC>(x)) + coef3;
    });

  backward_weight_bias<T><<<N, dim3(C, THREADS_PER_BLOCK / C)>>>(
      dy_data, X_data, mean_data, rstd_data, 
      H * W, C, G,
      dweight_data, dbias_data);

  AT_CUDA_CHECK(cudaGetLastError());
}

std::vector<torch::Tensor> gn_nhwc_cuda_backward(
    const torch::Tensor& dy,
    const torch::Tensor& X,
    const torch::Tensor& weight,
    const torch::Tensor& means,
    const torch::Tensor& rstds,
    const int G) {
  const int C = X.size(1);
  const torch::Tensor X_nhwc = X.permute({0, 2, 3, 1});
  const torch::Tensor dy_nhwc = dy.permute({0, 2, 3, 1});
  torch::Tensor dx = torch::zeros_like(dy_nhwc);
  torch::Tensor dgamma = torch::zeros({C}).to(dy.device());
  torch::Tensor dbeta = torch::zeros({C}).to(dy.device());

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
