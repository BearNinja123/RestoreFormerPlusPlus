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
#pragma unroll
  for (int i = 0; i < HWd; ++i) {
    const int reduce_idx = i * THREADS_PER_BLOCK + threadIdx.y * G + threadIdx.x;
    T x = X[blockIdx.x * HWC + reduce_idx];
    val = welford_op.reduce(val, static_cast<T_ACC>(x), reduce_idx); // last arg isn't used in src
  }

  y_reduce(val, welford_op, vals_reduced);
  if (threadIdx.y == 0) {
    T_ACC m1, m2;
    thrust::tie(m2, m1) = welford_op.project(vals_reduced[threadIdx.x]);
    means[blockIdx.x * G + threadIdx.x] = m1;
    rstds[blockIdx.x * G + threadIdx.x] = c10::cuda::compat::rsqrt(m2 + static_cast<T_ACC>(eps));
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
  const int c = threadIdx.x;
  const int g = threadIdx.x % G;
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
  const int g = THREADS_PER_BLOCK / G;
  const int HWd = H * W * D / g;

  const dim3 dimGrid(N);
  const dim3 dimBlock(G, g);
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
  
  compute_scale_biases<<<N, C>>>(
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
compute_backward_params(
        const T* dy,    // N x HWD x G
        const T* X,     // N x HWD x G
        const T* means, // N x G
        const T* rstds, // N x G
        const T* weight, // C (DG) (C)
        const int HW,
        const int C,
        const int G,
        at::acc_type<T, true>* coef2,
        at::acc_type<T, true>* coef3) {
  using T_ACC = at::acc_type<T, true>;
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;
  const int HWD = HW * C / G;
  const int HWd = HW * C / THREADS_PER_BLOCK;
  const int c = threadIdx.x; // [bix][tx]
  const int ng = blockIdx.x * G + (threadIdx.x % G);

#pragma unroll
  for (int i = 0; i < HWd; ++i) {
    int index = blockIdx.x * HW*C + i * THREADS_PER_BLOCK + threadIdx.y * C + threadIdx.x; // [bix][i][ty][tx]
    T_ACC gamma_v = static_cast<T_ACC>(weight[c]);
    sum1 += static_cast<T_ACC>(dy[index]) * static_cast<T_ACC>(X[index]) * gamma_v; // sum1 = sum(X * gamma * dy)
    sum2 += static_cast<T_ACC>(dy[index]) * gamma_v; // sum1 = sum(X * gamma * dy)
  }

  __shared__ T_ACC reduced_sum1[THREADS_PER_BLOCK];
  __shared__ T_ACC reduced_sum2[THREADS_PER_BLOCK];
  sum_reduce(sum1, reduced_sum1, G);
  sum_reduce(sum2, reduced_sum2, G);

  if (threadIdx.y == 0 && (int)threadIdx.x < G) {
    sum1 = reduced_sum1[threadIdx.x];
    sum2 = reduced_sum2[threadIdx.x];

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
        at::acc_type<T, true>* dweight,
        at::acc_type<T, true>* dbias) {
  using T_ACC = at::acc_type<T, true>;
  T_ACC dweight_sum = 0;
  T_ACC dbias_sum = 0;

  const int c = THREADS_PER_BLOCK / C;
  const int Hw = HW / c;
  const int ng = blockIdx.x * G + threadIdx.x % G; // [bix][i][ty][tx]

#pragma unroll
  for (int i = 0; i < Hw; ++i) {
    int index = blockIdx.x * HW*C + i * THREADS_PER_BLOCK + threadIdx.y * C + threadIdx.x; // [bix][i][ty][tx]
    T_ACC dy_elem = static_cast<T_ACC>(dy[index]);
    dweight_sum += dy_elem * (static_cast<T_ACC>(X[index]) - static_cast<T_ACC>(means[ng])) * static_cast<T_ACC>(rstds[ng]);
    dbias_sum += dy_elem;
  }

  __shared__ T_ACC reduced_dweight[THREADS_PER_BLOCK];
  __shared__ T_ACC reduced_dbias[THREADS_PER_BLOCK];
  sum_reduce(dweight_sum, reduced_dweight, blockDim.x);
  sum_reduce(dbias_sum, reduced_dbias, blockDim.x);

  if (threadIdx.y == 0) {
    dweight[blockIdx.x * C + threadIdx.x] = reduced_dweight[threadIdx.x];
    dbias[blockIdx.x * C + threadIdx.x] = reduced_dbias[threadIdx.x];
  }
}

/*
template <typename T>
__global__ void
compute_backward_params_weight_bias(
        const T* dy,    // N x HWD x G
        const T* X,     // N x HWD x G
        const T* means, // N x G
        const T* rstds, // N x G
        const T* weight, // C (DG) (C)
        const int HW,
        const int C,
        const int G,
        at::acc_type<T, true>* coef2,
        at::acc_type<T, true>* coef3) {
  using T_ACC = at::acc_type<T, true>;
  T_ACC sum1 = 0;
  T_ACC sum2 = 0;
  const int HWD = HW * C / G;
  const int HWd = HW * C / THREADS_PER_BLOCK;
  const int nc = blockIdx.x * G + threadIdx.x; // [bix][tx]
  const int ng = blockIdx.x * G + (threadIdx.x % G);

#pragma unroll
  for (int i = 0; i < HWd; ++i) {
    int index = blockIdx.x * HW*C + i * THREADS_PER_BLOCK + threadIdx.y * C + threadIdx.x; // [bix][i][ty][tx]
    T_ACC gamma_v = static_cast<T_ACC>(weight[nc]);
    T_ACC dy_elem = static_cast<T_ACC>(dy[index]);
    T_ACC X_elem = static_cast<T_ACC>(X[index]);
    sum1 += dy_elem * X_elem * gamma_v; // sum1 = sum(X * gamma * dy)
    sum2 += dy_elem * gamma_v; // sum1 = sum(X * gamma * dy)

    //dweight_sum += dy_elem * (static_cast<T_ACC>(X[index]) - static_cast<T_ACC>(means[ng])) * static_cast<T_ACC>(rstds[ng]);
    //dbias_sum += dy_elem;
  }

  __shared__ T_ACC reduced_sum1[THREADS_PER_BLOCK];
  __shared__ T_ACC reduced_sum2[THREADS_PER_BLOCK];
  sum_reduce(sum1, reduced_sum1, G);
  sum_reduce(sum2, reduced_sum2, G);

  if (threadIdx.y == 0 && (int)threadIdx.x < G) {
    sum1 = reduced_sum1[threadIdx.x];
    sum2 = reduced_sum2[threadIdx.x];

    T_ACC mean = static_cast<T_ACC>(means[ng]);
    T_ACC rstd = static_cast<T_ACC>(rstds[ng]);
    T_ACC x = (sum2 * static_cast<T_ACC>(means[ng]) - sum1) * rstd * rstd * rstd / HWD; // AKA -sum(norm(X[ng])[i] * gamma[g] * dy[ngi], i) / std[ng]^2 / D
    coef2[ng] = x;
    float c3a = -x * mean; // AKA sum(norm(X[ngi]) * gamma[g] * dy[ngi] * mean[ng], i)
    float c3b = -sum2 * rstd / HWD; // AKA -gamma * sum(dy) / std[ng] / D
    coef3[ng] = c3a + c3b;
  }

  T_ACC dweight_sum = 0;
  T_ACC dbias_sum = 0;

  const int c = THREADS_PER_BLOCK / C;
  const int Hw = HW / c;

#pragma unroll
  for (int i = 0; i < Hw; ++i) {
    int index = blockIdx.x * HW*C + i * THREADS_PER_BLOCK + threadIdx.y * C + threadIdx.x; // [bix][i][ty][tx]
    T_ACC dy_elem = static_cast<T_ACC>(dy[index]);
    dweight_sum += dy_elem * (static_cast<T_ACC>(X[index]) - static_cast<T_ACC>(means[ng])) * static_cast<T_ACC>(rstds[ng]);
    dbias_sum += dy_elem;
  }

  __shared__ T_ACC reduced_dweight[THREADS_PER_BLOCK];
  __shared__ T_ACC reduced_dbias[THREADS_PER_BLOCK];
  sum_reduce(dweight_sum, reduced_dweight, blockDim.x);
  sum_reduce(dbias_sum, reduced_dbias, blockDim.x);

  if (threadIdx.y == 0) {
    dweight[blockIdx.x * C + threadIdx.x] = reduced_dweight[threadIdx.x];
    dbias[blockIdx.x * C + threadIdx.x] = reduced_dbias[threadIdx.x];
    //atomicAdd(dweight + threadIdx.x, reduced_dweight[threadIdx.x]);
    //atomicAdd(dbias + threadIdx.x, reduced_dbias[threadIdx.x]);
  }
}*/

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
  const int g = THREADS_PER_BLOCK / G;

  const at::ScalarType kAccType =
      (X_nhwc.scalar_type() == at::kHalf || X_nhwc.scalar_type() == at::kBFloat16)
      ? at::kFloat
      : X_nhwc.scalar_type();

  torch::Tensor dweight_tmp = torch::empty({N, C}, dweight.options().dtype(kAccType));
  torch::Tensor dbias_tmp = torch::empty({N, C}, dbias.options().dtype(kAccType));
  T_ACC* dweight_tmpdata = dweight_tmp.mutable_data_ptr<T_ACC>();
  T_ACC* dbias_tmpdata = dbias_tmp.mutable_data_ptr<T_ACC>();

  torch::Tensor coef2 = torch::empty({N, G}, X_nhwc.options().dtype(kAccType));
  torch::Tensor coef3 = torch::empty({N, G}, X_nhwc.options().dtype(kAccType));
  T_ACC* coef2_data = coef2.mutable_data_ptr<T_ACC>();
  T_ACC* coef3_data = coef3.mutable_data_ptr<T_ACC>();

  const dim3 dimGrid(N);
  const dim3 dimBlock(G, g);
  compute_backward_params<T><<<N, dim3(C, THREADS_PER_BLOCK / C)>>>(
      dy_data, X_data, mean_data, rstd_data, weight_data,
      H * W, C, G,
      coef2_data, coef3_data
  );

  at::TensorIterator iter = at::TensorIteratorConfig()
    .resize_outputs(false)
    .check_all_same_dtype(std::is_same<T, T_ACC>::value)
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
      T_ACC result1 = (coef2 * static_cast<T_ACC>(x)) + coef3; // fma
      return (c1 * static_cast<T_ACC>(dy)) + result1; // fma
    });

  backward_weight_bias<T><<<N, dim3(C, THREADS_PER_BLOCK / C)>>>(
      dy_data, X_data, mean_data, rstd_data, 
      H * W, C, G,
      dweight_tmpdata, dbias_tmpdata);

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
