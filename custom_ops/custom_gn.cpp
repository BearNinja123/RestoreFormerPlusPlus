#include <torch/extension.h>
#include <torch/torch.h>
//#include <iostream>
//
void GroupNormKernelImpl(
    const torch::Tensor& X,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps,
    torch::Tensor& Y,
    torch::Tensor& mean,
    torch::Tensor& rstd);

void GroupNormBackwardKernelImpl(
    const torch::Tensor& dY,
    const torch::Tensor& X,
    const torch::Tensor& mean,
    const torch::Tensor& rstd,
    const torch::Tensor& gamma,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    torch::Tensor& dX,
    torch::Tensor& dgamma,
    torch::Tensor& dbeta);

std::vector<torch::Tensor> gn_nhwc_cuda_forward(
    const torch::Tensor& X,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const int G,
    float eps);

std::vector<torch::Tensor> gn_nhwc_cuda_forward2(
    const torch::Tensor& X,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const int G,
    float eps);

std::vector<torch::Tensor> gn_nhwc_cuda_forward3(
    const torch::Tensor& X,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const int G,
    float eps);

std::vector<torch::Tensor> gn_nhwc_cuda_backward(
    const torch::Tensor& dy,
    const torch::Tensor& X,
    const torch::Tensor& weight,
    const torch::Tensor& means,
    const torch::Tensor& rstds,
    const int G);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
std::vector<torch::Tensor> gn_nhwc_forward(
    const torch::Tensor X,
    const torch::Tensor weight,
    const torch::Tensor bias,
    const int G,
    float eps) {
  CHECK_CUDA(X);
  CHECK_CUDA(weight);
  CHECK_CUDA(bias);
  return gn_nhwc_cuda_forward(X, weight, bias, G, eps);
}

std::vector<torch::Tensor> gn_nhwc_forward2(
    const torch::Tensor X,
    const torch::Tensor weight,
    const torch::Tensor bias,
    const int G,
    float eps) {
  CHECK_CUDA(X);
  CHECK_CUDA(weight);
  CHECK_CUDA(bias);
  return gn_nhwc_cuda_forward2(X, weight, bias, G, eps);
}

std::vector<torch::Tensor> gn_nhwc_forward3(
    const torch::Tensor X,
    const torch::Tensor weight,
    const torch::Tensor bias,
    const int G,
    float eps) {
  CHECK_CUDA(X);
  CHECK_CUDA(weight);
  CHECK_CUDA(bias);
  return gn_nhwc_cuda_forward3(X, weight, bias, G, eps);
}

std::vector<torch::Tensor> gn_nhwc_backward(
    const torch::Tensor dy,
    const torch::Tensor X,
    const torch::Tensor weight,
    const torch::Tensor means,
    const torch::Tensor rstds,
    const int G) {
  CHECK_CUDA(dy);
  CHECK_CUDA(X);
  CHECK_CUDA(weight);
  CHECK_CUDA(means);
  CHECK_CUDA(rstds);
  return gn_nhwc_cuda_backward(dy, X, weight, means, rstds, G);
}

std::vector<torch::Tensor> gn_nchw_forward(
    const torch::Tensor X,
    const torch::Tensor weight,
    const torch::Tensor bias,
    const int G,
    float eps) {
  CHECK_CUDA(X);
  CHECK_CUDA(weight);
  CHECK_CUDA(bias);
  const int N = X.size(0);
  const int C = X.size(1);
  const int H = X.size(2);
  const int W = X.size(3);
  torch::Tensor Y = torch::empty_like(X);
  torch::Tensor mean = torch::empty({N, G}, X.options());
  torch::Tensor rstd = torch::empty({N, G}, X.options());
  GroupNormKernelImpl(
    X, weight, bias,
    N, C, H * W,
    G,
    eps,
    Y,
    mean,
    rstd);
  return {Y, mean, rstd};
}

std::vector<torch::Tensor> gn_nchw_backward(
    const torch::Tensor dy,
    const torch::Tensor X,
    const torch::Tensor weight,
    const torch::Tensor means,
    const torch::Tensor rstds,
    const int G) {
  CHECK_CUDA(dy);
  CHECK_CUDA(X);
  CHECK_CUDA(weight);
  CHECK_CUDA(means);
  CHECK_CUDA(rstds);
  const int N = X.size(0);
  const int C = X.size(1);
  const int H = X.size(2);
  const int W = X.size(3);
  torch::Tensor dX = torch::empty_like(X);
  torch::Tensor dgamma = torch::empty_like(weight);
  torch::Tensor dbeta = torch::empty_like(weight);
  GroupNormBackwardKernelImpl(
      dy, X,
      means, rstds, weight,
      N, C, H * W, G,
      dX, dgamma, dbeta);

  return {dX, dgamma, dbeta};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gn_nhwc_forward, "GN NHWC forward");
  m.def("forward2", &gn_nhwc_forward2, "GN NHWC forward2");
  m.def("forward3", &gn_nhwc_forward3, "GN NHWC forward3");
  m.def("backward", &gn_nhwc_backward, "GN NHWC backward");
  m.def("nchwforward", &gn_nchw_forward, "GN NCHW forward");
  m.def("nchwbackward", &gn_nchw_backward, "GN NCHW backward");
}
