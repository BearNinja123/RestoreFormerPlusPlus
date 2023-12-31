#include <torch/extension.h>
#include <torch/torch.h>
//#include <iostream>

std::vector<torch::Tensor> gn_nhwc_cuda_forward(
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gn_nhwc_forward, "GN NHWC backward");
  m.def("backward", &gn_nhwc_backward, "GN NHWC backward");
}
