// Python binding and input validation for the fused CUDA Vecchia covariance.
//
// The kernels live in vecchia_covariance_cuda_kernel.cu so that this file can
// be compiled by the host C++ compiler.  The supported statistical model is
// intentionally identical to vecchia_covariance_cpu.cpp: float64 tensors,
// seven covariance parameters, fixed coordinates, and Matern smoothness 1/2.

#include <torch/extension.h>

namespace {

void check_common_inputs(
    const torch::Tensor& params,
    const torch::Tensor& coordinates,
    const torch::Tensor& dummy) {
  TORCH_CHECK(params.is_cuda(), "params must be on CUDA");
  TORCH_CHECK(coordinates.is_cuda(), "coordinates must be on CUDA");
  TORCH_CHECK(dummy.is_cuda(), "dummy must be on CUDA");
  TORCH_CHECK(
      params.device() == coordinates.device() && params.device() == dummy.device(),
      "params, coordinates, and dummy must be on the same CUDA device");
  TORCH_CHECK(params.scalar_type() == torch::kFloat64, "params must be float64");
  TORCH_CHECK(
      coordinates.scalar_type() == torch::kFloat64,
      "coordinates must be float64");
  TORCH_CHECK(dummy.scalar_type() == torch::kBool, "dummy must be bool");
  TORCH_CHECK(
      params.dim() == 1 && params.numel() == 7,
      "params must have shape (7,)");
  TORCH_CHECK(
      coordinates.dim() == 3 && coordinates.size(2) == 3,
      "coordinates must have shape (batch, points, 3)");
  TORCH_CHECK(
      dummy.dim() == 2 && dummy.size(0) == coordinates.size(0) &&
          dummy.size(1) == coordinates.size(1),
      "dummy must have shape (batch, points)");
  TORCH_CHECK(params.is_contiguous(), "params must be contiguous");
  TORCH_CHECK(coordinates.is_contiguous(), "coordinates must be contiguous");
  TORCH_CHECK(dummy.is_contiguous(), "dummy must be contiguous");
}

}  // namespace

torch::Tensor covariance_cuda_forward(
    const torch::Tensor& params,
    const torch::Tensor& coordinates,
    const torch::Tensor& dummy);

torch::Tensor covariance_cuda_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& params,
    const torch::Tensor& coordinates,
    const torch::Tensor& dummy);

torch::Tensor covariance_forward(
    const torch::Tensor& params,
    const torch::Tensor& coordinates,
    const torch::Tensor& dummy) {
  check_common_inputs(params, coordinates, dummy);
  return covariance_cuda_forward(params, coordinates, dummy);
}

torch::Tensor covariance_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& params,
    const torch::Tensor& coordinates,
    const torch::Tensor& dummy) {
  check_common_inputs(params, coordinates, dummy);
  TORCH_CHECK(grad_output.is_cuda(), "grad_output must be on CUDA");
  TORCH_CHECK(
      grad_output.device() == params.device(),
      "grad_output and params must be on the same CUDA device");
  TORCH_CHECK(
      grad_output.scalar_type() == torch::kFloat64,
      "grad_output must be float64");
  TORCH_CHECK(
      grad_output.dim() == 3 &&
          grad_output.size(0) == coordinates.size(0) &&
          grad_output.size(1) == coordinates.size(1) &&
          grad_output.size(2) == coordinates.size(1),
      "grad_output must have shape (batch, points, points)");
  TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
  return covariance_cuda_backward(grad_output, params, coordinates, dummy);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.doc() = "Fused CUDA covariance assembly for smooth=0.5 Vecchia models";
  module.def("forward", &covariance_forward, "Fused CUDA covariance forward pass");
  module.def(
      "backward",
      &covariance_backward,
      "Fused CUDA covariance parameter gradient");
}
