// Fused CUDA covariance assembly and analytic first-order parameter gradient.
//
// This is the CUDA counterpart of vecchia_covariance_cpu.cpp.  It evaluates
// one triangle of each symmetric matrix instead of materializing temporary
// advection, pairwise-distance, correlation, and dummy-mask tensors.  The
// backward kernel produces one seven-parameter partial sum per CUDA block;
// ATen performs the small final reduction on the current stream.  No
// parameter-gradient atomics are needed.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int kThreads = 256;
constexpr int kTileWidth = 16;
constexpr int kParameterCount = 7;
constexpr double kCholeskyJitter = 1.0e-6;

__device__ __forceinline__ void load_parameters(
    const double* params,
    double& phi1,
    double& phi2,
    double& phi3,
    double& phi4,
    double& advection_latitude,
    double& advection_longitude,
    double& nugget,
    double& signal_variance) {
  phi1 = exp(params[0]);
  phi2 = exp(params[1]);
  phi3 = exp(params[2]);
  phi4 = exp(params[3]);
  advection_latitude = params[4];
  advection_longitude = params[5];
  nugget = exp(params[6]);
  signal_variance = phi1 / phi2;
}

__device__ __forceinline__ void pair_geometry(
    const double* coordinates,
    int64_t batch,
    int64_t point_i,
    int64_t point_j,
    int64_t points,
    double phi3,
    double phi4,
    double advection_latitude,
    double advection_longitude,
    double& latitude_difference,
    double& longitude_difference,
    double& time_difference,
    double& distance,
    bool& distance_has_gradient) {
  const int64_t offset_i = (batch * points + point_i) * 3;
  const int64_t offset_j = (batch * points + point_j) * 3;
  const double time_i = coordinates[offset_i + 2];
  const double time_j = coordinates[offset_j + 2];
  time_difference = time_i - time_j;
  const double advected_latitude_i =
      coordinates[offset_i] - advection_latitude * time_i;
  const double advected_latitude_j =
      coordinates[offset_j] - advection_latitude * time_j;
  const double advected_longitude_i =
      coordinates[offset_i + 1] - advection_longitude * time_i;
  const double advected_longitude_j =
      coordinates[offset_j + 1] - advection_longitude * time_j;
  latitude_difference = advected_latitude_i - advected_latitude_j;
  longitude_difference = advected_longitude_i - advected_longitude_j;

  const double squared_distance =
      phi3 * latitude_difference * latitude_difference +
      longitude_difference * longitude_difference +
      phi4 * time_difference * time_difference;
  // Match the Torch and CPU stable-square-root convention exactly: a true
  // zero remains zero; a positive sub-epsilon value is evaluated at epsilon
  // and has zero derivative through the clamp.
  distance = squared_distance > 0.0
      ? sqrt(fmax(squared_distance, DBL_EPSILON))
      : 0.0;
  distance_has_gradient = squared_distance > DBL_EPSILON;
}

__global__ void covariance_forward_kernel(
    const double* params,
    const double* coordinates,
    const bool* dummy,
    double* covariance,
    int64_t points,
    int64_t row_tiles) {
  __shared__ double shared_params[kParameterCount];
  if (threadIdx.x < kParameterCount) {
    shared_params[threadIdx.x] = params[threadIdx.x];
  }
  __syncthreads();

  double phi1;
  double phi2;
  double phi3;
  double phi4;
  double advection_latitude;
  double advection_longitude;
  double nugget;
  double signal_variance;
  load_parameters(
      shared_params,
      phi1,
      phi2,
      phi3,
      phi4,
      advection_latitude,
      advection_longitude,
      nugget,
      signal_variance);

  const int64_t flat_block = static_cast<int64_t>(blockIdx.x);
  const int64_t batch = flat_block / row_tiles;
  const int64_t row_tile = flat_block % row_tiles;
  const int64_t point_i =
      row_tile * kTileWidth + threadIdx.x / kTileWidth;

  // Each block owns one 16-row tile and walks only the column tiles on or
  // below its diagonal.  Every off-diagonal pair is evaluated once and copied
  // to its transpose, halving the expensive double-precision exp/geometry
  // work relative to a full N-by-N launch.
  for (int64_t column_tile = 0; column_tile <= row_tile; ++column_tile) {
    const int64_t point_j =
        column_tile * kTileWidth + threadIdx.x % kTileWidth;
    if (point_i >= points || point_j >= points || point_j > point_i) {
      continue;
    }

    const int64_t lower_index =
        (batch * points + point_i) * points + point_j;
    const int64_t upper_index =
        (batch * points + point_j) * points + point_i;
    const bool dummy_i = dummy[batch * points + point_i];
    const bool dummy_j = dummy[batch * points + point_j];
    double value;

    if (dummy_i || dummy_j) {
      value = (point_i == point_j && dummy_i) ? 1.0 : 0.0;
    } else {
      double latitude_difference;
      double longitude_difference;
      double time_difference;
      double distance;
      bool distance_has_gradient;
      pair_geometry(
          coordinates,
          batch,
          point_i,
          point_j,
          points,
          phi3,
          phi4,
          advection_latitude,
          advection_longitude,
          latitude_difference,
          longitude_difference,
          time_difference,
          distance,
          distance_has_gradient);
      value = signal_variance * exp(-phi2 * distance);
      if (point_i == point_j) {
        value += nugget + kCholeskyJitter;
      }
    }
    covariance[lower_index] = value;
    covariance[upper_index] = value;
  }
}

__global__ void covariance_backward_kernel(
    const double* grad_output,
    const double* params,
    const double* coordinates,
    const bool* dummy,
    double* partial_gradients,
    int64_t points,
    int64_t row_tiles) {
  __shared__ double shared_params[kParameterCount];
  __shared__ double shared_gradients[kParameterCount][kThreads];
  if (threadIdx.x < kParameterCount) {
    shared_params[threadIdx.x] = params[threadIdx.x];
  }
  __syncthreads();

  double phi1;
  double phi2;
  double phi3;
  double phi4;
  double advection_latitude;
  double advection_longitude;
  double nugget;
  double signal_variance;
  load_parameters(
      shared_params,
      phi1,
      phi2,
      phi3,
      phi4,
      advection_latitude,
      advection_longitude,
      nugget,
      signal_variance);

  double local[kParameterCount] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const int64_t flat_block = static_cast<int64_t>(blockIdx.x);
  const int64_t batch = flat_block / row_tiles;
  const int64_t row_tile = flat_block % row_tiles;
  const int64_t point_i =
      row_tile * kTileWidth + threadIdx.x / kTileWidth;

  for (int64_t column_tile = 0; column_tile <= row_tile; ++column_tile) {
    const int64_t point_j =
        column_tile * kTileWidth + threadIdx.x % kTileWidth;
    if (point_i >= points || point_j >= points || point_j > point_i) {
      continue;
    }
    if (dummy[batch * points + point_i] ||
        dummy[batch * points + point_j]) {
      continue;
    }

    double latitude_difference;
    double longitude_difference;
    double time_difference;
    double distance;
    bool distance_has_gradient;
    pair_geometry(
        coordinates,
        batch,
        point_i,
        point_j,
        points,
        phi3,
        phi4,
        advection_latitude,
        advection_longitude,
        latitude_difference,
        longitude_difference,
        time_difference,
        distance,
        distance_has_gradient);

    const double scaled_distance = phi2 * distance;
    const double covariance = signal_variance * exp(-scaled_distance);
    const int64_t lower_index =
        (batch * points + point_i) * points + point_j;
    const int64_t upper_index =
        (batch * points + point_j) * points + point_i;
    const double upstream = grad_output[lower_index] +
        (point_i == point_j ? 0.0 : grad_output[upper_index]);
    local[0] += upstream * covariance;
    local[1] += upstream * covariance * (-1.0 - scaled_distance);

    if (distance_has_gradient) {
      const double common = -0.5 * phi2 * covariance / distance;
      local[2] += upstream * common * phi3 *
          latitude_difference * latitude_difference;
      local[3] += upstream * common * phi4 *
          time_difference * time_difference;
      local[4] += upstream * phi2 * covariance * phi3 *
          latitude_difference * time_difference / distance;
      local[5] += upstream * phi2 * covariance *
          longitude_difference * time_difference / distance;
    }
    if (point_i == point_j) {
      local[6] += upstream * nugget;
    }
  }

#pragma unroll
  for (int parameter = 0; parameter < kParameterCount; ++parameter) {
    shared_gradients[parameter][threadIdx.x] = local[parameter];
  }
  __syncthreads();

  for (int offset = kThreads / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
#pragma unroll
      for (int parameter = 0; parameter < kParameterCount; ++parameter) {
        shared_gradients[parameter][threadIdx.x] +=
            shared_gradients[parameter][threadIdx.x + offset];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x < kParameterCount) {
    partial_gradients[flat_block * kParameterCount + threadIdx.x] =
        shared_gradients[threadIdx.x][0];
  }
}

int launch_block_count(int64_t batches, int64_t row_tiles) {
  const int64_t required = batches * row_tiles;
  TORCH_CHECK(
      required <= std::numeric_limits<int>::max(),
      "covariance batch is too large for one CUDA launch");
  return static_cast<int>(required);
}

}  // namespace

torch::Tensor covariance_cuda_forward(
    const torch::Tensor& params,
    const torch::Tensor& coordinates,
    const torch::Tensor& dummy) {
  const c10::cuda::CUDAGuard device_guard(params.device());
  const int64_t batches = coordinates.size(0);
  const int64_t points = coordinates.size(1);
  auto covariance = torch::empty({batches, points, points}, params.options());
  const int64_t total = batches * points * points;
  if (total == 0) {
    return covariance;
  }

  const int64_t row_tiles = (points + kTileWidth - 1) / kTileWidth;
  const int blocks = launch_block_count(batches, row_tiles);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(params.get_device());
  covariance_forward_kernel<<<blocks, kThreads, 0, stream>>>(
      params.const_data_ptr<double>(),
      coordinates.const_data_ptr<double>(),
      dummy.const_data_ptr<bool>(),
      covariance.data_ptr<double>(),
      points,
      row_tiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return covariance;
}

torch::Tensor covariance_cuda_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& params,
    const torch::Tensor& coordinates,
    const torch::Tensor& dummy) {
  const c10::cuda::CUDAGuard device_guard(params.device());
  const int64_t batches = coordinates.size(0);
  const int64_t points = coordinates.size(1);
  const int64_t total = batches * points * points;
  if (total == 0) {
    return torch::zeros({kParameterCount}, params.options());
  }

  const int64_t row_tiles = (points + kTileWidth - 1) / kTileWidth;
  const int blocks = launch_block_count(batches, row_tiles);
  auto partial_gradients =
      torch::empty({blocks, kParameterCount}, params.options());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(params.get_device());
  covariance_backward_kernel<<<blocks, kThreads, 0, stream>>>(
      grad_output.const_data_ptr<double>(),
      params.const_data_ptr<double>(),
      coordinates.const_data_ptr<double>(),
      dummy.const_data_ptr<bool>(),
      partial_gradients.data_ptr<double>(),
      points,
      row_tiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return partial_gradients.sum(0);
}
