// Fused CPU covariance assembly for the smooth=0.5 grouped Vecchia model.
//
// This file intentionally implements only the narrow hot path used by the
// corridor likelihood: float64 CPU tensors, a seven-parameter covariance,
// and a fixed exponential (Matern nu=1/2) correlation.  Ordering and
// conditioning geometry remain in Python and are supplied as fixed
// coordinates and a padding mask.

#include <torch/extension.h>

#include <ATen/Parallel.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

void check_inputs(
    const torch::Tensor& params,
    const torch::Tensor& coordinates,
    const torch::Tensor& dummy) {
  TORCH_CHECK(params.device().is_cpu(), "params must be on CPU");
  TORCH_CHECK(coordinates.device().is_cpu(), "coordinates must be on CPU");
  TORCH_CHECK(dummy.device().is_cpu(), "dummy must be on CPU");
  TORCH_CHECK(params.scalar_type() == torch::kFloat64, "params must be float64");
  TORCH_CHECK(
      coordinates.scalar_type() == torch::kFloat64,
      "coordinates must be float64");
  TORCH_CHECK(dummy.scalar_type() == torch::kBool, "dummy must be bool");
  TORCH_CHECK(params.dim() == 1 && params.numel() == 7, "params must have shape (7,)");
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

struct CovarianceParameters {
  double phi1;
  double phi2;
  double phi3;
  double phi4;
  double advection_latitude;
  double advection_longitude;
  double nugget;
  double signal_variance;
};

CovarianceParameters unpack_parameters(const double* params) {
  const double phi1 = std::exp(params[0]);
  const double phi2 = std::exp(params[1]);
  return CovarianceParameters{
      phi1,
      phi2,
      std::exp(params[2]),
      std::exp(params[3]),
      params[4],
      params[5],
      std::exp(params[6]),
      phi1 / phi2};
}

torch::Tensor advected_coordinates(
    const torch::Tensor& coordinates,
    const CovarianceParameters& parameters) {
  auto transformed = torch::empty_like(coordinates);
  const double* source = coordinates.const_data_ptr<double>();
  double* destination = transformed.data_ptr<double>();
  const int64_t point_count = coordinates.size(0) * coordinates.size(1);
  at::parallel_for(0, point_count, 1024, [&](int64_t begin, int64_t end) {
    for (int64_t point = begin; point < end; ++point) {
      const int64_t offset = point * 3;
      const double time = source[offset + 2];
      destination[offset] =
          source[offset] - parameters.advection_latitude * time;
      destination[offset + 1] =
          source[offset + 1] - parameters.advection_longitude * time;
      destination[offset + 2] = time;
    }
  });
  return transformed;
}

inline void pair_geometry(
    const double* transformed_coordinate_data,
    int64_t batch,
    int64_t point_i,
    int64_t point_j,
    int64_t points,
    const CovarianceParameters& parameters,
    double& latitude_difference,
    double& longitude_difference,
    double& time_difference,
    double& distance,
    bool& distance_has_gradient) {
  const int64_t offset_i = (batch * points + point_i) * 3;
  const int64_t offset_j = (batch * points + point_j) * 3;

  time_difference =
      transformed_coordinate_data[offset_i + 2] -
      transformed_coordinate_data[offset_j + 2];
  latitude_difference =
      transformed_coordinate_data[offset_i] - transformed_coordinate_data[offset_j];
  longitude_difference =
      transformed_coordinate_data[offset_i + 1] -
      transformed_coordinate_data[offset_j + 1];

  const double squared_distance =
      parameters.phi3 * latitude_difference * latitude_difference +
      longitude_difference * longitude_difference +
      parameters.phi4 * time_difference * time_difference;
  // Match _stable_sqrt_distance exactly: zero remains zero, while any
  // positive sub-epsilon squared distance is evaluated at float64 epsilon.
  distance = squared_distance > 0.0
      ? std::sqrt(std::max(squared_distance, std::numeric_limits<double>::epsilon()))
      : 0.0;
  distance_has_gradient =
      squared_distance > std::numeric_limits<double>::epsilon();
}

torch::Tensor covariance_forward(
    const torch::Tensor& params,
    const torch::Tensor& coordinates,
    const torch::Tensor& dummy) {
  check_inputs(params, coordinates, dummy);

  const int64_t batches = coordinates.size(0);
  const int64_t points = coordinates.size(1);
  auto covariance = torch::empty({batches, points, points}, params.options());

  const double* parameter_data = params.const_data_ptr<double>();
  const CovarianceParameters parameters = unpack_parameters(parameter_data);
  const auto transformed = advected_coordinates(coordinates, parameters);
  const double* coordinate_data = transformed.const_data_ptr<double>();
  const bool* dummy_data = dummy.const_data_ptr<bool>();
  double* covariance_data = covariance.data_ptr<double>();
  const int64_t row_count = batches * points;

  // Covariance is symmetric.  Evaluate each pair once and fill both entries.
  at::parallel_for(0, row_count, 32, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const int64_t point_i = row % points;
      const int64_t batch = row / points;
      const int64_t mask_i = batch * points + point_i;

      for (int64_t point_j = 0; point_j <= point_i; ++point_j) {
        const int64_t mask_j = batch * points + point_j;
        const int64_t lower_index = (row * points) + point_j;
        const int64_t upper_index =
            ((batch * points + point_j) * points) + point_i;
        double value;

        if (dummy_data[mask_i] || dummy_data[mask_j]) {
          value = (point_i == point_j && dummy_data[mask_i]) ? 1.0 : 0.0;
        } else {
          double latitude_difference;
          double longitude_difference;
          double time_difference;
          double distance;
          bool distance_has_gradient;
          pair_geometry(
              coordinate_data,
              batch,
              point_i,
              point_j,
              points,
              parameters,
              latitude_difference,
              longitude_difference,
              time_difference,
              distance,
              distance_has_gradient);
          value = parameters.signal_variance * std::exp(-parameters.phi2 * distance);
          if (point_i == point_j) {
            value += parameters.nugget + 1.0e-6;
          }
        }
        covariance_data[lower_index] = value;
        covariance_data[upper_index] = value;
      }
    }
  });

  return covariance;
}

torch::Tensor covariance_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& params,
    const torch::Tensor& coordinates,
    const torch::Tensor& dummy) {
  check_inputs(params, coordinates, dummy);
  TORCH_CHECK(grad_output.device().is_cpu(), "grad_output must be on CPU");
  TORCH_CHECK(
      grad_output.scalar_type() == torch::kFloat64,
      "grad_output must be float64");
  TORCH_CHECK(
      grad_output.dim() == 3 && grad_output.size(0) == coordinates.size(0) &&
          grad_output.size(1) == coordinates.size(1) &&
          grad_output.size(2) == coordinates.size(1),
      "grad_output must have shape (batch, points, points)");
  TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");

  const int64_t batches = coordinates.size(0);
  const int64_t points = coordinates.size(1);
  const int64_t row_count = batches * points;
  const int64_t thread_count = std::max<int64_t>(1, at::get_num_threads());
  auto partial_gradients = torch::zeros({thread_count, 7}, params.options());

  const double* parameter_data = params.const_data_ptr<double>();
  const CovarianceParameters parameters = unpack_parameters(parameter_data);
  const auto transformed = advected_coordinates(coordinates, parameters);
  const double* coordinate_data = transformed.const_data_ptr<double>();
  const bool* dummy_data = dummy.const_data_ptr<bool>();
  const double* output_gradient_data = grad_output.const_data_ptr<double>();
  double* partial_data = partial_gradients.data_ptr<double>();

  at::parallel_for(0, row_count, 32, [&](int64_t begin, int64_t end) {
    const int64_t thread_id = std::max<int64_t>(0, at::get_thread_num());
    std::array<double, 7> local{};

    for (int64_t row = begin; row < end; ++row) {
      const int64_t point_i = row % points;
      const int64_t batch = row / points;
      const int64_t mask_i = batch * points + point_i;

      for (int64_t point_j = 0; point_j <= point_i; ++point_j) {
        const int64_t mask_j = batch * points + point_j;
        if (dummy_data[mask_i] || dummy_data[mask_j]) {
          continue;
        }

        const int64_t lower_index = (row * points) + point_j;
        const int64_t upper_index =
            ((batch * points + point_j) * points) + point_i;
        const double upstream = output_gradient_data[lower_index] +
            (point_i == point_j ? 0.0 : output_gradient_data[upper_index]);
        double latitude_difference;
        double longitude_difference;
        double time_difference;
        double distance;
        bool distance_has_gradient;
        pair_geometry(
            coordinate_data,
            batch,
            point_i,
            point_j,
            points,
            parameters,
            latitude_difference,
            longitude_difference,
            time_difference,
            distance,
            distance_has_gradient);

        const double scaled_distance = parameters.phi2 * distance;
        const double covariance =
            parameters.signal_variance * std::exp(-scaled_distance);
        local[0] += upstream * covariance;
        local[1] += upstream * covariance * (-1.0 - scaled_distance);

        if (distance_has_gradient) {
          const double common = -0.5 * parameters.phi2 * covariance / distance;
          local[2] += upstream * common * parameters.phi3 *
              latitude_difference * latitude_difference;
          local[3] += upstream * common * parameters.phi4 *
              time_difference * time_difference;
          local[4] += upstream * parameters.phi2 * covariance * parameters.phi3 *
              latitude_difference * time_difference / distance;
          local[5] += upstream * parameters.phi2 * covariance *
              longitude_difference * time_difference / distance;
        }

        if (point_i == point_j) {
          local[6] += upstream * parameters.nugget;
        }
      }
    }

    double* destination = partial_data + thread_id * 7;
    for (int parameter = 0; parameter < 7; ++parameter) {
      destination[parameter] += local[parameter];
    }
  });

  return partial_gradients.sum(0);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.doc() = "Fused CPU covariance assembly for smooth=0.5 Vecchia models";
  module.def("forward", &covariance_forward, "Fused covariance forward pass");
  module.def("backward", &covariance_backward, "Fused covariance parameter gradient");
}
