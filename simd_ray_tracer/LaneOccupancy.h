#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "RenderTypes.h"

namespace simd8 {

struct LaneOccupancy {
  static constexpr std::size_t kLaneCount = 8;
  static constexpr std::size_t kNative4LaneCount = 4;

  explicit LaneOccupancy(const render::RenderConfig& config)
      : ray_active_by_step(static_cast<std::size_t>(config.max_steps)),
        ray_packets_by_step(static_cast<std::size_t>(config.max_steps)),
        native4_ray_active_by_step(static_cast<std::size_t>(config.max_steps)),
        native4_ray_packets_by_step(static_cast<std::size_t>(config.max_steps)),
        sdf_active_by_iteration(
            static_cast<std::size_t>(config.mandelbulb_iterations)),
        sdf_packets_by_iteration(
            static_cast<std::size_t>(config.mandelbulb_iterations)),
        native4_sdf_active_by_iteration(
            static_cast<std::size_t>(config.mandelbulb_iterations)),
        native4_sdf_packets_by_iteration(
            static_cast<std::size_t>(config.mandelbulb_iterations)) {}

  std::vector<std::uint64_t> ray_active_by_step;
  std::vector<std::uint64_t> ray_packets_by_step;
  std::vector<std::uint64_t> native4_ray_active_by_step;
  std::vector<std::uint64_t> native4_ray_packets_by_step;
  std::vector<std::uint64_t> sdf_active_by_iteration;
  std::vector<std::uint64_t> sdf_packets_by_iteration;
  std::vector<std::uint64_t> native4_sdf_active_by_iteration;
  std::vector<std::uint64_t> native4_sdf_packets_by_iteration;
  std::array<std::uint64_t, kLaneCount + 1> ray_active_histogram{};
  std::array<std::uint64_t, kLaneCount + 1> sdf_active_histogram{};

  std::uint64_t ray_packets = 0;
  std::uint64_t sdf_calls = 0;
  std::uint64_t hit_packets = 0;
  std::uint64_t native4_hit_packets = 0;
  std::uint64_t hit_lanes = 0;
  std::uint64_t miss_lanes = 0;
  std::uint64_t max_step_lanes = 0;
  std::uint64_t normal_batches = 0;
  std::uint64_t normal_hit_lanes = 0;
  std::uint64_t native4_normal_batches = 0;
};

}  // namespace simd8
