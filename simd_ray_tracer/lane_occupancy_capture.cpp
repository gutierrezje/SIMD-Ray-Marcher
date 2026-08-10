#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "Camera.h"
#include "LaneOccupancy.h"
#include "RenderTypes.h"
#include "SimdBackend.h"

namespace {

std::uint64_t sum(const std::vector<std::uint64_t>& values) {
  std::uint64_t total = 0;
  for (const std::uint64_t value : values) {
    total += value;
  }
  return total;
}

double ratio(std::uint64_t useful, std::uint64_t issued) {
  if (issued == 0) {
    return 0.0;
  }
  return static_cast<double>(useful) / static_cast<double>(issued);
}

void write_stage(std::ofstream& output, const char* stage,
                 const std::vector<std::uint64_t>& active,
                 const std::vector<std::uint64_t>& packets,
                 std::size_t lane_count) {
  for (std::size_t index = 0; index < active.size(); index += 1) {
    const std::uint64_t issued = packets[index] * lane_count;
    output << stage << ',' << index << ',' << packets[index] << ','
           << active[index] << ',' << issued << ','
           << ratio(active[index], issued) << '\n';
  }
}

void write_histogram(
    std::ofstream& output, const char* name,
    const std::array<std::uint64_t, simd8::LaneOccupancy::kLaneCount + 1>&
        histogram) {
  output << name << '=';
  for (std::size_t lanes = 0; lanes < histogram.size(); lanes += 1) {
    if (lanes != 0) {
      output << ',';
    }
    output << lanes << ':' << histogram[lanes];
  }
  output << '\n';
}

bool write_results(const std::filesystem::path& directory,
                   const render::RenderConfig& config,
                   const simd8::LaneOccupancy& occupancy) {
  std::filesystem::create_directories(directory);

  std::ofstream csv(directory / "lane-occupancy.csv");
  csv << std::fixed << std::setprecision(6);
  csv << "stage,iteration,packet_iterations,active_lane_iterations,"
         "issued_lane_slots,occupancy\n";
  write_stage(csv, "simde_avx8_ray_march", occupancy.ray_active_by_step,
              occupancy.ray_packets_by_step, simd8::LaneOccupancy::kLaneCount);
  write_stage(csv, "native4_model_ray_march",
              occupancy.native4_ray_active_by_step,
              occupancy.native4_ray_packets_by_step,
              simd8::LaneOccupancy::kNative4LaneCount);
  write_stage(csv, "simde_avx8_mandelbulb", occupancy.sdf_active_by_iteration,
              occupancy.sdf_packets_by_iteration,
              simd8::LaneOccupancy::kLaneCount);
  write_stage(csv, "native4_model_mandelbulb",
              occupancy.native4_sdf_active_by_iteration,
              occupancy.native4_sdf_packets_by_iteration,
              simd8::LaneOccupancy::kNative4LaneCount);

  const std::uint64_t ray_packet_iterations =
      sum(occupancy.ray_packets_by_step);
  const std::uint64_t ray_active = sum(occupancy.ray_active_by_step);
  const std::uint64_t ray_issued =
      ray_packet_iterations * simd8::LaneOccupancy::kLaneCount;
  const std::uint64_t sdf_packet_iterations =
      sum(occupancy.sdf_packets_by_iteration);
  const std::uint64_t sdf_active = sum(occupancy.sdf_active_by_iteration);
  const std::uint64_t sdf_issued =
      sdf_packet_iterations * simd8::LaneOccupancy::kLaneCount;
  const std::uint64_t native4_ray_packet_iterations =
      sum(occupancy.native4_ray_packets_by_step);
  const std::uint64_t native4_ray_active =
      sum(occupancy.native4_ray_active_by_step);
  const std::uint64_t native4_ray_issued =
      native4_ray_packet_iterations * simd8::LaneOccupancy::kNative4LaneCount;
  const std::uint64_t native4_sdf_packet_iterations =
      sum(occupancy.native4_sdf_packets_by_iteration);
  const std::uint64_t native4_sdf_active =
      sum(occupancy.native4_sdf_active_by_iteration);
  const std::uint64_t native4_sdf_issued =
      native4_sdf_packet_iterations * simd8::LaneOccupancy::kNative4LaneCount;
  const std::uint64_t normal_issued =
      occupancy.normal_batches * simd8::LaneOccupancy::kLaneCount;
  const std::uint64_t native4_normal_issued =
      occupancy.native4_normal_batches *
      simd8::LaneOccupancy::kNative4LaneCount;
  const std::uint64_t native4_deferred_normal_issued =
      occupancy.native4_hit_packets * simd8::LaneOccupancy::kNative4LaneCount;
  const std::uint64_t normal_sdf_calls = occupancy.normal_batches * 6;

  std::ofstream summary(directory / "lane-occupancy-summary.txt");
  summary << std::fixed << std::setprecision(6);
  summary << "backend=simde-avx2\n"
          << "packet_width=" << simd8::LaneOccupancy::kLaneCount << '\n'
          << "native4_model_status=counterfactual_not_measured\n"
          << "native4_model_scope=simd_lane_slots_with_independent_halves\n"
          << "native4_model_excludes=loop_tests_branches_scheduling_runtime\n"
          << "resolution=" << config.width << 'x' << config.height << '\n'
          << "ray_packets=" << occupancy.ray_packets << '\n'
          << "ray_packet_iterations=" << ray_packet_iterations << '\n'
          << "ray_active_lane_iterations=" << ray_active << '\n'
          << "ray_issued_lane_slots=" << ray_issued << '\n'
          << "ray_occupancy=" << ratio(ray_active, ray_issued) << '\n'
          << "native4_model_ray_packet_iterations="
          << native4_ray_packet_iterations << '\n'
          << "native4_model_ray_active_lane_iterations=" << native4_ray_active
          << '\n'
          << "native4_model_ray_issued_lane_slots=" << native4_ray_issued
          << '\n'
          << "native4_model_ray_lane_utilization="
          << ratio(native4_ray_active, native4_ray_issued) << '\n'
          << "sdf_calls=" << occupancy.sdf_calls << '\n'
          << "sdf_packet_iterations=" << sdf_packet_iterations << '\n'
          << "sdf_active_lane_iterations=" << sdf_active << '\n'
          << "sdf_issued_lane_slots=" << sdf_issued << '\n'
          << "sdf_occupancy=" << ratio(sdf_active, sdf_issued) << '\n'
          << "native4_model_sdf_packet_iterations="
          << native4_sdf_packet_iterations << '\n'
          << "native4_model_sdf_active_lane_iterations=" << native4_sdf_active
          << '\n'
          << "native4_model_sdf_issued_lane_slots=" << native4_sdf_issued
          << '\n'
          << "native4_model_sdf_lane_utilization="
          << ratio(native4_sdf_active, native4_sdf_issued) << '\n'
          << "hit_packets=" << occupancy.hit_packets << '\n'
          << "native4_model_hit_packets=" << occupancy.native4_hit_packets
          << '\n'
          << "hit_lanes=" << occupancy.hit_lanes << '\n'
          << "miss_lanes=" << occupancy.miss_lanes << '\n'
          << "max_step_lanes=" << occupancy.max_step_lanes << '\n'
          << "normal_batches=" << occupancy.normal_batches << '\n'
          << "normal_sdf_calls=" << normal_sdf_calls << '\n'
          << "normal_sdf_call_fraction="
          << ratio(normal_sdf_calls, occupancy.sdf_calls) << '\n'
          << "normal_hit_lanes=" << occupancy.normal_hit_lanes << '\n'
          << "normal_issued_lane_slots=" << normal_issued << '\n'
          << "normal_batch_occupancy="
          << ratio(occupancy.normal_hit_lanes, normal_issued) << '\n';
  summary << "native4_model_normal_batches=" << occupancy.native4_normal_batches
          << '\n'
          << "native4_model_normal_issued_lane_slots=" << native4_normal_issued
          << '\n'
          << "native4_model_normal_lane_utilization="
          << ratio(occupancy.normal_hit_lanes, native4_normal_issued) << '\n'
          << "native4_model_deferred_normal_issued_lane_slots="
          << native4_deferred_normal_issued << '\n'
          << "native4_model_deferred_normal_lane_utilization="
          << ratio(occupancy.normal_hit_lanes, native4_deferred_normal_issued)
          << '\n';
  write_histogram(summary, "ray_active_lane_histogram",
                  occupancy.ray_active_histogram);
  write_histogram(summary, "sdf_active_lane_histogram",
                  occupancy.sdf_active_histogram);

  return csv.good() && summary.good();
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " OUTPUT_DIRECTORY\n";
    return 2;
  }

  constexpr render::RenderConfig config = render::kDefaultRenderConfig;
  const Camera camera = make_default_camera();
  render::Image image(config);
  simd8::LaneOccupancy occupancy(config);
  simd8::render_with_lane_occupancy(camera, config, image, occupancy);

  if (!write_results(argv[1], config, occupancy)) {
    std::cerr << "Failed to write lane occupancy results\n";
    return 1;
  }

  std::cout << "Wrote lane occupancy results to " << argv[1] << '\n';
  return 0;
}
