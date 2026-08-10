#include "SimdBackend.h"

#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#ifdef __GNUC__
#include "avx_mathfun.h"
#define _mm256_log_ps(x) log256_ps(x)
#endif

namespace simd8 {

namespace {

bool has_nan(__m256 v) {
  __m256 mask = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
  return !_mm256_testz_ps(mask, mask);
}

constexpr unsigned kBothNativeHalves = 0x3U;

unsigned active_lane_bits(__m256 mask) {
  return static_cast<unsigned>(_mm256_movemask_ps(mask)) & 0xFFU;
}

std::size_t active_lane_count(__m256 mask) {
  return static_cast<std::size_t>(std::popcount(active_lane_bits(mask)));
}

unsigned active_native_halves(__m256 mask) {
  const unsigned bits = active_lane_bits(mask);
  return (bits & 0x0FU ? 0x1U : 0U) | (bits & 0xF0U ? 0x2U : 0U);
}

void record_native4_iteration(std::vector<std::uint64_t>& active_by_iteration,
                              std::vector<std::uint64_t>& packets_by_iteration,
                              std::size_t index, __m256 active_mask,
                              unsigned relevant_halves) {
  const unsigned bits = active_lane_bits(active_mask);
  for (unsigned half = 0; half < 2; half += 1) {
    if ((relevant_halves & (1U << half)) == 0) {
      continue;
    }
    const unsigned half_bits = (bits >> (half * 4)) & 0x0FU;
    if (half_bits == 0) {
      continue;
    }
    active_by_iteration[index] += std::popcount(half_bits);
    packets_by_iteration[index] += 1;
  }
}

void record_ray_iteration(LaneOccupancy& occupancy, int step,
                          __m256 active_mask) {
  const std::size_t active = active_lane_count(active_mask);
  const std::size_t index = static_cast<std::size_t>(step);
  occupancy.ray_active_by_step[index] += active;
  occupancy.ray_packets_by_step[index] += 1;
  occupancy.ray_active_histogram[active] += 1;
  record_native4_iteration(occupancy.native4_ray_active_by_step,
                           occupancy.native4_ray_packets_by_step, index,
                           active_mask, kBothNativeHalves);
}

void record_sdf_iteration(LaneOccupancy& occupancy, int iteration,
                          __m256 active_mask, unsigned relevant_halves) {
  const std::size_t active = active_lane_count(active_mask);
  const std::size_t index = static_cast<std::size_t>(iteration);
  occupancy.sdf_active_by_iteration[index] += active;
  occupancy.sdf_packets_by_iteration[index] += 1;
  occupancy.sdf_active_histogram[active] += 1;
  record_native4_iteration(occupancy.native4_sdf_active_by_iteration,
                           occupancy.native4_sdf_packets_by_iteration, index,
                           active_mask, relevant_halves);
}

template <bool CollectLaneOccupancy>
__m256 optim_mandelbulb(Vec3x8& p, render::RenderConfig const& config,
                        LaneOccupancy* occupancy,
                        unsigned native4_relevant_halves) {
  if constexpr (CollectLaneOccupancy) {
    occupancy->sdf_calls += 1;
  }

  Vec3x8 w(p.x256, p.y256, p.z256);
  __m256 m = w.dot(w);

  __m256 dz = _mm256_set1_ps(1.0f);
  const __m256 escape_threshold = _mm256_set1_ps(
      config.mandelbulb_escape_radius * config.mandelbulb_escape_radius);
  __m256 active_mask =
      _mm256_cmp_ps(_mm256_setzero_ps(), _mm256_setzero_ps(), _CMP_EQ_OQ);

  for (int iteration = 0; iteration < config.mandelbulb_iterations;
       iteration += 1) {
    if constexpr (CollectLaneOccupancy) {
      record_sdf_iteration(*occupancy, iteration, active_mask,
                           native4_relevant_halves);
    }

    __m256 m2 = _mm256_mul_ps(m, m);
    __m256 m4 = _mm256_mul_ps(m2, m2);
    // dz = 8.0 * sqrt(m4 * m2 * m) * dz + 1.0;
    __m256 next_dz = _mm256_mul_ps(m4, _mm256_mul_ps(m2, m));
    next_dz = _mm256_sqrt_ps(next_dz);
    next_dz = _mm256_mul_ps(_mm256_set1_ps(8.0f), _mm256_mul_ps(next_dz, dz));
    next_dz = _mm256_add_ps(next_dz, _mm256_set1_ps(1.0f));

    __m256 x = w.x256;
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 x4 = _mm256_mul_ps(x2, x2);
    __m256 y = w.y256;
    __m256 y2 = _mm256_mul_ps(y, y);
    __m256 y4 = _mm256_mul_ps(y2, y2);
    __m256 z = w.z256;
    __m256 z2 = _mm256_mul_ps(z, z);
    __m256 z4 = _mm256_mul_ps(z2, z2);

    __m256 k3 = _mm256_add_ps(x2, z2);

    // float k2 = 1. / std::sqrt(k3 * k3 * k3 * k3 * k3 * k3 * k3);
    __m256 k3sq = _mm256_mul_ps(k3, k3);
    __m256 k2 = _mm256_mul_ps(k3sq, k3sq);
    k2 = _mm256_mul_ps(_mm256_mul_ps(k2, k3sq), k3);
    k2 = _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_sqrt_ps(k2));

    // float k1 = x4 + y4 + z4 - 6.0 * y2 * z2 - 6.0 * x2 * y2 + 2.0 * z2 * x2;
    __m256 k1 = _mm256_add_ps(x4, _mm256_add_ps(y4, z4));
    __m256 k1l = _mm256_mul_ps(_mm256_set1_ps(6.f), _mm256_mul_ps(y2, z2));
    __m256 k1m = _mm256_mul_ps(_mm256_set1_ps(6.f), _mm256_mul_ps(x2, y2));
    __m256 k1r = _mm256_mul_ps(_mm256_set1_ps(2.f), _mm256_mul_ps(z2, x2));
    k1 = _mm256_sub_ps(k1, k1l);
    k1 = _mm256_sub_ps(k1, k1m);
    k1 = _mm256_add_ps(k1, k1r);

    __m256 k4 = _mm256_add_ps(_mm256_sub_ps(x2, y2), z2);

    Vec3x8 next_w;
    next_w.x256 = _mm256_mul_ps(_mm256_set1_ps(64.f),
                                _mm256_mul_ps(x, _mm256_mul_ps(y, z)));
    next_w.x256 =
        _mm256_mul_ps(next_w.x256, _mm256_mul_ps(_mm256_sub_ps(x2, z2), k4));
    next_w.x256 = _mm256_mul_ps(
        next_w.x256,
        _mm256_add_ps(_mm256_sub_ps(x4, _mm256_mul_ps(_mm256_set1_ps(6.f),
                                                      _mm256_mul_ps(x2, z2))),
                      z4));
    next_w.x256 = _mm256_mul_ps(next_w.x256, _mm256_mul_ps(k1, k2));
    next_w.x256 = _mm256_add_ps(next_w.x256, p.x256);

    next_w.y256 = _mm256_mul_ps(
        _mm256_mul_ps(_mm256_set1_ps(-16.f), _mm256_mul_ps(y2, k3)),
        _mm256_mul_ps(k4, k4));
    next_w.y256 = _mm256_add_ps(next_w.y256, p.y256);
    next_w.y256 = _mm256_add_ps(next_w.y256, _mm256_mul_ps(k1, k1));

    __m256 wz1 = _mm256_mul_ps(_mm256_set1_ps(-8.f), _mm256_mul_ps(y, k4));
    __m256 wz21 = _mm256_mul_ps(x4, x4);
    __m256 wz22 = _mm256_mul_ps(_mm256_set1_ps(28.f),
                                _mm256_mul_ps(x4, _mm256_mul_ps(x2, z2)));
    __m256 wz23 = _mm256_mul_ps(_mm256_set1_ps(70.f), _mm256_mul_ps(x4, z4));
    __m256 wz24 = _mm256_mul_ps(_mm256_set1_ps(28.f),
                                _mm256_mul_ps(x2, _mm256_mul_ps(z2, z4)));
    __m256 wz25 = _mm256_mul_ps(z4, z4);
    __m256 wz2 = _mm256_add_ps(
        _mm256_sub_ps(_mm256_add_ps(_mm256_sub_ps(wz21, wz22), wz23), wz24),
        wz25);
    __m256 wz3 = _mm256_mul_ps(k1, k2);
    next_w.z256 =
        _mm256_add_ps(p.z256, _mm256_mul_ps(wz1, _mm256_mul_ps(wz2, wz3)));

    __m256 next_m = next_w.dot(next_w);
    w.x256 = _mm256_blendv_ps(w.x256, next_w.x256, active_mask);
    w.y256 = _mm256_blendv_ps(w.y256, next_w.y256, active_mask);
    w.z256 = _mm256_blendv_ps(w.z256, next_w.z256, active_mask);
    dz = _mm256_blendv_ps(dz, next_dz, active_mask);
    m = _mm256_blendv_ps(m, next_m, active_mask);

    __m256 escaped_mask = _mm256_cmp_ps(next_m, escape_threshold, _CMP_GT_OS);
    active_mask = _mm256_andnot_ps(escaped_mask, active_mask);
    if (_mm256_testz_ps(active_mask, active_mask)) {
      break;
    }
  }
  return _mm256_div_ps(
      _mm256_mul_ps(_mm256_set1_ps(0.25f),
                    _mm256_mul_ps(_mm256_log_ps(m), _mm256_sqrt_ps(m))),
      dz);
}

__m256 sphere_sdf(Vec3x8& p) {
  const __m256 radius = _mm256_set1_ps(1.0f);
  __m256 length = p.length();
  assert(!has_nan(length));
  return _mm256_sub_ps(length, radius);
}

void set_color_to_image(render::Image& image, Vec3x8& color, __m256 xs,
                        __m256 ys, render::RenderConfig const& config) {
#ifdef __GNUC__
  __m128 xs_lo = _mm256_extractf128_ps(xs, 0);
  __m128 xs_hi = _mm256_extractf128_ps(xs, 1);
  __m128 ys_lo = _mm256_extractf128_ps(ys, 0);
  __m128 ys_hi = _mm256_extractf128_ps(ys, 1);
  __m128 color_x_lo = _mm256_extractf128_ps(color.x256, 0);
  __m128 color_x_hi = _mm256_extractf128_ps(color.x256, 1);
  __m128 color_y_lo = _mm256_extractf128_ps(color.y256, 0);
  __m128 color_y_hi = _mm256_extractf128_ps(color.y256, 1);
  __m128 color_z_lo = _mm256_extractf128_ps(color.z256, 0);
  __m128 color_z_hi = _mm256_extractf128_ps(color.z256, 1);

  float xs_arr_lo[4];
  float xs_arr_hi[4];
  float ys_arr_lo[4];
  float ys_arr_hi[4];
  float col_x_arr_lo[4];
  float col_x_arr_hi[4];
  float col_y_arr_lo[4];
  float col_y_arr_hi[4];
  float col_z_arr_lo[4];
  float col_z_arr_hi[4];

  _mm_storeu_ps(xs_arr_lo, xs_lo);
  _mm_storeu_ps(xs_arr_hi, xs_hi);
  _mm_storeu_ps(ys_arr_lo, ys_lo);
  _mm_storeu_ps(ys_arr_hi, ys_hi);
  _mm_storeu_ps(col_x_arr_lo, color_x_lo);
  _mm_storeu_ps(col_x_arr_hi, color_x_hi);
  _mm_storeu_ps(col_y_arr_lo, color_y_lo);
  _mm_storeu_ps(col_y_arr_hi, color_y_hi);
  _mm_storeu_ps(col_z_arr_lo, color_z_lo);
  _mm_storeu_ps(col_z_arr_hi, color_z_hi);

  for (int i = 0; i < 4; ++i) {
    int index = ((int)ys_arr_lo[i] * config.width + (int)xs_arr_lo[i]) *
                render::kColorChannels;
    if (index < config.width * config.height * render::kColorChannels) {
      image[index] = static_cast<unsigned char>(col_x_arr_lo[i]);
    }
    if (index + 1 < config.width * config.height * render::kColorChannels) {
      image[index + 1] = static_cast<unsigned char>(col_y_arr_lo[i]);
    }
    if (index + 2 < config.width * config.height * render::kColorChannels) {
      image[index + 2] = static_cast<unsigned char>(col_z_arr_lo[i]);
    }
  }
  for (int i = 0; i < 4; ++i) {
    int index = ((int)ys_arr_hi[i] * config.width + (int)xs_arr_hi[i]) *
                render::kColorChannels;
    if (index < config.width * config.height * render::kColorChannels) {
      image[index] = static_cast<unsigned char>(col_x_arr_hi[i]);
    }
    if (index + 1 < config.width * config.height * render::kColorChannels) {
      image[index + 1] = static_cast<unsigned char>(col_y_arr_hi[i]);
    }
    if (index + 2 < config.width * config.height * render::kColorChannels) {
      image[index + 2] = static_cast<unsigned char>(col_z_arr_hi[i]);
    }
  }

#else
  for (int i = 0; i < 8; i++) {
    int index = ((int)ys.m256_f32[i] * config.width + (int)xs.m256_f32[i]) *
                render::kColorChannels;
    if (index < config.width * config.height * render::kColorChannels) {
      image[index] = static_cast<unsigned char>(color.x256.m256_f32[i]);
    }
    if (index + 1 < config.width * config.height * render::kColorChannels) {
      image[index + 1] = static_cast<unsigned char>(color.y256.m256_f32[i]);
    }
    if (index + 2 < config.width * config.height * render::kColorChannels) {
      image[index + 2] = static_cast<unsigned char>(color.z256.m256_f32[i]);
    }
  }
#endif
}

}  // namespace

Vec3x8 ray_directions(Camera const& camera, __m256 x, __m256 y,
                      render::RenderConfig const& config) {
  __m256 aspect_ratio = _mm256_set1_ps(static_cast<float>(config.width) /
                                       static_cast<float>(config.height));
  __m256 fov_adjustment = _mm256_set1_ps(
      std::tan(config.fov_degrees * 0.5f * render::kPi / 180.0f));

  // x_adjustment = (2.0 * (x + 0.5) / width - 1.0) * aspect_ratio *
  // fov_adjustment
  __m256 x_centered = _mm256_add_ps(x, _mm256_set1_ps(0.5f));
  __m256 x_scaled = _mm256_mul_ps(
      x_centered, _mm256_set1_ps(2.0f / static_cast<float>(config.width)));
  __m256 x_adjustment =
      _mm256_mul_ps(_mm256_sub_ps(x_scaled, _mm256_set1_ps(1.0f)),
                    _mm256_mul_ps(aspect_ratio, fov_adjustment));
  // y_adjustment = (1.0 - 2.0 * (y + 0.5) / height) * fov_adjustment
  __m256 y_centered = _mm256_add_ps(y, _mm256_set1_ps(0.5f));
  __m256 y_scaled = _mm256_mul_ps(
      y_centered, _mm256_set1_ps(2.0f / static_cast<float>(config.height)));
  __m256 y_adjustment = _mm256_mul_ps(
      _mm256_sub_ps(_mm256_set1_ps(1.0f), y_scaled), fov_adjustment);

  Vec3x8 ray_directions = Vec3x8(camera.forward) +
                          Vec3x8(camera.right) * x_adjustment +
                          Vec3x8(camera.up) * y_adjustment;
  return ray_directions.normalize();
}

namespace {

template <bool CollectLaneOccupancy>
__m256 scene_sdf_impl(Vec3x8& p, render::RenderConfig const& config,
                      LaneOccupancy* occupancy,
                      unsigned native4_relevant_halves) {
  return optim_mandelbulb<CollectLaneOccupancy>(p, config, occupancy,
                                                native4_relevant_halves);
}

template <bool CollectLaneOccupancy>
Vec3x8 estimate_normal_impl(Vec3x8& p, render::RenderConfig const& config,
                            LaneOccupancy* occupancy,
                            unsigned native4_relevant_halves) {
  const __m256 eps = _mm256_set1_ps(config.min_distance);

  Vec3x8 px = p + Vec3x8(eps, _mm256_setzero_ps(), _mm256_setzero_ps());
  Vec3x8 nx = p - Vec3x8(eps, _mm256_setzero_ps(), _mm256_setzero_ps());

  Vec3x8 py = p + Vec3x8(_mm256_setzero_ps(), eps, _mm256_setzero_ps());
  Vec3x8 ny = p - Vec3x8(_mm256_setzero_ps(), eps, _mm256_setzero_ps());

  Vec3x8 pz = p + Vec3x8(_mm256_setzero_ps(), _mm256_setzero_ps(), eps);
  Vec3x8 nz = p - Vec3x8(_mm256_setzero_ps(), _mm256_setzero_ps(), eps);

  __m256 sdf_px = scene_sdf_impl<CollectLaneOccupancy>(px, config, occupancy,
                                                       native4_relevant_halves);
  __m256 sdf_nx = scene_sdf_impl<CollectLaneOccupancy>(nx, config, occupancy,
                                                       native4_relevant_halves);
  __m256 sdf_py = scene_sdf_impl<CollectLaneOccupancy>(py, config, occupancy,
                                                       native4_relevant_halves);
  __m256 sdf_ny = scene_sdf_impl<CollectLaneOccupancy>(ny, config, occupancy,
                                                       native4_relevant_halves);
  __m256 sdf_pz = scene_sdf_impl<CollectLaneOccupancy>(pz, config, occupancy,
                                                       native4_relevant_halves);
  __m256 sdf_nz = scene_sdf_impl<CollectLaneOccupancy>(nz, config, occupancy,
                                                       native4_relevant_halves);

  __m256 nx_grad = _mm256_sub_ps(sdf_px, sdf_nx);
  __m256 ny_grad = _mm256_sub_ps(sdf_py, sdf_ny);
  __m256 nz_grad = _mm256_sub_ps(sdf_pz, sdf_nz);

  return Vec3x8(nx_grad, ny_grad, nz_grad).normalize();
}

template <bool CollectLaneOccupancy>
MarchStep march_step_impl(Vec3x8 const& origins, Vec3x8 const& directions,
                          __m256 distance, render::RenderConfig const& config,
                          LaneOccupancy* occupancy) {
  Vec3x8 position = origins + directions * distance;
  return {position, distance,
          scene_sdf_impl<CollectLaneOccupancy>(position, config, occupancy,
                                               kBothNativeHalves)};
}

}  // namespace

__m256 scene_sdf(Vec3x8& p, render::RenderConfig const& config) {
  return scene_sdf_impl<false>(p, config, nullptr, kBothNativeHalves);
}

Vec3x8 estimate_normal(Vec3x8& p, render::RenderConfig const& config) {
  return estimate_normal_impl<false>(p, config, nullptr, kBothNativeHalves);
}

MarchStep march_step(Vec3x8 const& origins, Vec3x8 const& directions,
                     __m256 distance, render::RenderConfig const& config) {
  return march_step_impl<false>(origins, directions, distance, config, nullptr);
}

__m256 advance_distance(MarchStep const& step, __m256 active_mask,
                        render::RenderConfig const& config) {
  __m256 continue_mask =
      _mm256_cmp_ps(step.sdf, _mm256_set1_ps(config.min_distance), _CMP_GT_OS);
  continue_mask = _mm256_and_ps(continue_mask, active_mask);
  __m256 next_distance = _mm256_add_ps(step.distance, step.sdf);
  return _mm256_blendv_ps(step.distance, next_distance, continue_mask);
}

__m256 hit_mask(MarchStep const& step, render::RenderConfig const& config) {
  return _mm256_cmp_ps(step.sdf, _mm256_set1_ps(config.min_distance),
                       _CMP_LT_OS);
}

__m256 miss_mask(__m256 distance, render::RenderConfig const& config) {
  return _mm256_cmp_ps(distance, _mm256_set1_ps(config.max_distance),
                       _CMP_GT_OS);
}

void apply_hit_color(Vec3x8& color, Vec3x8 const& normal, __m256 mask) {
  color.addWithMask(Vec3x8(255.f), mask);
  color.multiplyWithMask(normal, mask);
}

void clamp_color(Vec3x8& color) {
  color.x256 = _mm256_max_ps(_mm256_set1_ps(0.0f),
                             _mm256_min_ps(_mm256_set1_ps(255.0f), color.x256));
  color.y256 = _mm256_max_ps(_mm256_set1_ps(0.0f),
                             _mm256_min_ps(_mm256_set1_ps(255.0f), color.y256));
  color.z256 = _mm256_max_ps(_mm256_set1_ps(0.0f),
                             _mm256_min_ps(_mm256_set1_ps(255.0f), color.z256));
}

namespace {

template <bool CollectLaneOccupancy>
Vec3x8 trace_ray_packet_impl(Camera const& camera, __m256 xs, __m256 ys,
                             render::RenderConfig const& config,
                             LaneOccupancy* occupancy) {
  if constexpr (CollectLaneOccupancy) {
    occupancy->ray_packets += 1;
  }

  Vec3x8 directions = ray_directions(camera, xs, ys, config);
  Vec3x8 ray_origins(camera.position);
  __m256 distances = _mm256_set1_ps(0.0f);
  __m256 active_mask =
      _mm256_set1_ps(-std::numeric_limits<float>::signaling_NaN());
  Vec3x8 color(0.0f);
  bool packet_hit = false;
  unsigned packet_hit_halves = 0;

  for (int step_count = 0; step_count < config.max_steps; ++step_count) {
    if constexpr (CollectLaneOccupancy) {
      record_ray_iteration(*occupancy, step_count, active_mask);
    }

    MarchStep step = march_step_impl<CollectLaneOccupancy>(
        ray_origins, directions, distances, config, occupancy);

    __m256 hits = _mm256_and_ps(hit_mask(step, config), active_mask);
    if (!_mm256_testz_ps(hits, hits)) {
      unsigned native4_relevant_halves = kBothNativeHalves;
      if constexpr (CollectLaneOccupancy) {
        const std::size_t hit_lanes = active_lane_count(hits);
        native4_relevant_halves = active_native_halves(hits);
        packet_hit = true;
        packet_hit_halves |= native4_relevant_halves;
        occupancy->hit_lanes += hit_lanes;
        occupancy->normal_batches += 1;
        occupancy->normal_hit_lanes += hit_lanes;
        occupancy->native4_normal_batches +=
            std::popcount(native4_relevant_halves);
      }
      Vec3x8 normals = estimate_normal_impl<CollectLaneOccupancy>(
          step.position, config, occupancy, native4_relevant_halves);
      apply_hit_color(color, normals, hits);
    }

    distances = advance_distance(step, active_mask, config);

    __m256 misses = _mm256_and_ps(miss_mask(distances, config), active_mask);
    if (!_mm256_testz_ps(misses, misses)) {
      if constexpr (CollectLaneOccupancy) {
        occupancy->miss_lanes += active_lane_count(misses);
      }
      color.multiplyWithMask(Vec3x8(0.0f), misses);
    }

    __m256 terminate_mask = _mm256_or_ps(hits, misses);
    active_mask = _mm256_andnot_ps(terminate_mask, active_mask);
    if (_mm256_testz_ps(active_mask, active_mask)) {
      break;
    }
  }

  if constexpr (CollectLaneOccupancy) {
    occupancy->hit_packets += packet_hit ? 1 : 0;
    occupancy->native4_hit_packets += std::popcount(packet_hit_halves);
    occupancy->max_step_lanes += active_lane_count(active_mask);
  }

  clamp_color(color);
  return color;
}

template <bool CollectLaneOccupancy>
void render_impl(Camera const& camera, render::RenderConfig const& config,
                 render::Image& image, LaneOccupancy* occupancy) {
  for (int y = 0; y < config.height; ++y) {
    for (int x = 0; x < config.width; x += 8) {
      __m256 xs =
          _mm256_setr_ps(x, x + 1, x + 2, x + 3, x + 4, x + 5, x + 6, x + 7);
      __m256 ys = _mm256_set1_ps(y);
      Vec3x8 color = trace_ray_packet_impl<CollectLaneOccupancy>(
          camera, xs, ys, config, occupancy);
      set_color_to_image(image, color, xs, ys, config);
    }
  }
}

}  // namespace

Vec3x8 trace_ray_packet(Camera const& camera, __m256 xs, __m256 ys,
                        render::RenderConfig const& config) {
  return trace_ray_packet_impl<false>(camera, xs, ys, config, nullptr);
}

void render(Camera const& camera, render::RenderConfig const& config,
            render::Image& image) {
  render_impl<false>(camera, config, image, nullptr);
}

#ifdef SIMD_RAY_MARCHER_ENABLE_LANE_OCCUPANCY
void render_with_lane_occupancy(const Camera& camera,
                                const render::RenderConfig& config,
                                render::Image& image,
                                LaneOccupancy& occupancy) {
  render_impl<true>(camera, config, image, &occupancy);
}
#endif

}  // namespace simd8
