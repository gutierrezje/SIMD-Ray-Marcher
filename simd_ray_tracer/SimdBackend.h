#pragma once

#include "Camera.h"
#include "RenderTypes.h"
#include "Vec3x8.h"

namespace simd8 {

struct MarchStep {
    Vec3x8 position;
    __m256 distance;
    __m256 sdf;
};

// These stages are public so the correctness harness can compare scalar and
// SIMD behavior at the first operation that diverges.
Vec3x8 ray_directions(Camera const& camera, __m256 x, __m256 y,
                      render::RenderConfig const& config);
__m256 scene_sdf(Vec3x8& p);
Vec3x8 estimate_normal(Vec3x8& p, render::RenderConfig const& config);
MarchStep march_step(Vec3x8 const& origins, Vec3x8 const& directions,
                     __m256 distance);
__m256 advance_distance(MarchStep const& step, __m256 active_mask,
                        render::RenderConfig const& config);
__m256 hit_mask(MarchStep const& step, render::RenderConfig const& config);
__m256 miss_mask(__m256 distance, render::RenderConfig const& config);
void apply_hit_color(Vec3x8& color, Vec3x8 const& normal, __m256 mask);
void clamp_color(Vec3x8& color);
Vec3x8 trace_ray_packet(Camera const& camera, __m256 xs, __m256 ys,
                        render::RenderConfig const& config);
void render(Camera const& camera, render::RenderConfig const& config,
            render::Image& image);

} // namespace simd8
