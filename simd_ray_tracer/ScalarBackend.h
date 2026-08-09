#pragma once

#include "Camera.h"
#include "RenderTypes.h"

namespace scalar {

struct MarchStep {
    Vec3 position;
    float distance;
    float sdf;
};

// These stages are public so the correctness harness can compare scalar and
// SIMD behavior at the first operation that diverges.
Vec3 ray_direction(Camera const& camera, float x, float y,
                   render::RenderConfig const& config);
float scene_sdf(Vec3 p);
Vec3 estimate_normal(Vec3 p, render::RenderConfig const& config);
MarchStep march_step(Vec3 origin, Vec3 direction, float distance);
float advance_distance(MarchStep const& step);
bool is_hit(MarchStep const& step, render::RenderConfig const& config);
bool is_miss(float distance, render::RenderConfig const& config);
render::Color hit_color(Vec3 position,
                        render::RenderConfig const& config);
render::Color trace_ray(Camera const& camera, int x, int y,
                        render::RenderConfig const& config);
void render(Camera const& camera, render::RenderConfig const& config,
            render::Image& image);

} // namespace scalar
