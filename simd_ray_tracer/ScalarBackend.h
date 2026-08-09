#pragma once

#include "Camera.h"
#include "RenderConfig.h"

namespace scalar {

Vec3 ray_direction(Camera const& camera, float x, float y,
                   render::RenderConfig const& config);
float scene_sdf(Vec3 p);
Vec3 estimate_normal(Vec3 p, render::RenderConfig const& config);

} // namespace scalar
