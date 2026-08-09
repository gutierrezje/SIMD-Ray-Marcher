#pragma once

#include "Camera.h"
#include "RenderConfig.h"

namespace scalar {

Vec3 ray_direction(Camera const& camera, float x, float y,
                   render::RenderConfig const& config);

} // namespace scalar
