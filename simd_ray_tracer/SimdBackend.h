#pragma once

#include "Camera.h"
#include "RenderConfig.h"
#include "Vec3x8.h"

namespace simd8 {

Vec3x8 ray_directions(Camera const& camera, __m256 x, __m256 y,
                      render::RenderConfig const& config);

} // namespace simd8
