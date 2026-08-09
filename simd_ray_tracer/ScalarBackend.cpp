#include <cmath>

#include "ScalarBackend.h"

namespace scalar {

Vec3 ray_direction(Camera const& camera, float x, float y,
                   render::RenderConfig const& config) {
    float aspect_ratio = static_cast<float>(config.width) / static_cast<float>(config.height);
    float fov_adjustment = std::tan(config.fov_degrees * 0.5f * render::kPi / 180.0f);
    float x_adjustment = (2.0f * (x + 0.5f) / static_cast<float>(config.width) - 1.0f) * aspect_ratio * fov_adjustment;
    float y_adjustment = (1.0f - 2.0f * (y + 0.5f) / static_cast<float>(config.height)) * fov_adjustment;
    return (camera.forward + camera.right * x_adjustment + camera.up * y_adjustment).normalize();
}

} // namespace scalar
