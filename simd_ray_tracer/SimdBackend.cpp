#include <cmath>

#include "SimdBackend.h"

namespace simd8 {

Vec3x8 ray_directions(Camera const& camera, __m256 x, __m256 y,
                      render::RenderConfig const& config) {
    __m256 aspect_ratio = _mm256_set1_ps(static_cast<float>(config.width) / static_cast<float>(config.height));
    __m256 fov_adjustment = _mm256_set1_ps(std::tan(config.fov_degrees * 0.5f * render::kPi / 180.0f));

    // x_adjustment = (2.0 * (x + 0.5) / width - 1.0) * aspect_ratio * fov_adjustment
    __m256 x_centered = _mm256_add_ps(x, _mm256_set1_ps(0.5f));
    __m256 x_scaled = _mm256_mul_ps(x_centered, _mm256_set1_ps(2.0f / static_cast<float>(config.width)));
    __m256 x_adjustment = _mm256_mul_ps(
        _mm256_sub_ps(x_scaled, _mm256_set1_ps(1.0f)),
        _mm256_mul_ps(aspect_ratio, fov_adjustment)
    );
    // y_adjustment = (1.0 - 2.0 * (y + 0.5) / height) * fov_adjustment
    __m256 y_centered = _mm256_add_ps(y, _mm256_set1_ps(0.5f));
    __m256 y_scaled = _mm256_mul_ps(y_centered, _mm256_set1_ps(2.0f / static_cast<float>(config.height)));
    __m256 y_adjustment = _mm256_mul_ps(
        _mm256_sub_ps(_mm256_set1_ps(1.0f), y_scaled),
        fov_adjustment
    );

    Vec3x8 ray_directions = Vec3x8(camera.forward) + Vec3x8(camera.right) * x_adjustment + Vec3x8(camera.up) * y_adjustment;
    return ray_directions.normalize();
}

} // namespace simd8
