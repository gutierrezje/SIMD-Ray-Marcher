#include "Camera.h"

Camera::Camera(Vec3 position, Vec3 target, Vec3 up) : position(position) {
    forward = (target - position).normalize();
    right = forward.cross(up).normalize();
    this->up = right.cross(forward).normalize();
}
Vec3 Camera::get_ray_direction(double x, double y) {
    double aspect_ratio = WIDTH / HEIGHT;
    double fov_adjustment = std::tan(FOV * 0.5 * M_PI / 180.0);
    double x_adjustment = (2.0 * (x + 0.5) / WIDTH - 1.0) * aspect_ratio * fov_adjustment;
    double y_adjustment = (1.0 - 2.0 * (y + 0.5) / HEIGHT) * fov_adjustment;
    return (forward + right * x_adjustment + up * y_adjustment).normalize();
}

Vec3x8 Camera::get_ray_directions(__m256 x, __m256 y) {
    __m256 aspect_ratio = _mm256_set1_ps(WIDTH / HEIGHT);
    __m256 fov_adjustment = _mm256_set1_ps(std::tan(FOV * 0.5 * M_PI / 180.0));

    // x_adjustment = (2.0 * (x + 0.5) / WIDTH - 1.0) * aspect_ratio * fov_adjustment
    __m256 x_centered = _mm256_add_ps(x, _mm256_set1_ps(0.5));
    __m256 x_scaled = _mm256_mul_ps(x_centered, _mm256_set1_ps(2.0 / WIDTH));
    __m256 x_adjustment = _mm256_mul_ps(
        _mm256_sub_ps(x_scaled, _mm256_set1_ps(1.0)),
        _mm256_mul_ps(aspect_ratio, fov_adjustment)
    );
    // y_adjustment = (1.0 - 2.0 * (y + 0.5) / HEIGHT) * fov_adjustment
    __m256 y_centered = _mm256_add_ps(y, _mm256_set1_ps(0.5));
    __m256 y_scaled = _mm256_mul_ps(y_centered, _mm256_set1_ps(2.0 / HEIGHT));
    __m256 y_adjustment = _mm256_mul_ps(
        _mm256_sub_ps(_mm256_set1_ps(1.0), y_scaled),
        fov_adjustment
    );

    Vec3x8 rayDirections = Vec3x8(forward) + Vec3x8(right) * x_adjustment + Vec3x8(up) * y_adjustment;
    return rayDirections.normalize();
}