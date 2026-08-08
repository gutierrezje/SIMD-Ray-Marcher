#pragma once

#include <numbers>

#include "Vec3.h"
#include "Vec3x8.h"

// Image parameters
inline constexpr int WIDTH = 2056;
inline constexpr int HEIGHT = 2056;

// Camera parameters
inline constexpr float CAMERA_DISTANCE = 3.0f;
inline constexpr float ASPECT_RATIO = 1.0f;
inline constexpr float FOV = 45.0f;
inline constexpr float PI = std::numbers::pi_v<float>;

class Camera {
public:
    Vec3 position;
    Vec3 forward;
    Vec3 right;
    Vec3 up;
    Camera(Vec3 position, Vec3 target, Vec3 up);
    Vec3 get_ray_direction(float x, float y);
    Vec3x8 get_ray_directions(__m256 x, __m256 y);
};
