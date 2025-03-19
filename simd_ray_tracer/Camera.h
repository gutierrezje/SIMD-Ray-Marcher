#pragma once

#include "Vec3.h"
#include "Vec3x8.h"

// Image parameters
inline constexpr int WIDTH = 256;
inline constexpr int HEIGHT = 256;

// Camera parameters
inline constexpr double CAMERA_DISTANCE = 3.0;
inline constexpr double ASPECT_RATIO = 1.0;
inline constexpr double FOV = 45.0;

// Constants
inline constexpr double M_PI = 3.14159265358979323846;
class Camera {
public:
    Vec3 position;
    Vec3 forward;
    Vec3 right;
    Vec3 up;
    Camera(Vec3 position, Vec3 target, Vec3 up);
    Vec3 get_ray_direction(double x, double y);
    Vec3x8 get_ray_directions(__m256 x, __m256 y);
};