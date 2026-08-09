#pragma once

#include "RenderConfig.h"
#include "Vec3.h"
#include "Vec3x8.h"

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
