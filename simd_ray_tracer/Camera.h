#pragma once

#include "Vec3.h"

struct Camera {
    Vec3 position;
    Vec3 forward;
    Vec3 right;
    Vec3 up;
};

Camera make_camera(Vec3 position, Vec3 target, Vec3 up);
Camera make_default_camera();
