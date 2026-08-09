#include "Camera.h"

Camera make_camera(Vec3 position, Vec3 target, Vec3 up) {
    Camera camera;
    camera.position = position;
    camera.forward = (target - position).normalize();
    camera.right = camera.forward.cross(up).normalize();
    camera.up = camera.right.cross(camera.forward).normalize();
    return camera;
}

Camera make_default_camera() {
    return make_camera(
        Vec3(0.0f, 0.0f, 3.0f),
        Vec3(0.0f, 0.0f, 0.0f),
        Vec3(0.0f, 1.0f, 0.0f));
}
