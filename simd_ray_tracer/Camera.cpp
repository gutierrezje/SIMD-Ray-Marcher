#include "Camera.h"

Camera make_camera(Vec3 position, Vec3 target, Vec3 up) {
    Camera camera;
    camera.position = position;
    camera.forward = (target - position).normalize();
    camera.right = camera.forward.cross(up).normalize();
    camera.up = camera.right.cross(camera.forward).normalize();
    return camera;
}
