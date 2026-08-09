#include <algorithm>
#include <cmath>

#include "ScalarBackend.h"

namespace scalar {

namespace {

constexpr const render::RenderConfig& kConfig = render::kDefaultRenderConfig;

float length(float x, float y, float z) {
    return std::sqrt(x * x + y * y + z * z);
}

float union_sdf(float a, float b) {
    return std::min(a, b);
}

float intersect_sdf(float a, float b) {
    return std::max(a, b);
}

float diff_sdf(float a, float b) {
    return std::max(a, -b);
}

float sphere_sdf(Vec3 p, Vec3 center, float radius) {
    return (p - center).length() - radius;
}

float box_sdf(Vec3 p, Vec3 center, Vec3 size) {
    Vec3 d = (p - center).abs() - size;
    float inside_distance = std::min(std::max(d.x, std::max(d.y, d.z)), 0.0f);
    float outside_distance = length(std::max(d.x, 0.0f), std::max(d.y, 0.0f), std::max(d.z, 0.0f));
    return inside_distance + outside_distance;
}

float mandelbulb1(Vec3 pos) {
    Vec3 z = pos;
    float dr = 1.0f;
    float r = 0.0f;
    for (int i = 0; i < kConfig.mandelbulb_iterations; i++) {
        r = z.length();
        if (r > kConfig.max_distance)
            break;

        float theta = std::acos(z.z / r);
        theta *= kConfig.power;

        float phi = std::atan2(z.y, z.x);
        phi *= kConfig.power;

        float zr = std::pow(r, kConfig.power);
        dr = std::pow(r, kConfig.power - 1.0f) * kConfig.power * dr + 1.0f;

        z = Vec3(
            std::sin(theta) * std::cos(phi),
            std::sin(phi) * std::sin(theta),
            std::cos(theta)
        )
            * zr
            + pos;
    }
    return 0.5f * std::log(r) * r / dr;
}

float mandelbulb(Vec3 pos) {
    Vec3 w = pos;
    float m = w.dot(w);

    float dz = 1.0f;

    for (int i = 0; i < 4; i++) {
        dz = 8.0f * std::pow(m, 3.5f) * dz + 1.0f;

        float r = w.length();
        float b = 8.0f * std::acos(w.y / r);
        float a = 8.0f * std::atan2(w.x, w.z);
        w = pos + Vec3(std::sin(b) * std::sin(a), std::cos(b), std::sin(b) * std::cos(a)) * std::pow(r, 8.0f);

        m = w.dot(w);
        if (m > 256.0f)
            break;
    }

    return 0.25f * std::log(m) * std::sqrt(m) / dz;
}

float optim_mandelbulb(Vec3 pos) {
    Vec3 w = pos;
    float m = w.dot(w);

    float dz = 1.0f;

    for (int i = 0; i < 4; i++) {
        float m2 = m * m;
        float m4 = m2 * m2;
        dz = 8.0f * std::sqrt(m4 * m2 * m) * dz + 1.0f;

        float x = w.x; float x2 = x * x; float x4 = x2 * x2;
        float y = w.y; float y2 = y * y; float y4 = y2 * y2;
        float z = w.z; float z2 = z * z; float z4 = z2 * z2;

        float k3 = x2 + z2;
        float k2 = 1.0f / std::sqrt(k3 * k3 * k3 * k3 * k3 * k3 * k3);
        float k1 = x4 + y4 + z4 - (6.0f * y2 * z2) - (6.0f * x2 * y2) + (2.0f * z2 * x2);
        float k4 = x2 - y2 + z2;

        w.x = pos.x + (64.0f * x * y * z) * (x2 - z2) * k4 * (x4 - 6.0f * x2 * z2 + z4) * k1 * k2;
        w.y = pos.y + -16.0f * y2 * k3 * k4 * k4 + k1 * k1;
        w.z = pos.z +
            -8.0f * y * k4 *
            (
                x4 * x4 -
                (28.0f * x4 * x2 * z2) +
                (70.0f * x4 * z4) -
                (28.0f * x2 * z2 * z4) +
                z4 * z4
            ) * k1 * k2;

        m = w.dot(w);
        if (m > 256.0f)
            break;
    }

    return 0.25f * std::log(m) * std::sqrt(m) / dz;
}

} // namespace

Vec3 ray_direction(Camera const& camera, float x, float y,
                   render::RenderConfig const& config) {
    float aspect_ratio = static_cast<float>(config.width) / static_cast<float>(config.height);
    float fov_adjustment = std::tan(config.fov_degrees * 0.5f * render::kPi / 180.0f);
    float x_adjustment = (2.0f * (x + 0.5f) / static_cast<float>(config.width) - 1.0f) * aspect_ratio * fov_adjustment;
    float y_adjustment = (1.0f - 2.0f * (y + 0.5f) / static_cast<float>(config.height)) * fov_adjustment;
    return (camera.forward + camera.right * x_adjustment + camera.up * y_adjustment).normalize();
}

float scene_sdf(Vec3 p) {
    return optim_mandelbulb(p);
}

Vec3 estimate_normal(Vec3 p, render::RenderConfig const& config) {
    Vec3 normal(
        scene_sdf(Vec3(p.x + config.min_distance, p.y, p.z)) - scene_sdf(Vec3(p.x - config.min_distance, p.y, p.z)),
        scene_sdf(Vec3(p.x, p.y + config.min_distance, p.z)) - scene_sdf(Vec3(p.x, p.y - config.min_distance, p.z)),
        scene_sdf(Vec3(p.x, p.y, p.z + config.min_distance)) - scene_sdf(Vec3(p.x, p.y, p.z - config.min_distance))
    );
    return normal.normalize();
}

} // namespace scalar
