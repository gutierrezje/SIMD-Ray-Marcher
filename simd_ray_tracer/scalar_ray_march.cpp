// simd_ray_tracer.cpp : Defines the entry point for the application.
//

#include "scalar_ray_march.h"

#include <chrono>

//#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h" // Or other image library

// Mandelbulb parameters
constexpr int ITERATIONS = 10;
constexpr double POWER = 8.0;

// Ray marching parameters
constexpr double MIN_DIST = 0.001;
constexpr double MAX_DIST = 100.0;
constexpr int MAX_STEPS = 100;

// Helper functions
double length(double x, double y, double z) {
    return std::sqrt(x * x + y * y + z * z);
}

// Constructive solid geometry functions
// https://iquilezles.org/www/articles/distfunctions/distfunctions.htm

double unionSDF(double a, double b) {
    return std::min(a, b);
}

double intersectSDF(double a, double b) {
    return std::max(a, b);
}

double diffSDF(double a, double b) {
    return std::max(a, -b);
}

// Signed distance functions

double sphereSDF(Vec3 p, Vec3 center, double radius) {
    return (p - center).length() - radius;
}

double boxSDF(Vec3 p, Vec3 center, Vec3 size) {
    Vec3 d = (p - center).abs() - size;
    double inside_distance = std::min(std::max(d.x, std::max(d.y, d.z)), 0.0);
    double outside_distance = length(std::max(d.x, 0.0), std::max(d.y, 0.0), std::max(d.z, 0.0));
    return inside_distance + outside_distance;
}

double mandelbulb1(Vec3 pos) {
    Vec3 z = pos;
    double dr = 1.0;
    double r = 0.0;
    for (int i = 0; i < ITERATIONS; i++) {
        r = z.length();
        if (r > MAX_DIST)
            break;

        double theta = std::acos(z.z / r);
        theta *= POWER;

        double phi = std::atan2(z.y, z.x);
        phi *= POWER;

        double zr = std::pow(r, POWER);
        dr = std::pow(r, POWER - 1.0) * POWER * dr + 1.0;

        z = Vec3(
            std::sin(theta) * std::cos(phi),
            std::sin(phi) * std::sin(theta),
            std::cos(theta)
        )
            * zr
            + pos;
    }
    return 0.5 * std::log(r) * r / dr;
}

double mandelbulb(Vec3 pos) {
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

    // Distance estimation
    return 0.25f * std::log(m) * std::sqrt(m) / dz;
}

double optimMandelbulb(Vec3 pos) {
    Vec3 w = pos;
    float m = w.dot(w);

    float dz = 1.0f;

    for (int i = 0; i < 4; i++) {
        float m2 = m * m;
        float m4 = m2 * m2;
        dz = 8.0 * sqrt(m4 * m2 * m) * dz + 1.0;

        float x = w.x; float x2 = x * x; float x4 = x2 * x2;
        float y = w.y; float y2 = y * y; float y4 = y2 * y2;
        float z = w.z; float z2 = z * z; float z4 = z2 * z2;

        float k3 = x2 + z2;
        float k2 = 1. / std::sqrt(k3 * k3 * k3 * k3 * k3 * k3 * k3);
        float k1 = x4 + y4 + z4 - 6.0 * y2 * z2 - 6.0 * x2 * y2 + 2.0 * z2 * x2;
        float k4 = x2 - y2 + z2;

        w.x = pos.x + 64.0 * x * y * z * (x2 - z2) * k4 * (x4 - 6.0 * x2 * z2 + z4) * k1 * k2;
        w.y = pos.y + -16.0 * y2 * k3 * k4 * k4 + k1 * k1;
        w.z = pos.z + -8.0 * y * k4 * (x4 * x4 - 28.0 * x4 * x2 * z2 + 70.0 * x4 * z4 - 28.0 * x2 * z2 * z4 + z4 * z4) * k1 * k2;

        m = w.dot(w);
        if (m > 256.0f)
            break;
    }

    // Distance estimation
    return 0.25f * std::log(m) * std::sqrt(m) / dz;
}

double sceneSDF(double x, double y, double z) {
    /*Vec3 p(x, y, z);
    double sphere = sphereSDF(p / 1.2, Vec3(0, 0, 0), 1.0) * 1.2;
    double cube = boxSDF(p, Vec3(0, 0, 0), Vec3(1, 1, 1));
    return intersectSDF(sphere, cube);*/
    //return mandelbulb1(Vec3(x, y, z));
    return sphereSDF(Vec3(x, y, z), Vec3(0, 0, 0), 1.0);
}

Vec3 estimateNormal(Vec3 p) {
    Vec3 normal(
        sceneSDF(p.x + MIN_DIST, p.y, p.z) - sceneSDF(p.x - MIN_DIST, p.y, p.z),
        sceneSDF(p.x, p.y + MIN_DIST, p.z) - sceneSDF(p.x, p.y - MIN_DIST, p.z),
        sceneSDF(p.x, p.y, p.z + MIN_DIST) - sceneSDF(p.x, p.y, p.z - MIN_DIST)
    );
    return normal.normalize();
}

void ray_march(Vec3 origin, Camera camera, std::vector<unsigned char>& image) {
    auto [x, y, z] = origin;
    Vec3 direction = camera.get_ray_direction(x, y);
    Vec3 ray_origin = camera.position;
    double distance = 0.0;
    for (int i = 0; i < MAX_STEPS; i++) {
        Vec3 p = ray_origin + direction * distance;
        double dist = sceneSDF(p.x, p.y, p.z);
        if (dist < MIN_DIST) {
            double color[3] = { 255, 255, 255 };
            // apply lighting
            color[0] = estimateNormal(p).x * 255;
            color[1] = estimateNormal(p).y * 255;
            color[2] = estimateNormal(p).z * 255;
            // clamp and apply color
            for (int j : {0, 1, 2}) {
                color[j] = std::max(0.0, std::min(255.0, color[j]));
                image[(y * WIDTH + x) * 3 + j] = color[j];
            }
            break;
        }
        distance += dist;
        if (distance > MAX_DIST) {
            image[(y * WIDTH + x) * 3] = 0;
            image[(y * WIDTH + x) * 3 + 1] = 0;
            image[(y * WIDTH + x) * 3 + 2] = 0;
            break;
        }
    }

}

//int main() {
//    std::vector<unsigned char> image(WIDTH * HEIGHT * 3);
//
//    Vec3 camera_position = Vec3(1., 0., 2.5);
//    Vec3 look_at = Vec3(0, 0, 0);
//    Vec3 up = Vec3(0, 1, 0);
//    Camera camera(camera_position, look_at, up);
//
//    // Set up timer
//    auto start = std::chrono::high_resolution_clock::now();
//
//    for (int y = 0; y < HEIGHT; y++) {
//        for (int x = 0; x < WIDTH; x++)  {
//            ray_march(Vec3(x, y, 0), camera, image);
//        }
//    }
//
//    auto end = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double> elapsed = end - start;
//    std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;
//
//    stbi_write_png("output.png", WIDTH, HEIGHT, 3, image.data(), WIDTH * 3);
//    return 0;
//}