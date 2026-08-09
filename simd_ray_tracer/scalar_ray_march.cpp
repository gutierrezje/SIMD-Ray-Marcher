// simd_ray_tracer.cpp : Defines the entry point for the application.
//

#include <iostream>
#include <algorithm>

#include "ScalarBackend.h"
#include "RenderTypes.h"

#include <chrono>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h" // Or other image library

namespace {
constexpr const render::RenderConfig& kConfig = render::kDefaultRenderConfig;
}

void ray_march(int x, int y, Camera const& camera, render::Image& image) {
    Vec3 direction = scalar::ray_direction(camera, static_cast<float>(x),
                                           static_cast<float>(y), kConfig);
    Vec3 ray_origin = camera.position;
    float distance = 0.0f;
    for (int i = 0; i < kConfig.max_steps; i++) {
        Vec3 p = ray_origin + direction * distance;
        float dist = scalar::scene_sdf(p);
        if (dist < kConfig.min_distance) {
            float color[3] = { 255.0f, 255.0f, 255.0f };
            // apply lighting
            color[0] = scalar::estimate_normal(p, kConfig).x * 255.0f;
            color[1] = scalar::estimate_normal(p, kConfig).y * 255.0f;
            color[2] = scalar::estimate_normal(p, kConfig).z * 255.0f;
            // clamp and apply color
            for (int j : {0, 1, 2}) {
                color[j] = std::max(0.0f, std::min(255.0f, color[j]));
                image[(y * kConfig.width + x) * render::kColorChannels + j] = color[j];
            }
            break;
        }
        distance += dist;
        if (distance > kConfig.max_distance) {
            image[(y * kConfig.width + x) * render::kColorChannels] = 0;
            image[(y * kConfig.width + x) * render::kColorChannels + 1] = 0;
            image[(y * kConfig.width + x) * render::kColorChannels + 2] = 0;
            break;
        }
    }

}

int main() {
    render::Image image(kConfig);

    Vec3 camera_position = Vec3(0.0f, 0.0f, 2.0f);
    Vec3 look_at = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 up = Vec3(0.0f, 1.0f, 0.0f);
    Camera camera = make_camera(camera_position, look_at, up);

    // Set up timer
    auto start = std::chrono::high_resolution_clock::now();

    for (int y = 0; y < kConfig.height; y++) {
        for (int x = 0; x < kConfig.width; x++)  {
            ray_march(x, y, camera, image);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;

    stbi_write_png("output.png", kConfig.width, kConfig.height,
                   render::kColorChannels, image.data(), image.row_stride());
    return 0;
}
