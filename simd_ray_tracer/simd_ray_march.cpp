#include <chrono>
#include <iostream>

#include "simd_compat.h"

#include "SimdBackend.h"
#include "RenderTypes.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h" // Or other image library

#include <limits>

namespace {
constexpr const render::RenderConfig& kConfig = render::kDefaultRenderConfig;
}

// Simd helper function to set Vec3x8 color to image
void setColorToImage(render::Image& image, Vec3x8& color, __m256 xs, __m256 ys)
{
#ifdef __GNUC__
    __m128 xs_lo = _mm256_extractf128_ps(xs, 0);
    __m128 xs_hi = _mm256_extractf128_ps(xs, 1);
    __m128 ys_lo = _mm256_extractf128_ps(ys, 0);
    __m128 ys_hi = _mm256_extractf128_ps(ys, 1);
    __m128 color_x_lo = _mm256_extractf128_ps(color.x256, 0);
    __m128 color_x_hi = _mm256_extractf128_ps(color.x256, 1);
    __m128 color_y_lo = _mm256_extractf128_ps(color.y256, 0);
    __m128 color_y_hi = _mm256_extractf128_ps(color.y256, 1);
    __m128 color_z_lo = _mm256_extractf128_ps(color.z256, 0);
    __m128 color_z_hi = _mm256_extractf128_ps(color.z256, 1);

    float xs_arr_lo[4];
    float xs_arr_hi[4];
    float ys_arr_lo[4];
    float ys_arr_hi[4];
    float col_x_arr_lo[4];
    float col_x_arr_hi[4];
    float col_y_arr_lo[4];
    float col_y_arr_hi[4];
    float col_z_arr_lo[4];
    float col_z_arr_hi[4];

    _mm_storeu_ps(xs_arr_lo, xs_lo);
    _mm_storeu_ps(xs_arr_hi, xs_hi);
    _mm_storeu_ps(ys_arr_lo, ys_lo);
    _mm_storeu_ps(ys_arr_hi, ys_hi);
    _mm_storeu_ps(col_x_arr_lo, color_x_lo);
    _mm_storeu_ps(col_x_arr_hi, color_x_hi);
    _mm_storeu_ps(col_y_arr_lo, color_y_lo);
    _mm_storeu_ps(col_y_arr_hi, color_y_hi);
    _mm_storeu_ps(col_z_arr_lo, color_z_lo);
    _mm_storeu_ps(col_z_arr_hi, color_z_hi);

    for (int i = 0; i < 4; ++i) {
        int index = ((int)ys_arr_lo[i] * kConfig.width + (int)xs_arr_lo[i]) * render::kColorChannels;
        if (index < kConfig.width * kConfig.height * render::kColorChannels) {
            image[index] = static_cast<unsigned char>(col_x_arr_lo[i]);
        }
        if (index + 1 < kConfig.width * kConfig.height * render::kColorChannels) {
            image[index + 1] = static_cast<unsigned char>(col_y_arr_lo[i]);
        }
        if (index + 2 < kConfig.width * kConfig.height * render::kColorChannels) {
            image[index + 2] = static_cast<unsigned char>(col_z_arr_lo[i]);
        }
    }
    for (int i = 0; i < 4; ++i) {
        int index = ((int)ys_arr_hi[i] * kConfig.width + (int)xs_arr_hi[i]) * render::kColorChannels;
        if (index < kConfig.width * kConfig.height * render::kColorChannels) {
            image[index] = static_cast<unsigned char>(col_x_arr_hi[i]);
        }
        if (index + 1 < kConfig.width * kConfig.height * render::kColorChannels) {
            image[index + 1] = static_cast<unsigned char>(col_y_arr_hi[i]);
        }
        if (index + 2 < kConfig.width * kConfig.height * render::kColorChannels) {
            image[index + 2] = static_cast<unsigned char>(col_z_arr_hi[i]);
        }
    }

#else
    for (int i = 0; i < 8; i++) {
        int index = ((int)ys.m256_f32[i] * kConfig.width + (int)xs.m256_f32[i]) * render::kColorChannels;
        if (index < kConfig.width * kConfig.height * render::kColorChannels) {
            image[index] = static_cast<unsigned char>(color.x256.m256_f32[i]);
        }
        if (index + 1 < kConfig.width * kConfig.height * render::kColorChannels) {
            image[index + 1] = static_cast<unsigned char>(color.y256.m256_f32[i]);
        }
        if (index + 2 < kConfig.width * kConfig.height * render::kColorChannels) {
            image[index + 2] = static_cast<unsigned char>(color.z256.m256_f32[i]);
        }
    }
#endif
}

void ray_march(Vec3x8 origins, Camera const& camera, render::Image& image)
{
    auto [xs, ys, zs] = origins;
    Vec3x8 directions = simd8::ray_directions(camera, xs, ys, kConfig);
    Vec3x8 ray_origins(camera.position);
    __m256 distances = _mm256_set1_ps(0.0f);

    Vec3x8 color(0.0f);
    __m256 activeMask = _mm256_set1_ps(-std::numeric_limits<float>::signaling_NaN());

    for (int i = 0; i < kConfig.max_steps; i++) {
        Vec3x8 p = ray_origins + directions * distances;
        __m256 dists = simd8::scene_sdf(p);

        // Check for rays that have reached the minimum distance
        __m256 mask = _mm256_cmp_ps(dists, _mm256_set1_ps(kConfig.min_distance), _CMP_LT_OS);
        // disable rays that have reached the minimum distance before
        mask = _mm256_and_ps(mask, activeMask);

        // check if any of the mask is non zero
        if (!_mm256_testz_ps(mask, mask)) {
            Vec3x8 normals = simd8::estimate_normal(p, kConfig);
            color.addWithMask(Vec3x8(255.f), mask);
            color.multiplyWithMask(normals, mask);
        }

        // update distances
        // invert mask
        __m256 maskInv = _mm256_cmp_ps(dists, _mm256_set1_ps(kConfig.min_distance), _CMP_GT_OS);
        maskInv = _mm256_and_ps(maskInv, activeMask);
        __m256 newDistances = _mm256_add_ps(distances, dists);
        distances = _mm256_blendv_ps(distances, newDistances, maskInv);

        // Check for rays that have reached the maximum distance
        __m256 mask2 = _mm256_cmp_ps(distances, _mm256_set1_ps(kConfig.max_distance), _CMP_GT_OS);
        mask2 = _mm256_and_ps(mask2, activeMask);
        if (!_mm256_testz_ps(mask2, mask2)) {
            color.multiplyWithMask(Vec3x8(0.0f), mask2);
        }

        // disable inactive rays
        __m256 terminateMask = _mm256_or_ps(mask, mask2);
        activeMask = _mm256_andnot_ps(terminateMask, activeMask);

        // check if all rays are inactive
        if (_mm256_testz_ps(activeMask, activeMask)) {
            break;
        }
    }

    ////// clamp color
    color.x256 = _mm256_max_ps(_mm256_set1_ps(0.0f), _mm256_min_ps(_mm256_set1_ps(255.0f), color.x256));
    color.y256 = _mm256_max_ps(_mm256_set1_ps(0.0f), _mm256_min_ps(_mm256_set1_ps(255.0f), color.y256));
    color.z256 = _mm256_max_ps(_mm256_set1_ps(0.0f), _mm256_min_ps(_mm256_set1_ps(255.0f), color.z256));
    setColorToImage(image, color, xs, ys);
}

int main()
{
    render::Image image(kConfig);

    Vec3 camera_position = Vec3(0.0f, 0.0f, 2.0f);
    Vec3 look_at = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 up = Vec3(0.0f, 1.0f, 0.0f);
    Camera camera = make_camera(camera_position, look_at, up);

    auto start = std::chrono::high_resolution_clock::now();

    for (int y = 0; y < kConfig.height; y += 1) {
        for (int x = 0; x < kConfig.width; x += 8) {
            __m256 us = _mm256_setr_ps(x, x + 1, x + 2, x + 3, x + 4, x + 5, x + 6, x + 7);
            __m256 vs = _mm256_set1_ps(y);
            ray_march(Vec3x8(us, vs, _mm256_set1_ps(0.0f)), camera, image);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    stbi_write_png("output.png", kConfig.width, kConfig.height,
                   render::kColorChannels, image.data(), image.row_stride());

    std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;
    return 0;
}
