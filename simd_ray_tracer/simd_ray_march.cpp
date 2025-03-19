#include "simd_ray_march.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h" // Or other image library

#include <limits>

// Mandelbulb parameters
constexpr int ITERATIONS = 10;
constexpr double POWER = 8.0;

// Ray marching parameters
constexpr double MIN_DIST = 0.001;
constexpr double MAX_DIST = 100.0;
constexpr int MAX_STEPS = 100;

#ifndef DEBUG
bool hasNan(__m256 v) {
    __m256 mask = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
    return !_mm256_testz_ps(mask, mask);
}
#else
bool checkNan(__m256 v) {
    return false;
}
#endif // DEBUG


__m256 sphereSDF(Vec3x8& p) {
    const __m256 radius = _mm256_set1_ps(1.0f);
    __m256 length = p.length();
    // assert if length is not a number
    assert(!hasNan(length));
    return _mm256_sub_ps(length, radius);
}

__m256 sceneSDF(Vec3x8& p) {
    return sphereSDF(p);
}

Vec3x8 estimateNormal(Vec3x8& const p) {
    const __m256 eps = _mm256_set1_ps(MIN_DIST);

    Vec3x8 px = p + Vec3x8(eps, _mm256_setzero_ps(), _mm256_setzero_ps());
    Vec3x8 nx = p - Vec3x8(eps, _mm256_setzero_ps(), _mm256_setzero_ps());

    Vec3x8 py = p + Vec3x8(_mm256_setzero_ps(), eps, _mm256_setzero_ps());
    Vec3x8 ny = p - Vec3x8(_mm256_setzero_ps(), eps, _mm256_setzero_ps());

    Vec3x8 pz = p + Vec3x8(_mm256_setzero_ps(), _mm256_setzero_ps(), eps);
    Vec3x8 nz = p - Vec3x8(_mm256_setzero_ps(), _mm256_setzero_ps(), eps);

    // Compute SDF values once per offset
    __m256 sdf_px = sceneSDF(px);
    __m256 sdf_nx = sceneSDF(nx);
    __m256 sdf_py = sceneSDF(py);
    __m256 sdf_ny = sceneSDF(ny);
    __m256 sdf_pz = sceneSDF(pz);
    __m256 sdf_nz = sceneSDF(nz);

    // Compute the gradient
    __m256 nx_grad = _mm256_sub_ps(sdf_px, sdf_nx);
    __m256 ny_grad = _mm256_sub_ps(sdf_py, sdf_ny);
    __m256 nz_grad = _mm256_sub_ps(sdf_pz, sdf_nz);

    return Vec3x8(nx_grad, ny_grad, nz_grad).normalize();
}

// Simd helper function to set Vec3x8 color to image
void setColorToImage(std::vector<unsigned char>& image, Vec3x8& color, __m256 xs, __m256 ys) {
    for (int i = 0; i < 8; i++) {
        int index = ((int)ys.m256_f32[i] * WIDTH + (int)xs.m256_f32[i]) * 3;
        if (index < WIDTH * HEIGHT * 3) {
            image[index] = static_cast<unsigned char>(color.x256.m256_f32[i]);
        }
        if (index + 1 < WIDTH * HEIGHT * 3) {
            image[index + 1] = static_cast<unsigned char>(color.y256.m256_f32[i]);
        }
        if (index + 2 < WIDTH * HEIGHT * 3) {
            image[index + 2] = static_cast<unsigned char>(color.z256.m256_f32[i]);
        }
    }
}

void ray_march(Vec3x8 origins, Camera& camera, std::vector<unsigned char>& image) {
    auto [xs, ys, zs] = origins;
    Vec3x8 directions = camera.get_ray_directions(xs, ys);
    Vec3x8 ray_origins(camera.position);
    __m256 distances = _mm256_set1_ps(0.0);


    Vec3x8 color(0.f);
    __m256 activeMask = _mm256_set1_ps(-std::numeric_limits<float>::signaling_NaN());
    for (int i = 0; i < MAX_STEPS; i++) {
        color.resetColor();

        Vec3x8 p = ray_origins + directions * distances;
        __m256 dists = sceneSDF(p);
        __m256 mask = _mm256_cmp_ps(dists, _mm256_set1_ps(MIN_DIST), _CMP_LT_OS);
        //assert(!hasNan(mask));
        __m256 maskInv2 = _mm256_cmp_ps(distances, _mm256_set1_ps(MAX_DIST), _CMP_GE_OS);

        // check if any of the mask is non zero
        if (!_mm256_testz_ps(mask, mask)) {
            Vec3x8 normals = estimateNormal(p);
            color.multiplyWithMask(normals, mask);
        }

        //// invert mask
        __m256 maskInv = _mm256_cmp_ps(dists, _mm256_set1_ps(MIN_DIST), _CMP_GT_OS);
        __m256 newDistances = _mm256_add_ps(distances, dists);

        distances = _mm256_blendv_ps(newDistances, distances, maskInv);
        //_mm256_setr_ps(-1, -1, -1, -1, 0, 0, 0, 0);//
        __m256 mask2 = _mm256_cmp_ps(distances, _mm256_set1_ps(static_cast<float>(MAX_DIST)), _CMP_GT_OS);

        color.multiplyWithMask(Vec3x8(0.0f), mask2);
    }

    ////// clamp color
    color.x256 = _mm256_max_ps(_mm256_set1_ps(0.0f), _mm256_min_ps(_mm256_set1_ps(255.0f), color.x256));
    color.y256 = _mm256_max_ps(_mm256_set1_ps(0.0f), _mm256_min_ps(_mm256_set1_ps(255.0f), color.y256));
    color.z256 = _mm256_max_ps(_mm256_set1_ps(0.0f), _mm256_min_ps(_mm256_set1_ps(255.0f), color.z256));
    setColorToImage(image, color, xs, ys);
}


int main() {
    std::vector<unsigned char> image(WIDTH * HEIGHT * 3);

    Vec3 camera_position = Vec3(1., 0., 2.5);
    Vec3 look_at = Vec3(0, 0, 0);
    Vec3 up = Vec3(0, 1, 0);
    Camera camera(camera_position, look_at, up);

    auto start = std::chrono::high_resolution_clock::now();

    for (int y = 0; y < HEIGHT; y += 1) {
        for (int x = 0; x < WIDTH; x += 8) {
            __m256 us = _mm256_setr_ps(x, x + 1, x + 2, x + 3, x + 4, x + 5, x + 6, x + 7);
            __m256 vs = _mm256_set1_ps(y);
            ray_march(Vec3x8(us, vs, _mm256_set1_ps(0.0)), camera, image);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    stbi_write_png("output.png", WIDTH, HEIGHT, 3, image.data(), WIDTH * 3);

    std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;
    return 0;
}
