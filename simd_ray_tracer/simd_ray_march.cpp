#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include <immintrin.h>

#include "Camera.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h" // Or other image library

#include <limits>

#ifdef __GNUC__
#include "avx_mathfun.h"
#define _mm256_log_ps(x) log256_ps(x)
#endif

// Mandelbulb parameters
constexpr int ITERATIONS = 10;
constexpr double POWER = 8.0;

// Ray marching parameters
constexpr double MIN_DIST = 0.001;
constexpr double MAX_DIST = 100.0;
constexpr int MAX_STEPS = 100;

bool hasNan(__m256 v)
{
#ifdef NDEBUG
    __m256 mask = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
    return !_mm256_testz_ps(mask, mask);
#else
    return false;
#endif // DEBUG
}

__m256 optimMandelbulb(Vec3x8& p)
{
    Vec3x8 w(p.x256, p.y256, p.z256);
    __m256 m = w.dot(w);

    __m256 dz = _mm256_set1_ps(1.0f);
    __m256 apply_mask = _mm256_set1_ps(-std::numeric_limits<float>::signaling_NaN());
    __m256 break_mask = _mm256_set1_ps(0.0f);

    for (int i = 0; i < 4; i++) {
        __m256 m2 = _mm256_mul_ps(m, m);
        __m256 m4 = _mm256_mul_ps(m2, m2);
        // dz = 8.0 * sqrt(m4 * m2 * m) * dz + 1.0;
        __m256 temp_dz = _mm256_mul_ps(m4, _mm256_mul_ps(m2, m));
        temp_dz = _mm256_sqrt_ps(temp_dz);
        temp_dz = _mm256_mul_ps(_mm256_set1_ps(8.0f), _mm256_mul_ps(temp_dz, dz));
        dz = _mm256_add_ps(temp_dz, _mm256_set1_ps(1.0f));

        __m256 x = w.x256;
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x4 = _mm256_mul_ps(x2, x2);
        __m256 y = w.y256;
        __m256 y2 = _mm256_mul_ps(y, y);
        __m256 y4 = _mm256_mul_ps(y2, y2);
        __m256 z = w.z256;
        __m256 z2 = _mm256_mul_ps(z, z);
        __m256 z4 = _mm256_mul_ps(z2, z2);

        __m256 k3 = _mm256_add_ps(x2, z2);

        // float k2 = 1. / std::sqrt(k3 * k3 * k3 * k3 * k3 * k3 * k3);
        __m256 k3sq = _mm256_mul_ps(k3, k3);
        __m256 k2 = _mm256_mul_ps(k3sq, k3sq);
        k2 = _mm256_mul_ps(_mm256_mul_ps(k2, k3sq), k3);
        k2 = _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_sqrt_ps(k2));

        // float k1 = x4 + y4 + z4 - 6.0 * y2 * z2 - 6.0 * x2 * y2 + 2.0 * z2 * x2;
        __m256 k1 = _mm256_add_ps(x4, _mm256_add_ps(y4, z4));
        __m256 k1l = _mm256_mul_ps(_mm256_set1_ps(6.f), _mm256_mul_ps(y2, z2));
        __m256 k1m = _mm256_mul_ps(_mm256_set1_ps(6.f), _mm256_mul_ps(x2, y2));
        __m256 k1r = _mm256_mul_ps(_mm256_set1_ps(2.f), _mm256_mul_ps(z2, x2));
        k1 = _mm256_sub_ps(k1, k1l);
        k1 = _mm256_sub_ps(k1, k1m);
        k1 = _mm256_add_ps(k1, k1r);

        __m256 k4 = _mm256_add_ps(_mm256_sub_ps(x2, y2), z2);

        w.x256 = _mm256_mul_ps(_mm256_set1_ps(64.f), _mm256_mul_ps(x, _mm256_mul_ps(y, z)));
        w.x256 = _mm256_mul_ps(w.x256, _mm256_mul_ps(_mm256_sub_ps(x2, z2), k4));
        w.x256 = _mm256_mul_ps(
            w.x256,
            _mm256_add_ps(
                _mm256_sub_ps(
                    x4,
                    _mm256_mul_ps(
                        _mm256_set1_ps(6.f),
                        _mm256_mul_ps(
                            x2,
                            z2))),
                z4));
        w.x256 = _mm256_mul_ps(w.x256, _mm256_mul_ps(k1, k2));
        w.x256 = _mm256_add_ps(w.x256, p.x256);

        w.y256 = _mm256_mul_ps(
            _mm256_mul_ps(
                _mm256_set1_ps(-16.f),
                _mm256_mul_ps(y2, k3)),
            _mm256_mul_ps(k4, k4));
        w.y256 = _mm256_add_ps(w.y256, p.y256);
        w.y256 = _mm256_add_ps(
            w.y256,
            _mm256_mul_ps(k1, k1));

        __m256 wz1 = _mm256_mul_ps(
            _mm256_set1_ps(-8.f),
            _mm256_mul_ps(y, k4));
        __m256 wz21 = _mm256_mul_ps(x4, x4);
        __m256 wz22 = _mm256_mul_ps(
            _mm256_set1_ps(28.f),
            _mm256_mul_ps(
                x4,
                _mm256_mul_ps(
                    x2,
                    z2)));
        __m256 wz23 = _mm256_mul_ps(
            _mm256_set1_ps(70.f),
            _mm256_mul_ps(
                x4,
                z4));
        __m256 wz24 = _mm256_mul_ps(
            _mm256_set1_ps(28.f),
            _mm256_mul_ps(
                x2,
                _mm256_mul_ps(
                    z2,
                    z4)));
        __m256 wz25 = _mm256_mul_ps(z4, z4);
        __m256 wz2 = _mm256_add_ps(
            _mm256_sub_ps(
                _mm256_add_ps(
                    _mm256_sub_ps(
                        wz21, wz22),
                    wz23),
                wz24),
            wz25);
        __m256 wz3 = _mm256_mul_ps(k1, k2);
        w.z256 = _mm256_add_ps(
            p.z256,
            _mm256_mul_ps(
                wz1,
                _mm256_mul_ps(wz2, wz3)));

        m = w.dotWithMask(w, apply_mask, m);
        apply_mask = _mm256_cmp_ps(m, _mm256_set1_ps(256.0f), _CMP_LT_OS);

        break_mask = _mm256_cmp_ps(m, _mm256_set1_ps(256.0f), _CMP_GT_OS);
        if (!_mm256_testz_ps(break_mask, break_mask)) {
            break;
        }
    }
    return _mm256_div_ps(
        _mm256_mul_ps(
            _mm256_set1_ps(0.25f),
            _mm256_mul_ps(
                _mm256_log_ps(m),
                _mm256_sqrt_ps(m))),
        dz);
}

__m256 sphereSDF(Vec3x8& p)
{
    const __m256 radius = _mm256_set1_ps(1.0f);
    __m256 length = p.length();
    // assert if length is not a number
    assert(!hasNan(length));
    return _mm256_sub_ps(length, radius);
}

__m256 sceneSDF(Vec3x8& p)
{
    return optimMandelbulb(p);
}

Vec3x8 estimateNormal(Vec3x8& p)
{
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
void setColorToImage(std::vector<unsigned char>& image, Vec3x8& color, __m256 xs, __m256 ys)
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
        int index = ((int)ys_arr_lo[i] * WIDTH + (int)xs_arr_lo[i]) * 3;
        if (index < WIDTH * HEIGHT * 3) {
            image[index] = static_cast<unsigned char>(col_x_arr_lo[i]);
        }
        if (index + 1 < WIDTH * HEIGHT * 3) {
            image[index + 1] = static_cast<unsigned char>(col_y_arr_lo[i]);
        }
        if (index + 2 < WIDTH * HEIGHT * 3) {
            image[index + 2] = static_cast<unsigned char>(col_z_arr_lo[i]);
        }
    }
    for (int i = 0; i < 4; ++i) {
        int index = ((int)ys_arr_hi[i] * WIDTH + (int)xs_arr_hi[i]) * 3;
        if (index < WIDTH * HEIGHT * 3) {
            image[index] = static_cast<unsigned char>(col_x_arr_hi[i]);
        }
        if (index + 1 < WIDTH * HEIGHT * 3) {
            image[index + 1] = static_cast<unsigned char>(col_y_arr_hi[i]);
        }
        if (index + 2 < WIDTH * HEIGHT * 3) {
            image[index + 2] = static_cast<unsigned char>(col_z_arr_hi[i]);
        }
    }

#else
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
#endif
}

void ray_march(Vec3x8 origins, Camera& camera, std::vector<unsigned char>& image)
{
    auto [xs, ys, zs] = origins;
    Vec3x8 directions = camera.get_ray_directions(xs, ys);
    Vec3x8 ray_origins(camera.position);
    __m256 distances = _mm256_set1_ps(0.0);

    Vec3x8 color(0.0f);
    __m256 activeMask = _mm256_set1_ps(-std::numeric_limits<float>::signaling_NaN());

    for (int i = 0; i < MAX_STEPS; i++) {
        Vec3x8 p = ray_origins + directions * distances;
        __m256 dists = sceneSDF(p);

        // Check for rays that have reached the minimum distance
        __m256 mask = _mm256_cmp_ps(dists, _mm256_set1_ps(MIN_DIST), _CMP_LT_OS);
        // disable rays that have reached the minimum distance before
        mask = _mm256_and_ps(mask, activeMask);

        // check if any of the mask is non zero
        if (!_mm256_testz_ps(mask, mask)) {
            Vec3x8 normals = estimateNormal(p);
            color.addWithMask(Vec3x8(255.f), mask);
            color.multiplyWithMask(normals, mask);
        }

        // update distances
        // invert mask
        __m256 maskInv = _mm256_cmp_ps(dists, _mm256_set1_ps(MIN_DIST), _CMP_GT_OS);
        maskInv = _mm256_and_ps(maskInv, activeMask);
        __m256 newDistances = _mm256_add_ps(distances, dists);
        distances = _mm256_blendv_ps(distances, newDistances, maskInv);

        // Check for rays that have reached the maximum distance
        __m256 mask2 = _mm256_cmp_ps(distances, _mm256_set1_ps(MAX_DIST), _CMP_GT_OS);
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
    std::vector<unsigned char> image(WIDTH * HEIGHT * 3);

    Vec3 camera_position = Vec3(0., 0., 2.);
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
