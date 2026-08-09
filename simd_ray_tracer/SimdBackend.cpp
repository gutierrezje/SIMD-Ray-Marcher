#include <cassert>
#include <cmath>
#include <limits>

#include "SimdBackend.h"

#ifdef __GNUC__
#include "avx_mathfun.h"
#define _mm256_log_ps(x) log256_ps(x)
#endif

namespace simd8 {

namespace {

bool has_nan(__m256 v)
{
    __m256 mask = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
    return !_mm256_testz_ps(mask, mask);
}

__m256 optim_mandelbulb(Vec3x8& p)
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

__m256 sphere_sdf(Vec3x8& p)
{
    const __m256 radius = _mm256_set1_ps(1.0f);
    __m256 length = p.length();
    assert(!has_nan(length));
    return _mm256_sub_ps(length, radius);
}

} // namespace

Vec3x8 ray_directions(Camera const& camera, __m256 x, __m256 y,
                      render::RenderConfig const& config) {
    __m256 aspect_ratio = _mm256_set1_ps(static_cast<float>(config.width) / static_cast<float>(config.height));
    __m256 fov_adjustment = _mm256_set1_ps(std::tan(config.fov_degrees * 0.5f * render::kPi / 180.0f));

    // x_adjustment = (2.0 * (x + 0.5) / width - 1.0) * aspect_ratio * fov_adjustment
    __m256 x_centered = _mm256_add_ps(x, _mm256_set1_ps(0.5f));
    __m256 x_scaled = _mm256_mul_ps(x_centered, _mm256_set1_ps(2.0f / static_cast<float>(config.width)));
    __m256 x_adjustment = _mm256_mul_ps(
        _mm256_sub_ps(x_scaled, _mm256_set1_ps(1.0f)),
        _mm256_mul_ps(aspect_ratio, fov_adjustment)
    );
    // y_adjustment = (1.0 - 2.0 * (y + 0.5) / height) * fov_adjustment
    __m256 y_centered = _mm256_add_ps(y, _mm256_set1_ps(0.5f));
    __m256 y_scaled = _mm256_mul_ps(y_centered, _mm256_set1_ps(2.0f / static_cast<float>(config.height)));
    __m256 y_adjustment = _mm256_mul_ps(
        _mm256_sub_ps(_mm256_set1_ps(1.0f), y_scaled),
        fov_adjustment
    );

    Vec3x8 ray_directions = Vec3x8(camera.forward) + Vec3x8(camera.right) * x_adjustment + Vec3x8(camera.up) * y_adjustment;
    return ray_directions.normalize();
}

__m256 scene_sdf(Vec3x8& p) {
    return optim_mandelbulb(p);
}

Vec3x8 estimate_normal(Vec3x8& p, render::RenderConfig const& config) {
    const __m256 eps = _mm256_set1_ps(config.min_distance);

    Vec3x8 px = p + Vec3x8(eps, _mm256_setzero_ps(), _mm256_setzero_ps());
    Vec3x8 nx = p - Vec3x8(eps, _mm256_setzero_ps(), _mm256_setzero_ps());

    Vec3x8 py = p + Vec3x8(_mm256_setzero_ps(), eps, _mm256_setzero_ps());
    Vec3x8 ny = p - Vec3x8(_mm256_setzero_ps(), eps, _mm256_setzero_ps());

    Vec3x8 pz = p + Vec3x8(_mm256_setzero_ps(), _mm256_setzero_ps(), eps);
    Vec3x8 nz = p - Vec3x8(_mm256_setzero_ps(), _mm256_setzero_ps(), eps);

    __m256 sdf_px = scene_sdf(px);
    __m256 sdf_nx = scene_sdf(nx);
    __m256 sdf_py = scene_sdf(py);
    __m256 sdf_ny = scene_sdf(ny);
    __m256 sdf_pz = scene_sdf(pz);
    __m256 sdf_nz = scene_sdf(nz);

    __m256 nx_grad = _mm256_sub_ps(sdf_px, sdf_nx);
    __m256 ny_grad = _mm256_sub_ps(sdf_py, sdf_ny);
    __m256 nz_grad = _mm256_sub_ps(sdf_pz, sdf_nz);

    return Vec3x8(nx_grad, ny_grad, nz_grad).normalize();
}

} // namespace simd8
