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

__m256 optim_mandelbulb(Vec3x8& p,
                        render::RenderConfig const& config)
{
    Vec3x8 w(p.x256, p.y256, p.z256);
    __m256 m = w.dot(w);

    __m256 dz = _mm256_set1_ps(1.0f);
    const __m256 escape_threshold = _mm256_set1_ps(
        config.mandelbulb_escape_radius * config.mandelbulb_escape_radius);
    __m256 active_mask = _mm256_cmp_ps(
        _mm256_setzero_ps(), _mm256_setzero_ps(), _CMP_EQ_OQ);

    for (int iteration = 0;
         iteration < config.mandelbulb_iterations;
         iteration += 1) {
        __m256 m2 = _mm256_mul_ps(m, m);
        __m256 m4 = _mm256_mul_ps(m2, m2);
        // dz = 8.0 * sqrt(m4 * m2 * m) * dz + 1.0;
        __m256 next_dz = _mm256_mul_ps(m4, _mm256_mul_ps(m2, m));
        next_dz = _mm256_sqrt_ps(next_dz);
        next_dz = _mm256_mul_ps(
            _mm256_set1_ps(8.0f), _mm256_mul_ps(next_dz, dz));
        next_dz = _mm256_add_ps(next_dz, _mm256_set1_ps(1.0f));

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

        Vec3x8 next_w;
        next_w.x256 = _mm256_mul_ps(_mm256_set1_ps(64.f), _mm256_mul_ps(x, _mm256_mul_ps(y, z)));
        next_w.x256 = _mm256_mul_ps(next_w.x256, _mm256_mul_ps(_mm256_sub_ps(x2, z2), k4));
        next_w.x256 = _mm256_mul_ps(
            next_w.x256,
            _mm256_add_ps(
                _mm256_sub_ps(
                    x4,
                    _mm256_mul_ps(
                        _mm256_set1_ps(6.f),
                        _mm256_mul_ps(
                            x2,
                            z2))),
                z4));
        next_w.x256 = _mm256_mul_ps(next_w.x256, _mm256_mul_ps(k1, k2));
        next_w.x256 = _mm256_add_ps(next_w.x256, p.x256);

        next_w.y256 = _mm256_mul_ps(
            _mm256_mul_ps(
                _mm256_set1_ps(-16.f),
                _mm256_mul_ps(y2, k3)),
            _mm256_mul_ps(k4, k4));
        next_w.y256 = _mm256_add_ps(next_w.y256, p.y256);
        next_w.y256 = _mm256_add_ps(
            next_w.y256,
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
        next_w.z256 = _mm256_add_ps(
            p.z256,
            _mm256_mul_ps(
                wz1,
                _mm256_mul_ps(wz2, wz3)));

        __m256 next_m = next_w.dot(next_w);
        w.x256 = _mm256_blendv_ps(w.x256, next_w.x256, active_mask);
        w.y256 = _mm256_blendv_ps(w.y256, next_w.y256, active_mask);
        w.z256 = _mm256_blendv_ps(w.z256, next_w.z256, active_mask);
        dz = _mm256_blendv_ps(dz, next_dz, active_mask);
        m = _mm256_blendv_ps(m, next_m, active_mask);

        __m256 escaped_mask = _mm256_cmp_ps(
            next_m, escape_threshold, _CMP_GT_OS);
        active_mask = _mm256_andnot_ps(escaped_mask, active_mask);
        if (_mm256_testz_ps(active_mask, active_mask)) {
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

void set_color_to_image(render::Image& image, Vec3x8& color, __m256 xs,
                        __m256 ys, render::RenderConfig const& config)
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
        int index = ((int)ys_arr_lo[i] * config.width + (int)xs_arr_lo[i]) * render::kColorChannels;
        if (index < config.width * config.height * render::kColorChannels) {
            image[index] = static_cast<unsigned char>(col_x_arr_lo[i]);
        }
        if (index + 1 < config.width * config.height * render::kColorChannels) {
            image[index + 1] = static_cast<unsigned char>(col_y_arr_lo[i]);
        }
        if (index + 2 < config.width * config.height * render::kColorChannels) {
            image[index + 2] = static_cast<unsigned char>(col_z_arr_lo[i]);
        }
    }
    for (int i = 0; i < 4; ++i) {
        int index = ((int)ys_arr_hi[i] * config.width + (int)xs_arr_hi[i]) * render::kColorChannels;
        if (index < config.width * config.height * render::kColorChannels) {
            image[index] = static_cast<unsigned char>(col_x_arr_hi[i]);
        }
        if (index + 1 < config.width * config.height * render::kColorChannels) {
            image[index + 1] = static_cast<unsigned char>(col_y_arr_hi[i]);
        }
        if (index + 2 < config.width * config.height * render::kColorChannels) {
            image[index + 2] = static_cast<unsigned char>(col_z_arr_hi[i]);
        }
    }

#else
    for (int i = 0; i < 8; i++) {
        int index = ((int)ys.m256_f32[i] * config.width + (int)xs.m256_f32[i]) * render::kColorChannels;
        if (index < config.width * config.height * render::kColorChannels) {
            image[index] = static_cast<unsigned char>(color.x256.m256_f32[i]);
        }
        if (index + 1 < config.width * config.height * render::kColorChannels) {
            image[index + 1] = static_cast<unsigned char>(color.y256.m256_f32[i]);
        }
        if (index + 2 < config.width * config.height * render::kColorChannels) {
            image[index + 2] = static_cast<unsigned char>(color.z256.m256_f32[i]);
        }
    }
#endif
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

__m256 scene_sdf(Vec3x8& p, render::RenderConfig const& config) {
    return optim_mandelbulb(p, config);
}

Vec3x8 estimate_normal(Vec3x8& p, render::RenderConfig const& config) {
    const __m256 eps = _mm256_set1_ps(config.min_distance);

    Vec3x8 px = p + Vec3x8(eps, _mm256_setzero_ps(), _mm256_setzero_ps());
    Vec3x8 nx = p - Vec3x8(eps, _mm256_setzero_ps(), _mm256_setzero_ps());

    Vec3x8 py = p + Vec3x8(_mm256_setzero_ps(), eps, _mm256_setzero_ps());
    Vec3x8 ny = p - Vec3x8(_mm256_setzero_ps(), eps, _mm256_setzero_ps());

    Vec3x8 pz = p + Vec3x8(_mm256_setzero_ps(), _mm256_setzero_ps(), eps);
    Vec3x8 nz = p - Vec3x8(_mm256_setzero_ps(), _mm256_setzero_ps(), eps);

    __m256 sdf_px = scene_sdf(px, config);
    __m256 sdf_nx = scene_sdf(nx, config);
    __m256 sdf_py = scene_sdf(py, config);
    __m256 sdf_ny = scene_sdf(ny, config);
    __m256 sdf_pz = scene_sdf(pz, config);
    __m256 sdf_nz = scene_sdf(nz, config);

    __m256 nx_grad = _mm256_sub_ps(sdf_px, sdf_nx);
    __m256 ny_grad = _mm256_sub_ps(sdf_py, sdf_ny);
    __m256 nz_grad = _mm256_sub_ps(sdf_pz, sdf_nz);

    return Vec3x8(nx_grad, ny_grad, nz_grad).normalize();
}

MarchStep march_step(Vec3x8 const& origins, Vec3x8 const& directions,
                     __m256 distance, render::RenderConfig const& config) {
    Vec3x8 position = origins + directions * distance;
    return {position, distance, scene_sdf(position, config)};
}

__m256 advance_distance(MarchStep const& step, __m256 active_mask,
                        render::RenderConfig const& config) {
    __m256 continue_mask = _mm256_cmp_ps(
        step.sdf, _mm256_set1_ps(config.min_distance), _CMP_GT_OS);
    continue_mask = _mm256_and_ps(continue_mask, active_mask);
    __m256 next_distance = _mm256_add_ps(step.distance, step.sdf);
    return _mm256_blendv_ps(step.distance, next_distance, continue_mask);
}

__m256 hit_mask(MarchStep const& step, render::RenderConfig const& config) {
    return _mm256_cmp_ps(step.sdf, _mm256_set1_ps(config.min_distance),
                         _CMP_LT_OS);
}

__m256 miss_mask(__m256 distance, render::RenderConfig const& config) {
    return _mm256_cmp_ps(distance, _mm256_set1_ps(config.max_distance),
                         _CMP_GT_OS);
}

void apply_hit_color(Vec3x8& color, Vec3x8 const& normal, __m256 mask) {
    color.addWithMask(Vec3x8(255.f), mask);
    color.multiplyWithMask(normal, mask);
}

void clamp_color(Vec3x8& color) {
    color.x256 = _mm256_max_ps(
        _mm256_set1_ps(0.0f),
        _mm256_min_ps(_mm256_set1_ps(255.0f), color.x256));
    color.y256 = _mm256_max_ps(
        _mm256_set1_ps(0.0f),
        _mm256_min_ps(_mm256_set1_ps(255.0f), color.y256));
    color.z256 = _mm256_max_ps(
        _mm256_set1_ps(0.0f),
        _mm256_min_ps(_mm256_set1_ps(255.0f), color.z256));
}

Vec3x8 trace_ray_packet(Camera const& camera, __m256 xs, __m256 ys,
                        render::RenderConfig const& config) {
    Vec3x8 directions = ray_directions(camera, xs, ys, config);
    Vec3x8 ray_origins(camera.position);
    __m256 distances = _mm256_set1_ps(0.0f);
    __m256 active_mask =
        _mm256_set1_ps(-std::numeric_limits<float>::signaling_NaN());
    Vec3x8 color(0.0f);

    for (int step_count = 0; step_count < config.max_steps; ++step_count) {
        MarchStep step = march_step(ray_origins, directions, distances, config);

        __m256 hits = _mm256_and_ps(hit_mask(step, config), active_mask);
        if (!_mm256_testz_ps(hits, hits)) {
            Vec3x8 normals = estimate_normal(step.position, config);
            apply_hit_color(color, normals, hits);
        }

        distances = advance_distance(step, active_mask, config);

        __m256 misses = _mm256_and_ps(miss_mask(distances, config),
                                      active_mask);
        if (!_mm256_testz_ps(misses, misses)) {
            color.multiplyWithMask(Vec3x8(0.0f), misses);
        }

        __m256 terminate_mask = _mm256_or_ps(hits, misses);
        active_mask = _mm256_andnot_ps(terminate_mask, active_mask);
        if (_mm256_testz_ps(active_mask, active_mask)) {
            break;
        }
    }

    clamp_color(color);
    return color;
}

void render(Camera const& camera, render::RenderConfig const& config,
            render::Image& image) {
    for (int y = 0; y < config.height; ++y) {
        for (int x = 0; x < config.width; x += 8) {
            __m256 xs = _mm256_setr_ps(
                x, x + 1, x + 2, x + 3, x + 4, x + 5, x + 6, x + 7);
            __m256 ys = _mm256_set1_ps(y);
            Vec3x8 color = trace_ray_packet(camera, xs, ys, config);
            set_color_to_image(image, color, xs, ys, config);
        }
    }
}

} // namespace simd8
