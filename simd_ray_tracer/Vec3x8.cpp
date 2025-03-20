#include "Vec3x8.h"

Vec3x8::Vec3x8() : x256(_mm256_set1_ps(0.0)), y256(_mm256_set1_ps(0.0)), z256(_mm256_set1_ps(0.0)) {}

Vec3x8::Vec3x8(__m256 s) : x256(s), y256(s), z256(s) {}

Vec3x8::Vec3x8(Vec3 v) : x256(_mm256_set1_ps(v.x)), y256(_mm256_set1_ps(v.y)), z256(_mm256_set1_ps(v.z)) {}

Vec3x8::Vec3x8(__m256 xs, __m256 ys, __m256 zs) : x256(xs), y256(ys), z256(zs) {}

Vec3x8 Vec3x8::operator+(const Vec3x8& other) const {
    return Vec3x8(_mm256_add_ps(x256, other.x256), _mm256_add_ps(y256, other.y256), _mm256_add_ps(z256, other.z256));
}

Vec3x8 Vec3x8::operator-(const Vec3x8& other) const {
    return Vec3x8(_mm256_sub_ps(x256, other.x256), _mm256_sub_ps(y256, other.y256), _mm256_sub_ps(z256, other.z256));
}

Vec3x8 Vec3x8::operator*(const Vec3x8& other) const {
    return Vec3x8(_mm256_mul_ps(x256, other.x256), _mm256_mul_ps(y256, other.y256), _mm256_mul_ps(z256, other.z256));
}

Vec3x8 Vec3x8::operator/(const Vec3x8& other) const {
    return Vec3x8(_mm256_div_ps(x256, other.x256), _mm256_div_ps(y256, other.y256), _mm256_div_ps(z256, other.z256));
}

Vec3x8 Vec3x8::operator*(double scalar) const {
    return Vec3x8(_mm256_mul_ps(x256, _mm256_set1_ps(scalar)), _mm256_mul_ps(y256, _mm256_set1_ps(scalar)), _mm256_mul_ps(z256, _mm256_set1_ps(scalar)));
}

Vec3x8 Vec3x8::operator/(double scalar) const {
    return Vec3x8(_mm256_div_ps(x256, _mm256_set1_ps(scalar)), _mm256_div_ps(y256, _mm256_set1_ps(scalar)), _mm256_div_ps(z256, _mm256_set1_ps(scalar)));
}

Vec3x8 Vec3x8::operator+(double scalar) const {
    return Vec3x8(_mm256_add_ps(x256, _mm256_set1_ps(scalar)), _mm256_add_ps(y256, _mm256_set1_ps(scalar)), _mm256_add_ps(z256, _mm256_set1_ps(scalar)));
}

Vec3x8 Vec3x8::operator-(double scalar) const {
    return Vec3x8(_mm256_sub_ps(x256, _mm256_set1_ps(scalar)), _mm256_sub_ps(y256, _mm256_set1_ps(scalar)), _mm256_sub_ps(z256, _mm256_set1_ps(scalar)));
}

void Vec3x8::addWithMask(const Vec3x8& other, __m256 mask) {
    x256 = _mm256_blendv_ps(x256, _mm256_add_ps(x256, other.x256), mask);
    y256 = _mm256_blendv_ps(y256, _mm256_add_ps(y256, other.y256), mask);
    z256 = _mm256_blendv_ps(z256, _mm256_add_ps(z256, other.z256), mask);
}

void Vec3x8::multiplyWithMask(const Vec3x8& other, __m256 mask) {
    x256 = _mm256_blendv_ps(x256, _mm256_mul_ps(x256, other.x256), mask);
    y256 = _mm256_blendv_ps(y256, _mm256_mul_ps(y256, other.y256), mask);
    z256 = _mm256_blendv_ps(z256, _mm256_mul_ps(z256, other.z256), mask);
}

__m256 Vec3x8::dotWithMask(const Vec3x8& other, __m256 mask, __m256 dotProd) const {
    return _mm256_blendv_ps(
        dotProd,
        _mm256_add_ps(
            _mm256_add_ps(
                _mm256_mul_ps(x256, other.x256),
                _mm256_mul_ps(y256, other.y256)
            ),
            _mm256_mul_ps(z256, other.z256)
        ),
        mask
    );
}

void Vec3x8::resetColor() {
    x256 = _mm256_set1_ps(255.f);
    y256 = _mm256_set1_ps(255.f);
    z256 = _mm256_set1_ps(255.f);
}

Vec3x8 Vec3x8::abs() const {
    return Vec3x8(
        _mm256_andnot_ps(
            _mm256_set1_ps(-0.0),
            x256
        ),
        _mm256_andnot_ps(
            _mm256_set1_ps(-0.0),
            y256
        ),
        _mm256_andnot_ps(
            _mm256_set1_ps(-0.0),
            z256
        )
    );
}

__m256 Vec3x8::dot(const Vec3x8& other) const {
    // xs * other.xs + ys * other.ys + zs * other.zs
    return _mm256_add_ps(
        _mm256_add_ps(
            _mm256_mul_ps(x256, other.x256),
            _mm256_mul_ps(y256, other.y256)
        ),
        _mm256_mul_ps(z256, other.z256)
    );
}

Vec3x8 Vec3x8::normalize() const {
    __m256 l = length();
    // xs / l, ys / l, zs / l
    // avoid dividing by zero
    if (!_mm256_testz_ps(l, l)) {
        return Vec3x8();
    }
    __m256 l_inv = _mm256_div_ps(_mm256_set1_ps(1.0), l);
    __m256 x = _mm256_mul_ps(x256, l_inv);
    __m256 y = _mm256_mul_ps(y256, l_inv);
    __m256 z = _mm256_mul_ps(z256, l_inv);
    return Vec3x8(x, y, z);
}

__m256 Vec3x8::length() const {
    // sqrt(xs^2 + ys^2 + zs^2)
    __m256 squared =
        _mm256_add_ps(
            _mm256_add_ps(
                _mm256_mul_ps(x256, x256),
                _mm256_mul_ps(y256, y256)
            ),
            _mm256_mul_ps(z256, z256)
        );
    return _mm256_sqrt_ps(_mm256_max_ps(squared, _mm256_setzero_ps()));
}
