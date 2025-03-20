#pragma once

#include <immintrin.h>
#include "Vec3.h"

class Vec3x8 {
public:
    __m256 x256;
    __m256 y256;
    __m256 z256;

    Vec3x8();
    Vec3x8(__m256 xs, __m256 ys, __m256 zs);
    Vec3x8(__m256 s);
    Vec3x8(Vec3 v);

    Vec3x8 operator+(const Vec3x8& other) const;
    Vec3x8 operator-(const Vec3x8& other) const;
    Vec3x8 operator*(const Vec3x8& other) const;
    Vec3x8 operator/(const Vec3x8& other) const;
    Vec3x8 operator*(double scalar) const;
    Vec3x8 operator/(double scalar) const;
    Vec3x8 operator+(double scalar) const;
    Vec3x8 operator-(double scalar) const;

    void addWithMask(const Vec3x8& other, __m256 mask);
    void multiplyWithMask(const Vec3x8& other, __m256 mask);
    __m256 dotWithMask(const Vec3x8& other, __m256 mask, __m256 dotProd) const;
    void resetColor();


    Vec3x8 abs() const;
    __m256 dot(const Vec3x8& other) const;
    Vec3x8 normalize() const;
    __m256 length() const;
};