#pragma once

#include <cmath>

class Vec3 {
public:
    float x, y, z;
    Vec3(); // Default constructor
    Vec3(float v);
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& other) const;
    Vec3 operator-(const Vec3& other) const;
    Vec3 operator*(float scalar) const;
    Vec3 operator/(float scalar) const;
    Vec3 operator+=(const Vec3& other);
    Vec3 operator-=(const Vec3& other);
    float dot(const Vec3& other) const;
    Vec3 cross(const Vec3& other) const;
    float length() const;
    Vec3 normalize() const;
    Vec3 abs() const;
};
