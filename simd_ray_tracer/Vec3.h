#pragma once

#include <cmath>

class Vec3 {
public:
    double x, y, z;
    Vec3(); // Default constructor
    Vec3(double v);
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& other) const;
    Vec3 operator-(const Vec3& other) const;
    Vec3 operator*(double scalar) const;
    Vec3 operator/(double scalar) const;
    Vec3 operator+=(const Vec3& other);
    Vec3 operator-=(const Vec3& other);
    double dot(const Vec3& other) const;
    Vec3 cross(const Vec3& other) const;
    double length() const;
    Vec3 normalize() const;
    Vec3 abs() const;
};