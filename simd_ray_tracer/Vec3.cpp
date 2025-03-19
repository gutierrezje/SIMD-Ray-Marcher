#include "Vec3.h"

Vec3::Vec3() : x(0), y(0), z(0) {}
Vec3::Vec3(double v) : x(v), y(v), z(v) {}
Vec3 Vec3::abs() const {
    return Vec3(std::abs(x), std::abs(y), std::abs(z));
}

Vec3 Vec3::operator+(const Vec3& other) const {
    return Vec3(x + other.x, y + other.y, z + other.z);
}

Vec3 Vec3::operator-(const Vec3& other) const {
    return Vec3(x - other.x, y - other.y, z - other.z);
}

Vec3 Vec3::operator*(double scalar) const {
    return Vec3(x * scalar, y * scalar, z * scalar);
}

Vec3 Vec3::operator/(double scalar) const {
    return Vec3(x / scalar, y / scalar, z / scalar);
}

double Vec3::dot(const Vec3& other) const {
    return x * other.x + y * other.y + z * other.z;
}

Vec3 Vec3::cross(const Vec3& other) const {
    return Vec3(
        y * other.z - z * other.y,
        z * other.x - x * other.z,
        x * other.y - y * other.x
    );
}

double Vec3::length() const {
    return std::sqrt(x * x + y * y + z * z);
}

Vec3 Vec3::normalize() const {
    double l = length();
    return Vec3(x / l, y / l, z / l);
}

Vec3 Vec3::operator+=(const Vec3& other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
}

Vec3 Vec3::operator-=(const Vec3& other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
}