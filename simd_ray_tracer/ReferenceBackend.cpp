#include "ReferenceBackend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

namespace reference {

Vec3d Vec3d::operator+(Vec3d other) const {
  return {x + other.x, y + other.y, z + other.z};
}

Vec3d Vec3d::operator-(Vec3d other) const {
  return {x - other.x, y - other.y, z - other.z};
}

Vec3d Vec3d::operator*(double scalar) const {
  return {x * scalar, y * scalar, z * scalar};
}

double Vec3d::dot(Vec3d other) const {
  return x * other.x + y * other.y + z * other.z;
}

double Vec3d::length() const { return std::sqrt(dot(*this)); }

Vec3d Vec3d::normalize() const {
  double magnitude = length();
  return magnitude == 0.0 ? Vec3d{} : *this * (1.0 / magnitude);
}

namespace {

Vec3d from_float(Vec3 value) {
  return {static_cast<double>(value.x), static_cast<double>(value.y),
          static_cast<double>(value.z)};
}

render::Color shade(Vec3d position, render::RenderConfig const& config) {
  Vec3d normal = estimate_normal(position, config);
  std::array<double, render::kColorChannels> components{
      normal.x * 255.0,
      normal.y * 255.0,
      normal.z * 255.0,
  };
  render::Color color{};
  for (int channel = 0; channel < render::kColorChannels; ++channel) {
    color[channel] =
        static_cast<render::Pixel>(std::clamp(components[channel], 0.0, 255.0));
  }
  return color;
}

}  // namespace

Vec3d ray_direction(Camera const& camera, double x, double y,
                    render::RenderConfig const& config) {
  double width = static_cast<double>(config.width);
  double height = static_cast<double>(config.height);
  double aspect_ratio = width / height;
  double fov_degrees = static_cast<double>(config.fov_degrees);
  double fov_adjustment =
      std::tan(fov_degrees * 0.5 * std::numbers::pi_v<double> / 180.0);
  double x_adjustment =
      (2.0 * (x + 0.5) / width - 1.0) * aspect_ratio * fov_adjustment;
  double y_adjustment = (1.0 - 2.0 * (y + 0.5) / height) * fov_adjustment;
  return (from_float(camera.forward) + from_float(camera.right) * x_adjustment +
          from_float(camera.up) * y_adjustment)
      .normalize();
}

// Independent, readable power-N Mandelbulb distance estimate based on the
// spherical-coordinate formulation described by Inigo Quilez. Escape status
// is part of the result because the estimate is authoritative only outside the
// bounded set.
// https://iquilezles.org/articles/mandelbulb/
DistanceSample sample_sdf(Vec3d position, render::RenderConfig const& config) {
  Vec3d z = position;
  double derivative = 1.0;
  double radius = 0.0;
  double power = static_cast<double>(config.power);
  double escape_radius = static_cast<double>(config.mandelbulb_escape_radius);

  radius = z.length();
  if (radius > escape_radius) {
    return {0.5 * std::log(radius) * radius / derivative, true, 0};
  }

  for (int iteration = 0; iteration < config.mandelbulb_iterations;
       ++iteration) {
    radius = z.length();
    if (radius == 0.0) {
      return {0.0, false, iteration};
    }

    double polar = std::acos(std::clamp(z.y / radius, -1.0, 1.0));
    double azimuth = std::atan2(z.x, z.z);
    double radius_power = std::pow(radius, power);
    derivative = std::pow(radius, power - 1.0) * power * derivative + 1.0;
    polar *= power;
    azimuth *= power;

    double sin_polar = std::sin(polar);
    z = position + Vec3d{sin_polar * std::sin(azimuth), std::cos(polar),
                         sin_polar * std::cos(azimuth)} *
                       radius_power;
    radius = z.length();
    if (radius > escape_radius) {
      return {0.5 * std::log(radius) * radius / derivative, true,
              iteration + 1};
    }
  }

  return {0.5 * std::log(radius) * radius / derivative, false,
          config.mandelbulb_iterations};
}

double scene_sdf(Vec3d position, render::RenderConfig const& config) {
  return sample_sdf(position, config).distance;
}

Vec3d estimate_normal(Vec3d position, render::RenderConfig const& config) {
  double epsilon = static_cast<double>(config.min_distance);
  Vec3d dx{epsilon, 0.0, 0.0};
  Vec3d dy{0.0, epsilon, 0.0};
  Vec3d dz{0.0, 0.0, epsilon};
  return Vec3d{
      scene_sdf(position + dx, config) - scene_sdf(position - dx, config),
      scene_sdf(position + dy, config) - scene_sdf(position - dy, config),
      scene_sdf(position + dz, config) - scene_sdf(position - dz, config),
  }
      .normalize();
}

render::Color trace_ray(Camera const& camera, int x, int y,
                        render::RenderConfig const& config) {
  Vec3d direction = ray_direction(camera, static_cast<double>(x),
                                  static_cast<double>(y), config);
  Vec3d origin = from_float(camera.position);
  double distance = 0.0;
  double minimum_distance = static_cast<double>(config.min_distance);
  double maximum_distance = static_cast<double>(config.max_distance);

  for (int step = 0; step < config.max_steps; ++step) {
    Vec3d position = origin + direction * distance;
    double sdf = scene_sdf(position, config);
    if (sdf < minimum_distance) {
      return shade(position, config);
    }
    distance += sdf;
    if (distance > maximum_distance) {
      break;
    }
  }
  return {};
}

void render(Camera const& camera, render::RenderConfig const& config,
            render::Image& image) {
  for (int y = 0; y < config.height; ++y) {
    for (int x = 0; x < config.width; ++x) {
      render::Color color = trace_ray(camera, x, y, config);
      for (int channel = 0; channel < render::kColorChannels; ++channel) {
        image[(y * config.width + x) * render::kColorChannels + channel] =
            color[channel];
      }
    }
  }
}

}  // namespace reference
