#pragma once

#include "Camera.h"
#include "RenderTypes.h"

namespace reference {

struct Vec3d {
  double x;
  double y;
  double z;

  Vec3d operator+(Vec3d other) const;
  Vec3d operator-(Vec3d other) const;
  Vec3d operator*(double scalar) const;
  double dot(Vec3d other) const;
  double length() const;
  Vec3d normalize() const;
};

struct DistanceSample {
  double distance;
  bool escaped;
  int iterations;
};

Vec3d ray_direction(Camera const& camera, double x, double y,
                    render::RenderConfig const& config);
DistanceSample sample_sdf(Vec3d position, render::RenderConfig const& config);
double scene_sdf(Vec3d position, render::RenderConfig const& config);
Vec3d estimate_normal(Vec3d position, render::RenderConfig const& config);
render::Color trace_ray(Camera const& camera, int x, int y,
                        render::RenderConfig const& config);
void render(Camera const& camera, render::RenderConfig const& config,
            render::Image& image);

}  // namespace reference
