#pragma once

#include <numbers>

namespace render {

inline constexpr int kColorChannels = 3;
inline constexpr float kPi = std::numbers::pi_v<float>;

struct RenderConfig {
    int width = 2056;
    int height = 2056;
    int mandelbulb_iterations = 10;
    float power = 8.0f;
    float min_distance = 0.001f;
    float max_distance = 100.0f;
    int max_steps = 100;
    float fov_degrees = 45.0f;
};

inline constexpr RenderConfig kDefaultRenderConfig{};

} // namespace render
