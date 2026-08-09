#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <vector>

namespace render {

inline constexpr int kColorChannels = 3;
inline constexpr float kPi = std::numbers::pi_v<float>;

struct RenderConfig {
    int width = 2056;
    int height = 2056;
    int mandelbulb_iterations = 10;
    float power = 8.0f;
    float mandelbulb_escape_radius = 16.0f;
    float min_distance = 0.001f;
    float max_distance = 100.0f;
    int max_steps = 100;
    float fov_degrees = 45.0f;
};

inline constexpr RenderConfig kDefaultRenderConfig{};

using Pixel = std::uint8_t;
using Color = std::array<Pixel, kColorChannels>;

class Image {
public:
    explicit Image(const RenderConfig& config)
        : width_(config.width),
          height_(config.height),
          pixels_(static_cast<std::size_t>(width_) *
                  static_cast<std::size_t>(height_) *
                  kColorChannels) {}

    int width() const noexcept { return width_; }
    int height() const noexcept { return height_; }
    std::size_t row_stride() const noexcept {
        return static_cast<std::size_t>(width_) * kColorChannels;
    }

    Pixel* data() noexcept { return pixels_.data(); }
    const Pixel* data() const noexcept { return pixels_.data(); }
    std::size_t size() const noexcept { return pixels_.size(); }

    Pixel& operator[](std::size_t index) noexcept { return pixels_[index]; }
    const Pixel& operator[](std::size_t index) const noexcept {
        return pixels_[index];
    }

private:
    int width_;
    int height_;
    std::vector<Pixel> pixels_;
};

} // namespace render
