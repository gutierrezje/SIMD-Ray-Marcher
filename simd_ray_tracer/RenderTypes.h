#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "RenderConfig.h"
#include "Vec3.h"

namespace render {

using Pixel = std::uint8_t;

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

struct RayResult {
    bool hit = false;
    int steps = 0;
    float distance = 0.0f;
    Vec3 position{};
    Vec3 normal{};
    std::array<Pixel, kColorChannels> color{};
};

} // namespace render
