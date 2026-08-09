#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

#include "RenderTypes.h"
#include "ScalarBackend.h"
#include "SimdBackend.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace {

enum class Backend {
    scalar,
    simd,
    compare,
};

struct DiffStats {
    std::size_t differing_channels = 0;
    int max_absolute_difference = 0;
    std::uint64_t total_absolute_difference = 0;
};

constexpr const render::RenderConfig& kConfig =
    render::kDefaultRenderConfig;

void print_usage(char const* executable) {
    std::cout << "Usage: " << executable
              << " [--backend=scalar|simd|compare]\n";
}

bool parse_backend(std::string_view value, Backend& backend) {
    if (value == "scalar") {
        backend = Backend::scalar;
        return true;
    }
    if (value == "simd") {
        backend = Backend::simd;
        return true;
    }
    if (value == "compare") {
        backend = Backend::compare;
        return true;
    }
    return false;
}

template <typename Function>
double measure_seconds(Function&& function) {
    auto start = std::chrono::steady_clock::now();
    function();
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

bool write_png(char const* path, render::Image const& image) {
    return stbi_write_png(path, image.width(), image.height(),
                          render::kColorChannels, image.data(),
                          static_cast<int>(image.row_stride())) != 0;
}

DiffStats compare_images(render::Image const& scalar_image,
                         render::Image const& simd_image) {
    DiffStats stats;
    for (std::size_t index = 0; index < scalar_image.size(); ++index) {
        int scalar_value = scalar_image[index];
        int simd_value = simd_image[index];
        int difference = scalar_value > simd_value
            ? scalar_value - simd_value
            : simd_value - scalar_value;
        if (difference != 0) {
            ++stats.differing_channels;
        }
        if (difference > stats.max_absolute_difference) {
            stats.max_absolute_difference = difference;
        }
        stats.total_absolute_difference +=
            static_cast<std::uint64_t>(difference);
    }
    return stats;
}

Camera default_camera() {
    return make_camera(
        Vec3(0.0f, 0.0f, 2.0f),
        Vec3(0.0f, 0.0f, 0.0f),
        Vec3(0.0f, 1.0f, 0.0f));
}

int run_single_backend(Backend backend, Camera const& camera) {
    render::Image image(kConfig);
    double elapsed = measure_seconds([&] {
        if (backend == Backend::scalar) {
            scalar::render(camera, kConfig, image);
        } else {
            simd8::render(camera, kConfig, image);
        }
    });

    if (!write_png("output.png", image)) {
        std::cerr << "Failed to write output.png\n";
        return 1;
    }

    std::cout << "Backend: "
              << (backend == Backend::scalar ? "scalar" : "simd") << '\n'
              << "Elapsed time: " << elapsed << " s\n"
              << "Wrote: output.png\n";
    return 0;
}

int run_comparison(Camera const& camera) {
    render::Image scalar_image(kConfig);
    render::Image simd_image(kConfig);

    double scalar_elapsed = measure_seconds([&] {
        scalar::render(camera, kConfig, scalar_image);
    });
    double simd_elapsed = measure_seconds([&] {
        simd8::render(camera, kConfig, simd_image);
    });

    DiffStats stats = compare_images(scalar_image, simd_image);
    double mean_absolute_difference =
        static_cast<double>(stats.total_absolute_difference) /
        static_cast<double>(scalar_image.size());

    bool scalar_written = write_png("scalar_output.png", scalar_image);
    bool simd_written = write_png("simd_output.png", simd_image);
    if (!scalar_written || !simd_written) {
        std::cerr << "Failed to write comparison images\n";
        return 1;
    }

    std::cout << "Scalar elapsed time: " << scalar_elapsed << " s\n"
              << "SIMD elapsed time: " << simd_elapsed << " s\n"
              << "Speedup: " << scalar_elapsed / simd_elapsed << "x\n"
              << "Differing channels: " << stats.differing_channels << " / "
              << scalar_image.size() << '\n'
              << "Maximum absolute channel difference: "
              << stats.max_absolute_difference << '\n'
              << "Mean absolute channel difference: "
              << mean_absolute_difference << '\n'
              << "Wrote: scalar_output.png, simd_output.png\n";
    return 0;
}

} // namespace

int main(int argc, char* argv[]) {
    Backend backend = Backend::simd;
    constexpr std::string_view backend_prefix = "--backend=";

    for (int index = 1; index < argc; ++index) {
        std::string_view argument = argv[index];
        if (argument == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        if (!argument.starts_with(backend_prefix) ||
            !parse_backend(argument.substr(backend_prefix.size()), backend)) {
            std::cerr << "Unknown argument: " << argument << '\n';
            print_usage(argv[0]);
            return 2;
        }
    }

    Camera camera = default_camera();
    if (backend == Backend::compare) {
        return run_comparison(camera);
    }
    return run_single_backend(backend, camera);
}
