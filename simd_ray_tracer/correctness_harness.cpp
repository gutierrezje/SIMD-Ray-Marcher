#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ReferenceBackend.h"
#include "ScalarBackend.h"
#include "SimdBackend.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace {

constexpr int kLanes = 8;
constexpr std::uint32_t kDefaultSeed = 0x5EEDU;
using LaneValues = std::array<float, kLanes>;
using PointBatch = std::array<Vec3, kLanes>;
using ColorLanes = std::array<LaneValues, render::kColorChannels>;

struct Comparison {
    std::string name;
    float tolerance = 0.0f;
    std::string expected_name = "scalar";
    std::string actual_name = "simd";
    bool track_ulp = true;
    std::size_t samples = 0;
    std::size_t skipped = 0;
    std::size_t mismatches = 0;
    std::size_t nonfinite_mismatches = 0;
    double absolute_error_sum = 0.0;
    float max_absolute_error = 0.0f;
    std::uint32_t max_ulp_error = 0;
    std::string first_mismatch;

    bool passed() const { return mismatches == 0; }
};

struct RenderPair {
    Comparison scalar_simd;
    Comparison reference_scalar;
    Comparison reference_simd;
};

enum class LocationKind : std::uint8_t {
    PixelCoord,
    BatchPoint,
    RayChannel,
    RenderChannel
};

// Formatted lazily only upon a mismatch.
struct LocationContext {
    LocationKind kind = LocationKind::PixelCoord;
    int int_a = 0;
    int int_b = 0;
    int int_c = 0;
    int int_d = 0;
    float float_a = 0.0f;
    Vec3 point = Vec3(0.0f, 0.0f, 0.0f);
    std::size_t index = 0;
    char const* suffix = "";
};

struct ImageTask {
    char const* filename;
    render::Image const* image;
};

// Per-kind constructors keep the generic slots readable at call sites.
LocationContext pixel_location(float x, int y) {
    LocationContext ctx;
    ctx.kind = LocationKind::PixelCoord;
    ctx.int_a = y;
    ctx.float_a = x;
    return ctx;
}

LocationContext batch_point_location(int batch, int lane, Vec3 point) {
    LocationContext ctx;
    ctx.kind = LocationKind::BatchPoint;
    ctx.int_a = batch;
    ctx.int_b = lane;
    ctx.point = point;
    return ctx;
}

LocationContext ray_channel_location(int lane, int x, int y, int channel) {
    LocationContext ctx;
    ctx.kind = LocationKind::RayChannel;
    ctx.int_a = lane;
    ctx.int_b = x;
    ctx.int_c = y;
    ctx.int_d = channel;
    return ctx;
}

LocationContext render_channel_location(std::size_t index) {
    LocationContext ctx;
    ctx.kind = LocationKind::RenderChannel;
    ctx.index = index;
    return ctx;
}

std::string format_location(LocationContext const& ctx) {
    switch (ctx.kind) {
    case LocationKind::PixelCoord:
        return "pixel(" + std::to_string(ctx.float_a) + "," +
               std::to_string(ctx.int_a) + ")" + ctx.suffix;
    case LocationKind::BatchPoint:
        return "batch " + std::to_string(ctx.int_a) + " lane " +
               std::to_string(ctx.int_b) + " p=(" +
               std::to_string(ctx.point.x) + "," + std::to_string(ctx.point.y) +
               "," + std::to_string(ctx.point.z) + ")" + ctx.suffix;
    case LocationKind::RayChannel:
        return "pixel lane " + std::to_string(ctx.int_a) + " at (" +
               std::to_string(ctx.int_b) + "," + std::to_string(ctx.int_c) +
               ") channel " + std::to_string(ctx.int_d);
    case LocationKind::RenderChannel:
        return "channel " + std::to_string(ctx.index);
    }
    return "";
}

std::uint32_t ordered_bits(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    return (bits & 0x80000000U) != 0 ? ~bits : bits | 0x80000000U;
}

std::uint32_t ulp_distance(float a, float b) {
    std::uint32_t lhs = ordered_bits(a);
    std::uint32_t rhs = ordered_bits(b);
    return lhs > rhs ? lhs - rhs : rhs - lhs;
}

void compare_value(Comparison& result, float expected, float actual,
                   LocationContext const& ctx) {
    ++result.samples;
    bool same_nonfinite = !std::isfinite(expected) && !std::isfinite(actual) &&
                          std::isnan(expected) == std::isnan(actual) &&
                          std::signbit(expected) == std::signbit(actual);
    if (!std::isfinite(expected) || !std::isfinite(actual)) {
        if (!same_nonfinite) {
            ++result.mismatches;
            ++result.nonfinite_mismatches;
            if (result.first_mismatch.empty()) {
                result.first_mismatch = format_location(ctx) + " " +
                                        result.expected_name + "=" +
                                        std::to_string(expected) + " " +
                                        result.actual_name + "=" +
                                        std::to_string(actual);
            }
        }
        return;
    }

    float error = std::abs(expected - actual);
    result.absolute_error_sum += static_cast<double>(error);
    result.max_absolute_error = std::max(result.max_absolute_error, error);
    if (result.track_ulp) {
        result.max_ulp_error =
            std::max(result.max_ulp_error, ulp_distance(expected, actual));
    }
    if (error > result.tolerance) {
        ++result.mismatches;
        if (result.first_mismatch.empty()) {
            result.first_mismatch = format_location(ctx) + " " +
                                    result.expected_name + "=" +
                                    std::to_string(expected) + " " +
                                    result.actual_name + "=" +
                                    std::to_string(actual);
        }
    }
}

LaneValues lanes(__m256 value) {
    LaneValues result{};
    _mm256_storeu_ps(result.data(), value);
    return result;
}

__m256 pack(LaneValues const& values) {
    return _mm256_loadu_ps(values.data());
}

void compare_vec3(Comparison& result, Vec3 expected,
                  LaneValues const& xs, LaneValues const& ys,
                  LaneValues const& zs, int lane,
                  LocationContext ctx) {
    ctx.suffix = ".x";
    compare_value(result, expected.x, xs[lane], ctx);
    ctx.suffix = ".y";
    compare_value(result, expected.y, ys[lane], ctx);
    ctx.suffix = ".z";
    compare_value(result, expected.z, zs[lane], ctx);
}

Comparison compare_camera(Camera const& camera,
                          render::RenderConfig const& config) {
    Comparison result{"scalar vs SIMD camera", 2.0e-6f};
    for (int y : {0, config.height / 3, config.height / 2, config.height - 1}) {
        LaneValues xs{};
        LaneValues ys{};
        for (int lane = 0; lane < kLanes; ++lane) {
            xs[lane] = static_cast<float>((lane * (config.width - 1)) / 7);
            ys[lane] = static_cast<float>(y);
        }
        Vec3x8 actual =
            simd8::ray_directions(camera, pack(xs), pack(ys), config);
        LaneValues ax = lanes(actual.x256);
        LaneValues ay = lanes(actual.y256);
        LaneValues az = lanes(actual.z256);
        for (int lane = 0; lane < kLanes; ++lane) {
            Vec3 expected =
                scalar::ray_direction(camera, xs[lane], ys[lane], config);
            compare_vec3(result, expected, ax, ay, az, lane,
                         pixel_location(xs[lane], y));
        }
    }
    return result;
}

PointBatch random_points(std::mt19937& generator, int batch) {
    std::uniform_real_distribution<float> outer(-3.0f, 3.0f);
    std::uniform_real_distribution<float> near_surface(-1.25f, 1.25f);
    PointBatch points{};
    for (int lane = 0; lane < kLanes; ++lane) {
        std::uniform_real_distribution<float>& distribution =
            ((batch + lane) % 2 == 0) ? outer : near_surface;
        points[lane] = Vec3(distribution(generator), distribution(generator),
                            distribution(generator));
    }
    return points;
}

Vec3x8 pack_points(PointBatch const& points) {
    LaneValues xs{}, ys{}, zs{};
    for (int lane = 0; lane < kLanes; ++lane) {
        xs[lane] = points[lane].x;
        ys[lane] = points[lane].y;
        zs[lane] = points[lane].z;
    }
    return Vec3x8(pack(xs), pack(ys), pack(zs));
}

reference::Vec3d to_reference(Vec3 point) {
    return {static_cast<double>(point.x), static_cast<double>(point.y),
            static_cast<double>(point.z)};
}

Comparison compare_reference_sdf(std::mt19937& generator,
                                 render::RenderConfig const& config,
                                 std::string name) {
    Comparison result{std::move(name), 1.0e-4f, "reference", "scalar"};
    for (int batch = 0; batch < 512; ++batch) {
        PointBatch points = random_points(generator, batch);
        for (int lane = 0; lane < kLanes; ++lane) {
            reference::DistanceSample expected =
                reference::sample_sdf(to_reference(points[lane]), config);
            if (!expected.escaped) {
                ++result.skipped;
                continue;
            }
            compare_value(result, static_cast<float>(expected.distance),
                          scalar::scene_sdf(points[lane]),
                          batch_point_location(batch, lane, points[lane]));
        }
    }
    return result;
}

Comparison compare_sdf(std::mt19937& generator) {
    Comparison result{"scalar vs SIMD SDF", 1.0e-5f};
    for (int batch = 0; batch < 512; ++batch) {
        PointBatch points = random_points(generator, batch);
        Vec3x8 packet = pack_points(points);
        LaneValues actual = lanes(simd8::scene_sdf(packet));
        for (int lane = 0; lane < kLanes; ++lane) {
            compare_value(result, scalar::scene_sdf(points[lane]),
                          actual[lane],
                          batch_point_location(batch, lane, points[lane]));
        }
    }
    return result;
}

Comparison compare_normals(std::mt19937& generator,
                           render::RenderConfig const& config) {
    Comparison result{"scalar vs SIMD normal", 1.0e-3f};
    for (int batch = 0; batch < 64; ++batch) {
        PointBatch points = random_points(generator, batch);
        Vec3x8 packet = pack_points(points);
        Vec3x8 actual = simd8::estimate_normal(packet, config);
        LaneValues ax = lanes(actual.x256);
        LaneValues ay = lanes(actual.y256);
        LaneValues az = lanes(actual.z256);
        for (int lane = 0; lane < kLanes; ++lane) {
            compare_vec3(result, scalar::estimate_normal(points[lane], config),
                         ax, ay, az, lane,
                         batch_point_location(batch, lane, points[lane]));
        }
    }
    return result;
}

Comparison compare_reference_normals(std::mt19937& generator,
                                     render::RenderConfig const& config) {
    // ULP against the zero target is meaningless for the angle metric.
    Comparison result{"reference vs scalar normal angle (degrees)", 0.1f,
                      "target", "angle", false};
    constexpr double radians_to_degrees = 180.0 / std::numbers::pi_v<double>;
    for (int batch = 0; batch < 64; ++batch) {
        PointBatch points = random_points(generator, batch);
        for (int lane = 0; lane < kLanes; ++lane) {
            if (!reference::sample_sdf(to_reference(points[lane]), config)
                     .escaped) {
                ++result.skipped;
                continue;
            }
            reference::Vec3d expected =
                reference::estimate_normal(to_reference(points[lane]), config);
            Vec3 actual = scalar::estimate_normal(points[lane], config);
            double dot = expected.x * static_cast<double>(actual.x) +
                         expected.y * static_cast<double>(actual.y) +
                         expected.z * static_cast<double>(actual.z);
            double angle =
                std::acos(std::clamp(dot, -1.0, 1.0)) * radians_to_degrees;
            compare_value(result, 0.0f, static_cast<float>(angle),
                          batch_point_location(batch, lane, points[lane]));
        }
    }
    return result;
}

Comparison compare_rays(Camera const& camera,
                        render::RenderConfig const& config) {
    Comparison result{"scalar vs SIMD traced ray channels"};
    for (int y = 0; y < config.height; y += 17) {
        LaneValues xs{}, ys{};
        for (int lane = 0; lane < kLanes; ++lane) {
            xs[lane] = static_cast<float>(lane * 17);
            ys[lane] = static_cast<float>(y);
        }
        Vec3x8 actual =
            simd8::trace_ray_packet(camera, pack(xs), pack(ys), config);
        ColorLanes channels{lanes(actual.x256), lanes(actual.y256),
                            lanes(actual.z256)};
        for (int lane = 0; lane < kLanes; ++lane) {
            int x = static_cast<int>(xs[lane]);
            render::Color expected = scalar::trace_ray(camera, x, y, config);
            for (int channel = 0; channel < render::kColorChannels;
                 ++channel) {
                render::Pixel actual_channel =
                    static_cast<render::Pixel>(channels[channel][lane]);
                compare_value(result, static_cast<float>(expected[channel]),
                              static_cast<float>(actual_channel),
                              ray_channel_location(lane, x, y, channel));
            }
        }
    }
    return result;
}

RenderPair compare_render(Camera const& camera,
                          render::RenderConfig const& config,
                          std::filesystem::path const& output_directory) {
    render::Image reference_image(config), scalar_image(config),
        simd_image(config);
    reference::render(camera, config, reference_image);
    scalar::render(camera, config, scalar_image);
    simd8::render(camera, config, simd_image);

    RenderPair pair{
        {"scalar vs SIMD render channels"},
        {"reference vs scalar render channels", 0.0f, "reference", "scalar"},
        {"reference vs SIMD render channels", 0.0f, "reference", "simd"},
    };
    render::Image scalar_simd_diff(config), reference_scalar_diff(config),
        reference_simd_diff(config);

    auto amplified_difference = [](render::Pixel left, render::Pixel right) {
        int absolute =
            std::abs(static_cast<int>(left) - static_cast<int>(right));
        return static_cast<render::Pixel>(std::min(255, absolute * 8));
    };
    for (std::size_t index = 0; index < reference_image.size(); ++index) {
        LocationContext location = render_channel_location(index);
        compare_value(pair.scalar_simd, static_cast<float>(scalar_image[index]),
                      static_cast<float>(simd_image[index]), location);
        compare_value(pair.reference_scalar,
                      static_cast<float>(reference_image[index]),
                      static_cast<float>(scalar_image[index]), location);
        compare_value(pair.reference_simd,
                      static_cast<float>(reference_image[index]),
                      static_cast<float>(simd_image[index]), location);
        scalar_simd_diff[index] =
            amplified_difference(scalar_image[index], simd_image[index]);
        reference_scalar_diff[index] =
            amplified_difference(reference_image[index], scalar_image[index]);
        reference_simd_diff[index] =
            amplified_difference(reference_image[index], simd_image[index]);
    }

    std::filesystem::create_directories(output_directory);
    ImageTask const tasks[] = {
        {"reference.png", &reference_image},
        {"scalar.png", &scalar_image},
        {"simd.png", &simd_image},
        {"scalar_simd_diff_x8.png", &scalar_simd_diff},
        {"reference_scalar_diff_x8.png", &reference_scalar_diff},
        {"reference_simd_diff_x8.png", &reference_simd_diff},
    };
    bool wrote = true;
    for (ImageTask const& task : tasks) {
        std::filesystem::path path = output_directory / task.filename;
        wrote = stbi_write_png(path.string().c_str(), task.image->width(),
                               task.image->height(), render::kColorChannels,
                               task.image->data(),
                               static_cast<int>(task.image->row_stride())) !=
                    0 &&
                wrote;
    }
    if (!wrote) {
        std::cerr << "Failed to write correctness images\n";
    }
    return pair;
}

void print(std::ostream& output, Comparison const& result) {
    double mean = result.samples == 0
                      ? 0.0
                      : result.absolute_error_sum /
                            static_cast<double>(result.samples);
    output << (result.passed() ? "PASS " : "FAIL ") << result.name << ": "
           << result.mismatches << '/' << result.samples
           << " mismatches, skipped=" << result.skipped
           << ", nonfinite=" << result.nonfinite_mismatches
           << ", max_abs=" << result.max_absolute_error
           << ", mean_abs=" << mean;
    if (result.track_ulp) {
        output << ", max_ulp=" << result.max_ulp_error;
    }
    output << '\n';
    if (!result.first_mismatch.empty()) {
        output << "  first: " << result.first_mismatch << '\n';
    }
}

std::string_view architecture() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#else
    return "unknown";
#endif
}

std::string_view build_type() {
#ifdef NDEBUG
    return "release";
#else
    return "debug";
#endif
}

} // namespace

int main(int argc, char* argv[]) {
    std::filesystem::path output_directory =
        std::filesystem::path(SIMD_RAY_MARCHER_SOURCE_DIR) / "images" /
        "correctness" / "current";
    int resolution = 128;
    if (argc > 3) {
        std::cerr << "Usage: " << argv[0]
                  << " [output-directory] [resolution]\n";
        return 2;
    }
    if (argc >= 2) {
        output_directory = argv[1];
    }
    if (argc == 3) {
        std::string_view value = argv[2];
        const std::from_chars_result parse_result = std::from_chars(
            value.data(), value.data() + value.size(), resolution);
        if (parse_result.ec != std::errc{} ||
            parse_result.ptr != value.data() + value.size() ||
            resolution <= 0 || resolution > 4096 ||
            resolution % kLanes != 0) {
            std::cerr << "Resolution must be a positive multiple of " << kLanes
                      << " no larger than 4096\n";
            return 2;
        }
    }

    Camera camera = make_default_camera();
    render::RenderConfig ray_config = render::kDefaultRenderConfig;
    ray_config.width = resolution;
    ray_config.height = resolution;
    std::mt19937 generator(kDefaultSeed);
    render::RenderConfig four_iteration_config = ray_config;
    four_iteration_config.mandelbulb_iterations = 4;
    std::vector<Comparison> results;
    results.push_back(compare_camera(camera, ray_config));
    results.push_back(compare_reference_sdf(
        generator, four_iteration_config,
        "reference vs scalar SDF (matched 4 iterations)"));
    generator.seed(kDefaultSeed);
    results.push_back(compare_reference_sdf(
        generator, ray_config,
        "reference vs scalar SDF (configured iterations)"));
    generator.seed(kDefaultSeed);
    results.push_back(compare_sdf(generator));
    generator.seed(kDefaultSeed);
    results.push_back(compare_reference_normals(generator, ray_config));
    generator.seed(kDefaultSeed);
    results.push_back(compare_normals(generator, ray_config));
    results.push_back(compare_rays(camera, ray_config));
    RenderPair render_pair =
        compare_render(camera, ray_config, output_directory);
    results.push_back(std::move(render_pair.scalar_simd));
    results.push_back(std::move(render_pair.reference_scalar));
    results.push_back(std::move(render_pair.reference_simd));

    std::ostringstream report;
    report << "architecture=" << architecture() << '\n'
           << "build=" << build_type() << '\n'
           << "simd_api=avx2\n"
           << "compatibility_layer=simde\n"
           << "reference=float64-trigonometric\n"
           << "reference_iterations=" << ray_config.mandelbulb_iterations
           << '\n'
           << "reference_escape_radius=" << ray_config.mandelbulb_escape_radius
           << '\n'
           << "camera_position=" << camera.position.x << ','
           << camera.position.y << ',' << camera.position.z << '\n'
           << "camera_forward=" << camera.forward.x << ',' << camera.forward.y
           << ',' << camera.forward.z << '\n'
           << "fov_degrees=" << ray_config.fov_degrees << '\n'
           << "resolution=" << resolution << 'x' << resolution << '\n'
           << "sample_seed=0x5EED\n";

    bool passed = true;
    for (const Comparison& result : results) {
        print(report, result);
        passed = passed && result.passed();
    }
    report << "Overall: " << (passed ? "PASS" : "FAIL") << '\n';
    std::cout << report.str() << "Artifacts: " << output_directory << '\n';

    std::ofstream report_file(output_directory / "report.txt");
    if (!report_file) {
        std::cerr << "Failed to write correctness report\n";
        return 2;
    }
    report_file << report.str();
    return passed ? 0 : 1;
}
