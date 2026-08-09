# CMakeList.txt : CMake project for simd_ray_tracer, include source and define
# project specific logic here.
#

# SIMDe keeps the existing AVX-shaped implementation buildable on ARM while
# the renderer is modernized. The full commit pin is the v0.8.2 release.
include(FetchContent)
FetchContent_Declare(
  simde
  GIT_REPOSITORY https://github.com/simd-everywhere/simde.git
  GIT_TAG 71fd833d9666141edcd1d3c109a80e228303d8d7 # v0.8.2
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(simde)

set(RAY_MARCHER_CORE_SOURCES
    "Camera.h"
    "Camera.cpp"
    "RenderTypes.h"
    "ScalarBackend.h"
    "ScalarBackend.cpp"
    "SimdBackend.h"
    "SimdBackend.cpp"
    "Vec3.h"
    "Vec3.cpp"
    "Vec3x8.h"
    "Vec3x8.cpp"
)

add_executable(ray_march
    "ray_march.cpp"
    ${RAY_MARCHER_CORE_SOURCES}
)

add_executable(correctness_harness
    "correctness_harness.cpp"
    ${RAY_MARCHER_CORE_SOURCES}
)

set(SIMD_RAY_MARCHER_X86 FALSE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86)$")
    set(SIMD_RAY_MARCHER_X86 TRUE)
elseif(MSVC AND CMAKE_GENERATOR_PLATFORM MATCHES "^(x64|Win32)$")
    set(SIMD_RAY_MARCHER_X86 TRUE)
endif()

function(configure_ray_marcher_target target)
    set_target_properties(${target} PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )
    target_include_directories(${target} SYSTEM PRIVATE "${simde_SOURCE_DIR}")

    if(MSVC)
        if(SIMD_RAY_MARCHER_X86)
            target_compile_options(${target} PRIVATE /arch:AVX2)
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
           CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
           CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        target_compile_options(${target} PRIVATE
            -O3
            -Wdouble-promotion
            -Werror=double-promotion
        )
        if(SIMD_RAY_MARCHER_X86)
            target_compile_options(${target} PRIVATE -mavx2)
        endif()
    endif()
endfunction()

configure_ray_marcher_target(ray_march)
configure_ray_marcher_target(correctness_harness)
target_compile_definitions(correctness_harness PRIVATE
    SIMD_RAY_MARCHER_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

# TODO: Add tests and install targets if needed.
