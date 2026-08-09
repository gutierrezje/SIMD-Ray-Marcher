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

# Add source to this project's executable.
add_executable (simd_ray_march "simd_ray_march.cpp"   "Camera.h" "SimdBackend.h" "SimdBackend.cpp" "RenderConfig.h" "RenderTypes.h" "Vec3.h" "Vec3.cpp" "Camera.cpp" "Vec3x8.h" "Vec3x8.cpp")
add_executable (scalar_ray_march "scalar_ray_march.cpp"   "Camera.h" "ScalarBackend.h" "ScalarBackend.cpp" "RenderConfig.h" "RenderTypes.h" "Vec3.h" "Vec3.cpp" "Camera.cpp")

set(SIMD_RAY_MARCHER_X86 FALSE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i[3-6]86)$")
    set(SIMD_RAY_MARCHER_X86 TRUE)
elseif(MSVC AND CMAKE_GENERATOR_PLATFORM MATCHES "^(x64|Win32)$")
    set(SIMD_RAY_MARCHER_X86 TRUE)
endif()

foreach(target simd_ray_march scalar_ray_march)
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
endforeach()

# TODO: Add tests and install targets if needed.
