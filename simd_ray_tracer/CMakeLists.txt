# CMakeList.txt : CMake project for simd_ray_tracer, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable (simd_ray_tracer "scalar_ray_march.cpp" "scalar_ray_march.h" "stb_image.h" "Camera.h" "Vec3.h" "Vec3.cpp" "Camera.cpp" "simd_ray_march.h" "simd_ray_march.cpp" "Vec3x8.h" "Vec3x8.cpp")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET simd_ray_tracer PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
