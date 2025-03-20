# CMakeList.txt : CMake project for simd_ray_tracer, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable (simd_ray_march "simd_ray_march.cpp"   "Camera.h" "Vec3.h" "Vec3.cpp" "Camera.cpp" "Vec3x8.h" "Vec3x8.cpp")
add_executable (scalar_ray_march "scalar_ray_march.cpp"   "Camera.h" "Vec3.h" "Vec3.cpp" "Camera.cpp" "Vec3x8.h" "Vec3x8.cpp")
if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET simd_ray_march PROPERTY CXX_STANDARD 20)
  set_property(TARGET scalar_ray_march PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
