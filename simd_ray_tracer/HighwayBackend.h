#pragma once

#include <cstddef>

namespace highway_backend {

struct ProbeResult {
  std::size_t float_lanes;
  const char* target_name;
  bool arithmetic_ok;
  bool masks_ok;
  bool logarithm_ok;
};

ProbeResult probe();

}  // namespace highway_backend
