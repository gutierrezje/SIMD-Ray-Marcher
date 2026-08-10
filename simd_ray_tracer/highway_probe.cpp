#include <iostream>

#include "HighwayBackend.h"

int main() {
  const highway_backend::ProbeResult result = highway_backend::probe();

  std::cout << "Highway target: " << result.target_name << '\n'
            << "float lanes: " << result.float_lanes << '\n'
            << "arithmetic: " << (result.arithmetic_ok ? "ok" : "failed")
            << '\n'
            << "masks: " << (result.masks_ok ? "ok" : "failed") << '\n'
            << "logarithm: " << (result.logarithm_ok ? "ok" : "failed") << '\n';

  const bool succeeded =
      result.arithmetic_ok && result.masks_ok && result.logarithm_ok;
  return succeeded ? 0 : 1;
}
