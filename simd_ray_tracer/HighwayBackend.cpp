#include "HighwayBackend.h"

#include "HighwayBackend-inl.h"
#include "hwy/highway.h"

namespace highway_backend {

ProbeResult probe() {
  ProbeResult result{};
  HWY_STATIC_DISPATCH(RunProbe)(&result);
  return result;
}

}  // namespace highway_backend
