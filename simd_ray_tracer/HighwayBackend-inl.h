#ifndef SIMD_RAY_TRACER_HIGHWAY_BACKEND_INL_H_
#define SIMD_RAY_TRACER_HIGHWAY_BACKEND_INL_H_
#endif

#if defined(SIMD_RAY_TRACER_HIGHWAY_BACKEND_INL_H_TARGET) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef SIMD_RAY_TRACER_HIGHWAY_BACKEND_INL_H_TARGET
#undef SIMD_RAY_TRACER_HIGHWAY_BACKEND_INL_H_TARGET
#else
#define SIMD_RAY_TRACER_HIGHWAY_BACKEND_INL_H_TARGET
#endif

#include <cmath>
#include <cstddef>

#include "HighwayBackend.h"
#include "hwy/aligned_allocator.h"
#include "hwy/contrib/math/math-inl.h"
#include "hwy/highway.h"
#include "hwy/targets.h"

HWY_BEFORE_NAMESPACE();
namespace highway_backend {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

void RunProbe(ProbeResult* result) {
  using D = hn::ScalableTag<float>;
  using V = hn::Vec<D>;

  const D d;
  const std::size_t lanes = hn::Lanes(d);
  hwy::AlignedFreeUniquePtr<float[]> input = hwy::AllocateAligned<float>(lanes);
  hwy::AlignedFreeUniquePtr<float[]> arithmetic =
      hwy::AllocateAligned<float>(lanes);
  hwy::AlignedFreeUniquePtr<float[]> logarithms =
      hwy::AllocateAligned<float>(lanes);

  bool arithmetic_ok = input != nullptr && arithmetic != nullptr;
  bool logarithm_ok = input != nullptr && logarithms != nullptr;
  if (arithmetic_ok && logarithm_ok) {
    for (std::size_t lane = 0; lane < lanes; lane += 1) {
      input[lane] = static_cast<float>(lane + 1);
    }

    const V values = hn::Load(d, input.get());
    const V scaled =
        hn::Add(hn::Mul(values, hn::Set(d, 2.0f)), hn::Set(d, 1.0f));
    hn::Store(hn::Sqrt(scaled), d, arithmetic.get());
    hn::Store(hn::Log(d, values), d, logarithms.get());

    for (std::size_t lane = 0; lane < lanes; lane += 1) {
      const float expected_arithmetic = std::sqrt(2.0f * input[lane] + 1.0f);
      const float expected_logarithm = std::log(input[lane]);
      arithmetic_ok = arithmetic_ok &&
                      std::abs(arithmetic[lane] - expected_arithmetic) <= 1e-6f;
      logarithm_ok = logarithm_ok &&
                     std::abs(logarithms[lane] - expected_logarithm) <= 1e-6f;
    }
  }

  const V zero = hn::Zero(d);
  const bool masks_ok = !hn::AllFalse(d, hn::Eq(zero, zero)) &&
                        hn::AllFalse(d, hn::Lt(zero, zero));

  result->float_lanes = lanes;
  result->target_name = hwy::TargetName(HWY_TARGET);
  result->arithmetic_ok = arithmetic_ok;
  result->masks_ok = masks_ok;
  result->logarithm_ok = logarithm_ok;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace highway_backend
HWY_AFTER_NAMESPACE();

#endif  // per-target include guard
