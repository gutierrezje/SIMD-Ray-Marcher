#pragma once

// Transitional compatibility layer: the renderer still uses AVX-shaped
// operations, while SIMDe maps them to the available instructions on ARM.
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || \
    defined(_M_IX86)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#ifndef SIMDE_ENABLE_NATIVE_ALIASES
#define SIMDE_ENABLE_NATIVE_ALIASES
#endif
#include <simde/x86/avx2.h>
#else
#error "SIMD-Ray-Marcher currently supports x86 and ARM64 targets only."
#endif
