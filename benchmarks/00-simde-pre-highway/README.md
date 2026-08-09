# SIMDe pre-Highway baseline

This is the single-threaded performance baseline immediately before replacing
the AVX2-through-SIMDe backend with Google Highway.

## Result

| Backend | Median | MAD | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Scalar | 6.212575 s | 0.010940 s | 6.181720 s | 6.230390 s |
| SIMD via SIMDe | 2.091810 s | 0.018645 s | 2.057810 s | 2.115300 s |

The SIMD backend is **2.9700x faster** by the ratio of medians.

The authoritative run used AC power. A preceding battery-powered run measured
6.369905 seconds scalar and 2.137495 seconds SIMD. Plugging in reduced median
render time by 2.47% and 2.14%, respectively, while leaving the relative SIMD
speedup effectively unchanged (2.98x on battery versus 2.97x on AC power).

## Protocol

- Render the default 2056x2056 article scene.
- Use the application-reported `std::chrono::steady_clock` duration, which
  measures rendering and excludes PNG encoding.
- Run one unmeasured warmup for each backend.
- Run 10 measured trials per backend.
- Interleave the backends and alternate which backend runs first.
- Verify AC power before and after the run and check for power or thermal
  warnings.
- Verify that every output hash is stable within its backend.

The scalar and SIMD hashes differ because the known floating-point rendering
differences are preserved. Their correctness envelope is recorded separately
in `images/correctness/02-render-config-authoritative`.

See `raw-runs.csv` for the authoritative AC-powered measurements,
`raw-runs-battery.csv` for the initial battery-powered comparison, and
`environment.txt` for the build and machine configuration.
