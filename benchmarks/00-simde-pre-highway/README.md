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

## Lane occupancy

An instrumented, untimed render records useful lane slots inside the two
divergent SIMD loops and at normal-estimation call sites. The observed
eight-wide SIMDe execution is:

| Stage | Useful lane iterations | Issued lane slots | Occupancy |
|---|---:|---:|---:|
| Ray marching | 109,780,664 | 118,613,048 | 92.5536% |
| Mandelbulb iterations | 550,249,328 | 571,993,728 | 96.1985% |
| Normal batches | 2,531,953 | 8,541,520 | 29.6429% |

Adjacent pixels remain coherent through both the ray marcher and Mandelbulb
escape loop. Normal estimation is the clear utilization outlier: the renderer
issued 1,067,690 packet-wide normal calculations for 319,066 packets that
eventually contained a hit, and those calculations account for 30.1710% of all
packet SDF calls. Deferring normal estimation until each ray packet finishes
would reduce normal batches by 70.1162% and raise their projected outer-lane
occupancy to 99.1939% for this scene, before changing the six-sample normal
formula.

These are logical eight-lane AVX packet measurements. On Apple Silicon, SIMDe
implements each logical packet with two native four-lane NEON vectors. The
occupancy capture is intentionally separate from performance timing because
mask extraction, population counts, and counter updates add work to the hot
loops.

### Counterfactual native-four lane-slot model

The same masks are also split into low and high four-lane halves. Each half is
counted only while it has active lanes, and normal SDF work is counted only for
halves containing hits. This models independently terminating native-four
packets without implementing or executing a four-wide renderer:

| Stage | Modeled useful lane iterations | Modeled issued lane slots | Modeled lane utilization |
|---|---:|---:|---:|
| Ray marching | 109,780,664 | 113,785,444 | 96.4804% |
| Mandelbulb iterations | 461,411,658 | 469,809,552 | 98.2125% |
| Immediate normal batches | 2,531,953 | 5,262,476 | 48.1133% |
| Deferred normal batches | 2,531,953 | 2,541,632 | 99.6192% |

The model places immediate-normal utilization between the previously derived
29.6429% and 59.2858% bounds. Within the model, deferring normals reduces the
number of four-lane normal batches from 1,315,619 to 635,408, or 51.7027%.

This is a lane-slot counterfactual, not measured hardware occupancy and not a
performance prediction. It does not execute the additional loop tests and
branches required by independent packets, nor does it model compiler
scheduling, front-end effects, caches, or runtime. A native Highway/NEON render
is required to measure those effects. The model exists only to separate the
packet-width hypothesis from the later Highway and deferred-normal changes.

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
`environment.txt` for the build and machine configuration. Per-iteration
occupancy is in `lane-occupancy.csv`; aggregate counts and active-lane
histograms are in `lane-occupancy-summary.txt`.
