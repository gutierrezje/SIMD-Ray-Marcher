# Rendering baselines

This directory preserves rendering evidence at architecture milestones. A
baseline is a matched set produced by `correctness_harness`:

- `reference.png`: independent float64/trigonometric reference image, when present
- `scalar.png`: scalar float32 image
- `simd.png`: candidate backend image
- `*_diff_x8.png`: named per-channel absolute differences multiplied by eight
- `report.txt`: numeric stage comparisons and build metadata

Use an immutable, descriptive directory for a milestone, for example:

```text
images/correctness/00-article-framing-before-fixes/
images/correctness/01-lane-local-escape-fix/
images/correctness/02-render-config-authoritative/
images/correctness/03-highway-neon/
images/correctness/04-highway-multithreaded/
```

Running the harness without an output argument writes to
`images/correctness/current/`, which is a disposable local comparison. Pass a
milestone directory when the result should be retained. Never replace an older
milestone: the sequence is the record of both rendering fixes and architecture
changes.

The optional second argument selects a square resolution. Correctness milestones
use the default 128 pixels; article assets use 1200 pixels:

```text
correctness_harness images/article/normal-color-framing-1200 1200
```

FLIP is an additional end-to-end perceptual metric over a reference/candidate
image pair. Prefer `reference.png` as the reference when present; older
milestones use `scalar.png`. FLIP complements, but does not replace, the
stage-level numeric report.
Performance results belong in a separate benchmark record so correctness runs
do not mix timing noise into rendering evidence.

The Mandelbulb distance estimate is an exterior estimate, not an exact signed
distance. Reference SDF and normal statistics therefore record bounded samples
as skipped instead of treating their interior values as an exact oracle.
