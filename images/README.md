# Rendering baselines

This directory preserves rendering evidence at architecture milestones. A
baseline is a matched set produced by `correctness_harness`:

- `scalar.png`: scalar reference image
- `simd.png`: candidate backend image
- `absolute_diff_x8.png`: per-channel absolute difference multiplied by eight
- `report.txt`: numeric stage comparisons and build metadata

Use an immutable, descriptive directory for a milestone, for example:

```text
images/correctness/00-simde-neon-before-fixes/
images/correctness/01-simde-neon-correct/
images/correctness/02-highway-neon/
images/correctness/03-highway-multithreaded/
```

Running the harness without an output argument writes to
`images/correctness/current/`, which is a disposable local comparison. Pass a
milestone directory when the result should be retained. Never replace an older
milestone: the sequence is the record of both rendering fixes and architecture
changes.

FLIP is an additional end-to-end perceptual metric over `scalar.png` and
`simd.png`. It complements, but does not replace, the stage-level numeric report.
Performance results belong in a separate benchmark record so correctness runs
do not mix timing noise into rendering evidence.
