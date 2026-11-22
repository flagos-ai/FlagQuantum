# Controlled-gate commutation results

Owner: Compiler. Classification: `single_device_fast_path`.

The [implementation and protocol](COMPILER_CONTROLLED_COMMUTATION.md) were
published at `1b859d7cf1de0929b9e5ce53c515c663058d004e` before observation.
The control is production main `46ba1523ed0c47fcf304d64d33b34d3f6c4b9629`;
the only production difference is the structural commutation extension. The
experiment patches old/new checker functions for equal lifetimes. It does not
enable the separate commuting-frontier selector or native-cost experiments.

## Complete application observations

Both declared draws completed: 240 observations using the inspected nonce and
240 using one fresh nonce generated after publication and frozen before either
draw ran. Each draw covers all 60 inputs, both complete workflows and both
variants, with compiler seed 11. All 480 observations pass unchanged SDKBench
`9aa9260920984e2ca05bcb293abb885fccd9c144` independent acceptance.
There are 160 complete-operator checks at width six and 320 three-probe checks
at widths eight/twelve; the latter cannot establish universal equivalence.

| Draw | Workflow | Improved CZ/depth | Equal CZ/depth | Regressed CZ/depth | Tradeoff |
| --- | --- | ---: | ---: | ---: | ---: |
| Inspected | Ordinary restored | 0 | 60 | 0 | 0 |
| Inspected | Target artifact | 0 | 60 | 0 | 0 |
| Fresh | Ordinary restored | 0 | 60 | 0 | 0 |
| Fresh | Target artifact | 0 | 60 | 0 | 0 |

Do not interpret this as closing the existing entangler/depth gap. These
application circuits do not demonstrate the three-to-one CX cancellation of the
focused uncomputation example. Each draw does change the normalized output for
the 12 ordinary repetition cases: each has one fewer RZ instruction, with
unchanged CZ count and dependency depth. The remaining 108 paired outputs per
draw are byte-identical after normalization. Total instruction count is an
additional output diagnostic inspected after observation, not a replacement for
the declared CZ/depth objectives. The target-artifact outputs do not improve.

The source proof and real uncomputation workflow support broader structural
coverage. These application results establish bounded before/after acceptance
and disclose limited benefit; they do not establish a global SDK ranking,
significance, broad performance improvement or independently administered
qualification. All source pins, manifests, payloads and original compile-wall
acceptance values must be retained.

## Resource follow-up declaration

Declare this follow-up after the complete quality observations and before its
resource measurements. Preserve the frozen implementation and use inspected
QFT6/grid, repetition6/ring and QFT12/grid, both complete workflows and both
checker variants, compiler seed 11. Keep the existing adapter configurations,
120-second deadlines, operation budgets and independent acceptance.

Each of six case/workflow pairs uses two warmups and nine alternating retained
samples per variant, fresh adapters, equal checker-patch lifetimes and one
numerical thread. Measure adapter execution outside independent acceptance.
Measure one additional traced Python-allocation peak per variant after garbage
collection, keeping adapter creation outside tracing. Verify every retained
final output and audit against its frozen quality result and replay independent
acceptance outside timing. Keep all raw samples and failures, with no favorable
rerun or case removal. Freeze the driver and this declaration by source hashes.

Use a 240-second process-group supervisor and no simultaneous local test suite.
No resource admission threshold is introduced after measurement. Shared-host
medians and Python allocation peaks cannot certify significance, a latency
ranking, native allocations, peak RSS or formal Linux cgroup memory. Formal
host qualification remains user-deferred; no machine or service may be restarted.

## Archives and independent replay

The raw [inspected index](evidence/controlled-commutation-inspected-screen.json)
and [fresh index](evidence/controlled-commutation-fresh-screen.json) each retain
three hashed payload chunks. They reconstruct the original 3,828,280-byte and
3,886,439-byte reports exactly. Each draw retains 116 distinct normalized
outputs and every one of its 240 observation rows. The
[analysis](evidence/controlled-commutation-analysis.json) lists all output changes,
including the additional instruction-count diagnostic.

A fresh process [independently replayed](evidence/controlled-commutation-replay.json)
all 480 observations without importing FlagQuantum, Qiskit or Cirq. Maximum
observed residuals are 6.072657351995283e-14 for the inspected draw and
2.758150189067026e-14 for the fresh draw against the unchanged 1e-8 threshold.
Fixture hashes, complete case/variant coverage, output hashes, layouts and exact
returned metrics are checked. Three-probe rows retain their limited coverage.

## First resource observations and limitations

The [complete resource record](evidence/controlled-commutation-resources.json)
retains every timed sample, allocation observation, driver and protocol snapshot.
The first run completed within the supervisor budget and was not rerun.

| Input | Workflow | Baseline median ms | Candidate median ms | Candidate/baseline | Baseline Python peak bytes | Candidate Python peak bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 06/qft/grid | ordinary-restored | 7.778 | 7.713 | 0.9917 | 472558 | 471969 |
| 06/qft/grid | target-artifact | 93.818 | 91.605 | 0.9764 | 2198408 | 2198392 |
| 06/repetition/ring | ordinary-restored | 6.712 | 8.055 | 1.2000 | 552720 | 562802 |
| 06/repetition/ring | target-artifact | 121.201 | 121.713 | 1.0042 | 2566909 | 2566859 |
| 12/qft/grid | ordinary-restored | 30.475 | 30.396 | 0.9974 | 2113695 | 2115138 |
| 12/qft/grid | target-artifact | 565.080 | 562.130 | 0.9948 | 10118879 | 10118879 |

The ordinary repetition6/ring median grows by about 20.0%, despite removing only
one RZ. Do not hide this adverse observation or infer significance from nine
shared-host samples. Other ratios close to one do not establish parity or an
improvement. The raw field named `compiler_stage_python_peak_bytes` covers the
complete adapter execution window, not an isolated compiler transform. It
excludes native allocations and is not peak RSS.

The resource driver validates the last timed output and audit per variant
against quality hashes and independent acceptance. It does not retain or validate
each intermediate timed payload or the traced payload. Thus it does not fully
meet the declaration's per-sample validation intent; these observations do not
certify complete resource-protocol conformance or resource admission. Retain
this limitation with the original driver and data rather than rewrite the
protocol or rerun for a favorable median.

An [untimed diagnostic](evidence/controlled-commutation-work-census.json) examines
the adverse inspected ordinary repetition6/ring case without replacing its
measurements. Both outputs and audits still match the frozen quality hashes.
The baseline makes four commutation-analysis calls and 396 predicate calls; the
candidate makes six analysis calls and 610 predicate calls. Its initial widest
block grows from three to ten, and initial logical pair queries grow from 82 to
121. Broader blocks and additional fixed-point work are concrete costs of the
newly enabled simplification. This census does not isolate how much of the
observed wall-time difference comes from each cost or from shared-host noise.

## Decision

Keep this implementation under draft review. It proves useful structural
coverage and the default uncomputation example, while these application draws
show no entangler/depth improvement and the resource observation includes a
material regression. Do not call this a solution to the remaining application
routing gap or enable the separate commuting-frontier policy on its strength.
The next implementation must address analysis/fixed-point work and evaluate
final entangler count, dependency depth and resources together under a new
frozen declaration. Formal Linux qualification remains deferred.
