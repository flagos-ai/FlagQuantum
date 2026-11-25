# PF-00: Python synthesis time and process-memory baseline

PF-00 implements revision 9 sections 13.3/13.4/13.6. Its accepted production
baseline is AO-02, #1013, merged as
`103250df54a0cf7204a8057baae5316ae0a4144f` after 57 successful checks and three
configured GPU-pool skips. The actual measurement commit and script hashes are
retained in each receipt. This benchmark-only change does not optimize production
code, alter API/contracts, add dependencies or change candidate selection.

## User workflow

The ordinary user still writes a single composable program:

```python
import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Const, Output, QNum, allocate, hadamard_transform


@synth.qfunc
def energy(x: Const[QNum], y: Const[QNum], result: Output[QNum]) -> None:
    result |= x*x+y


@synth.qfunc
def main(x: Output[QNum], y: Output[QNum], result: Output[QNum]) -> None:
    allocate(2, x)
    allocate(2, y)
    hadamard_transform(x)
    hadamard_transform(y)
    energy(x, y, result)


program = synth.synthesize(main, memory_limit_bytes=64 * 1024**2)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "y", "result"))))
print(program.report["resources"])
```

There are sixteen outcomes `(x,y,x*x+y)` of probability 1/16. The actual
uniform program remains 16 total qubits, eight workspace qubits, 81 logical
operations and depth 59. The matching Qmod and independent nine-qubit Circuit
are retained in `examples.synthesis.tutorials.arithmetic.arithmetic_implementations.main` and the AO-00
benchmark reference; no new user performance configuration is needed.

PF-00 establishes how much Python work that request incurs, where diagnostic
work is attributed, and the worker's RSS. It does not imply a speed improvement,
Rust necessity, the AO-03 nine-qubit goal, GPU/QPU execution or release readiness.

## Measurement boundary

`benchmarks/synthesis/performance.py` launches isolated process groups. The
benchmark never patches production methods or changes checks. There is no
production observer, public profiling API, new IR or persistent-worker cache.

- Primary E0 is `perf_counter_ns` around the actual `synth.synthesize` call,
  including checked Program delivery. It excludes execution and readout.
  Each formal timing sample uses a fresh process, two identical warmups and one
  measurement; the default retains five such samples. GC and contracts remain
  enabled; PyTorch intra/inter-op and BLAS/OpenMP threads are fixed to one.
- Cold startup-to-import and dependency-import time use five additional new
  processes. This is a cold Python process, not a flushed OS page cache.
  Definition/source preparation (including native decoration) is
  recorded separately, before E0, rather than silently charged to a pass.
  First synthesis-call latency is retained from the first warmup in each worker.
- E1 includes separately timed `fq.run` and named readout, plus E0, for bounded
  small-state tasks. Full complex amplitudes, including every zero-workspace
  component, are checked against independent mathematics. Larger tasks record
  bounded independent X/CX/CCX interpretation with nonzero XOR targets and an
  explicit state-execution budget exclusion. No exponential simulation is
  performed during compilation or for large-state claims.
- A separate diagnostic process runs two warmups and one profiled call.
  `performance_probe.py` records nested C1 admission, C2 analysis, C3 arithmetic,
  C4 target, C5 selection/cost, and C6 delivery spans. Inclusive and exclusive
  nanoseconds, parent IDs, calls, implementation/schedule context, direct input
  scale and string/bytes lengths (including actual JSON when present)
  are retained. Exclusive totals plus unattributed time equal diagnostic E0;
  inclusive nested times are never added to estimate total synthesis time.
  Generator resumptions remain charged to their enclosing spans. Direct argument
  lengths are not unique processed-node counts, allocations or sizes of
  serialized nested objects; unobserved byte volume remains unknown.
- Diagnostic attribution includes probe overhead; it is not a measurement of
  natural unprofiled per-pass time. Compare its E0 with the primary distribution
  to expose perturbation. Counted Instruction reconstruction, serialization,
  content hashing and validation call paths identify PF-01 hypotheses, not
  automatically removable redundant work.

## OS memory

Primary memory trials use a separate fresh worker, the same two warmups and an
unprofiled synthesis call. The controller observes root-worker and simultaneous
process-tree RSS with `ps`, normally every 15 ms. Raw KiB, byte conversions,
actual intervals, process counts and sampler wall cost are preserved. Shared
pages may be counted twice by tree sums. The orchestration launcher is excluded
from the compiler-worker scope. This runner supports Linux and macOS.

Record window-start/current RSS, sampled compilation-window peak, window-end
RSS, and lifetime `ru_maxrss` including import, source preparation and warmups.
The lifetime mark and compile-window end are recorded before diagnostic report
construction, verification or simulation. Report construction has its own phase
and nanoseconds, outside the compilation RSS window. It cannot be reset or
subtracted to create a phase peak. macOS high-water units are bytes; Linux units
are KiB. Missing observations are failures/unknown, never zero.

Diagnostic RSS points additionally carry the last reported C1–C6 context. These
are coarse, sampled observations under the separate intrusive probe; very short
stages and transitions can be missed. They are not primary per-stage memory or
exact allocation peaks. Raw spans provide attribution; primary unprofiled RSS
provides the main compiler-memory baseline. Statevector planning bytes,
workspace width and compiler RSS remain distinct metrics.

The default full matrix takes one independent primary memory observation per
configuration, explicitly `memory_trials=1`, not five memory repeats. Use
`--memory-trials 5` for a repeat cohort or memory-benefit comparison. A single
RSS observation is not a robust memory-distribution or percentile claim.

## Workloads, refusals and reproducibility

The complete ninety-case AO-00 domain/configuration remains intact. Supplemental
cases cover native qfunc/Qmod/scalar equivalents, two- and eight-bit repeated
expressions, square/constant/terminal-output rules at multiple widths, helper,
inline and exchanged spellings, logical/named-basis/line targets, and real
one-qubit-budget refusals. Generated native source text and SHA-256 are retained;
source capture still goes through the normal Compiler. A generated native H
prefix must be exactly the declared input-only preparation before the existing
independent gate interpreter checks the arithmetic XOR block.

Record the clean measurement commit, accepted AO-02 production tree, script and
oracle hashes, unchanged AO JSON hashes, resolved package versions, Python/OS/
CPU availability, quota visibility, thread/cache/GC policy, seed and deadlines.
The default production pin is the accepted AO-02 merge. The driver refuses a
different production tree unless the caller supplies its own full
`--production-revision` pin (for explicitly labeled development/future cohorts).
Tests pin their own checkout, including a CI merge checkout; that does not
relabel another production version as the accepted P0. Timings are
host-specific observations, with raw samples, median and min/max; five samples
do not establish a reliable p95 or a language speedup.

Each synthesis request retains its 30-second wall budget. Source setup, startup,
verification and the full worker have separate bounded allowances; warmups are
not squeezed into a single call's budget. The controller kills/reaps the complete
worker group on failure and verifies no group remains. Receipts distinguish
refusal, timeout, unsupported, validation failure, measurement failure and
cancellation. An incomplete/failed trial stays in the denominator and is excluded
from successful timing statistics; missing samples are never silently discarded.

Run on an otherwise idle host from the recorded clean measurement commit:

```bash
python -m benchmarks.synthesis.performance --output /tmp/pf00.json
python -m benchmarks.synthesis.performance_results /tmp/pf00.json \
  --summary /tmp/pf00-summary.json
gzip -n -c /tmp/pf00.json > /tmp/pf00.json.gz
```

A bounded development run can select an exact inventory ID with `--case`, or a
case name to select its admitted configurations. It records unselected work and
is never relabeled a complete corpus. `--samples 1` is for development; the reader
requires `--development` and will not accept it as a formal five-sample baseline.
The full raw receipt is authoritative; the summary is a review aid.

## Acceptance and limitations

Focused tests exercise real independent workflows, separate fresh workers,
RSS units/scope, span double counting, parent IDs, source/resource hash tampering,
missing samples, GC/profiling changes, statistics, timeout cleanup and actual
budget refusals. Existing AO contracts remain unchanged. Fixed timing thresholds
are deliberately absent from functional CI.

The final PR description and delivery ledger retain the measured commit,
experiment results and test commands. Neither tool tests nor a development pilot
constitutes formal baseline evidence. PF-01 removes measured redundant work;
PF-02 changes no trust boundary without proof; PF-03 establishes P1. Rust remains
a separately justified future stage against P1, never inferred from this P0 alone.

## Preserved measurement and verifier correction

The first complete matrix is pinned to
`601f179f5d973bd00d2c560bb20383390cf772ab`; its raw receipt is preserved,
including all 125 attempted configurations. Three non-restoring line targets
exposed a benchmark-oracle coordinate error: `Program.registers` is logical,
while the returned statevector is physical. Named readout and the independent
logical XOR interpreter were already correct. The corrected complete-state
oracle applies the selected target layout to expected support indices.

Only those three configurations are remeasured in a separate committed cohort.
The manifest explicitly chooses the corrected records for them and the original
records elsewhere; it does not relabel source revisions or erase failed trials.
The synthesis timer/driver and production tree are unchanged. Probe exception
cleanup additionally preserves the original exception if a callback is interrupted;
it publishes no incomplete spans. Original diagnostic failures remain failures,
not retroactively successful or reclassified observations.

The reader validates clean source/oracle hashes against the recorded Git commit,
and verifies that its production tree matches the explicit production pin. Thus
immutable receipts remain verifiable after this oracle-only correction. A missing
commit requires fetching that exact revision; there is no silent current-source
substitution. Dirty development receipts remain ineligible for formal evidence.

## Frozen P0 observations

The source commits are `601f179f5d973bd00d2c560bb20383390cf772ab`
(initial 125 configurations) and
`44afa3c1971924ff74230bf76404ae29f6f99aba`
(the three corrected line configurations). Both have the accepted AO-02
production tree and a byte-identical timing driver. Their distinct probe/source
hashes are retained; the second probe changes only interrupted-exit handling.

The [manifest](../../benchmarks/synthesis/baselines/pf00-manifest.json) maps each
of the 125 unique configurations to its cohort. There are 128 configuration
attempts overall. All original records, including 21 failed line-oracle trials,
remain in the [initial raw receipt](../../benchmarks/synthesis/baselines/pf00-initial.json.gz.parts.json).
The [correction receipt](../../benchmarks/synthesis/baselines/pf00-line_correction.json.gz)
contains 21 independently verified replacement trials. The
[overview](../../benchmarks/synthesis/baselines/pf00-summary.json) is derived from
that explicit selection, with no successful-only hiding of the remaining gaps.

| Selected cohort | Verified | Expected refusal | Timeout | Measurement failure |
| --- | ---: | ---: | ---: | ---: |
| Primary E0 trials | 605 | 15 | 5 | 0 |
| Independent primary RSS trials | 121 | 3 | 1 | 0 |
| Separate diagnostic trials | 115 | 3 | 2 | 5 |

124 configurations have all five accepted primary measurements: 121 compiled
configurations and three real budget-refusal configurations. The original
`mixed_products-16x16` has no successful formal sample; all five timing attempts
and its memory/diagnostic attempts time out. Six additional large configurations
lack complete diagnostic spans (five recorded incomplete-stack observer failures,
one timeout). Those original failure labels remain unchanged. An updated exit
handler preserves original exceptions, but does not manufacture missing data.
This is a partial diagnostic baseline, not a claim that all 125 have full C1–C6
attribution. RSS uses one observation per configuration, not a five-trial memory
distribution.

Host: Apple M5, Python 3.12.14, PyTorch 2.13.0, one compiler worker and one
intra/inter-op thread. The host was not exclusive; frequency/thermal state is
unmeasured. Initial cold-process startup-to-import median is 503.2 ms (five
samples, 482.6–791.6 ms). It is a cold process, not a flushed OS page cache.

| Two-bit uniform energy, logical target | E0 median | Window sampled peak RSS | First run median | Named read median |
| --- | ---: | ---: | ---: | ---: |
| Scalar Python | 33.70 ms | 206.58 MiB | 49.13 ms | 22.32 ms |
| Unified Python qfunc | 32.47 ms | 206.88 MiB | 46.11 ms | 21.40 ms |
| Qmod | 32.52 ms | 207.16 MiB | 46.21 ms | 21.82 ms |

Each retains the same 16/8/81/59 total/workspace/operations/depth and 1 MiB
planned statevector. Planned state bytes are not compiler RSS. The close frontend
numbers are observations, not significant frontend speed differences.

The separate qfunc diagnostic observes 43 `plan_to_dict` calls, 50 IR
`from_dict` calls and 4,498 Instruction constructions. These include necessary
candidate construction and validation; high counts do not prove redundancy.
Its exclusive C1–C6 spans are approximately 4.41/0.98/5.38/29.10/8.07/238.40 ms
**under the intrusive probe**. They are not unprofiled per-pass latency or a
natural C6 percentage, and must not be compared to the 32.47 ms primary E0 as
if both were the same cohort. PF-01 must demonstrate actual end-to-end/RSS
benefit before removing any work.

Reproduce the selection and deterministic gzip bundle from the two raw receipts:

```bash
python -m benchmarks.synthesis.performance_bundle \
  --initial /tmp/pf00-p0.json \
  --line-correction /tmp/pf00-line-correction.json \
  --output /tmp/pf00-bundle
```

The correction run uses the same driver command at its recorded commit with
three exact `--case` values: `energy-2x2:scalar:inline:line:success`,
`energy-2x2:qfunc:inline:line:success` and
`energy-2x2:qmod:inline:line:success`. Script/source snapshots remain in Git;
archived measurement/oracle bytes are not rewritten to the current verifier.

Validation remains scoped: the final PF-00/AO contract suite passes, strict
mypy/Ruff/Black/API/architecture checks pass, and the original AO artifacts and
production files are unchanged. The broader local default run has 17,667 passed,
540 skipped and 21 failures; its newly observed abbreviated-citation failure is
fixed and passes a targeted rerun, while the remaining 20 nodes were present in
the prior local baseline. The broader benchmark/release run has 1,054 passed,
95 skipped, one installed-package import failure and seven errors from a single
240-second MPS recovery fixture timeout. No full-local-suite green is claimed.

The initial gzip is stored as byte-identical ordered parts to respect the existing
1,800,000-byte repository file limit. Its `.parts.json` index records each part's
size/SHA-256 and the complete gzip SHA-256. The reader validates and joins those
bytes before decompression; no raw sample or original measurement byte changes.
The bundle command produces the parts automatically. A missing/modified part
fails validation. This is archive storage, not a change to either cohort.
