# Compilation-scoped commutation reuse results

Owner: Compiler. Classification: `single_device_fast_path`.

The implementation and protocol were published at
`e06fea47e6216f689ca750d9b353f75184baf150` before either draw.
All 480 observations pass unchanged SDKBench acceptance: 160 complete operators
and 320 limited three-probe checks. Every paired output, audit and returned
metric matches exactly. Both complete archives retain original compile times,
source pins, nonces, configuration, failures and hashed normalized payloads.

The first resource experiment passes its frozen development controls. Ordinary
QFT6/grid, repetition6/ring and QFT12/grid median ratios are respectively
0.927109, 0.965032 and 0.940522. The three target-artifact ratios are 0.992724,
0.997683 and 0.993580. All allocation observations pass the explicit limits.
These nine-sample shared-host diagnostics establish no significance or ranking.
Retain all 144 individually checked warmup, timed and traced payloads and the
original driver/protocol in the resource archive.

## Exact-main resource control declaration

The initial control disables the candidate's private scope, preserving the new
standalone analysis wrapper in both variants. It isolates reuse, but is not an
exact execution of integrated main: even the disabled-scope baseline pays for a
ContextVar lookup and an additional wrapper call. Do not present its speedup
ratios as exact-main speedup or sufficient production admission evidence.

Declare one distinct follow-up before its execution. Load main
`5b9cf3878664393e8667ca175ec8669f37fb3bc9`'s complete unchanged commutation
module and substitute its analyze function at all three consumers, disabling the
private scope. Compare that with the published candidate using its analysis and
scope, with equal patch lifetimes and fresh adapters. Do not change production
code, source pins, SDKBench acceptance, cases or thresholds. This is a different
baseline required for fair main comparison, not a favorable rerun of the first
experiment. Retain both controls regardless of outcome.

Use the same six fixed inspected case/workflow pairs, two warmups, nine
alternating samples and one traced peak per variant; verify all 144 execution
outputs/audits outside timing and independently accept the final timed outputs.
Freeze the driver hash before the sole follow-up run. Use one numerical thread,
240-second process-group budget and no simultaneous tests. Candidate medians
must be at most 0.95 times exact-main in at least two ordinary cases, no control
may exceed 1.05, and each traced peak must be within both 1.10 times baseline
and 128 KiB above it. If this exact-main control fails, withdraw the executable
candidate and retain its exact patch and all observations. Do not tune, loosen
thresholds or rerun for a better median.

Formal Linux qualification remains deferred. No machine or service is restarted.


## Exact-main observations and decision

The [exact-main control](evidence/scoped-commutation-main-resources.json) completes
once and passes its declaration, published at
`628fa6f1bc49e87d514ab2950d2986b7dc3c9de8` before execution. The frozen driver
loads the baseline's complete analysis and patches every captured analysis
consumer as well as the private scope. All 144 warmup, timed and traced outputs
and audits match the frozen quality archive. Final timed outputs pass independent
acceptance with the exact prior metrics. No favorable rerun occurred.

| Input | Workflow | Main median ms | Candidate median ms | Candidate/main | Main Python peak bytes | Candidate Python peak bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 06/qft/grid | ordinary-restored | 7.658 | 7.034 | 0.918488 | 472366 | 478897 |
| 06/qft/grid | target-artifact | 90.953 | 90.009 | 0.989622 | 2198358 | 2198292 |
| 06/repetition/ring | ordinary-restored | 6.343 | 6.046 | 0.953132 | 552760 | 565196 |
| 06/repetition/ring | target-artifact | 116.070 | 114.694 | 0.988141 | 2566909 | 2566859 |
| 12/qft/grid | ordinary-restored | 29.505 | 27.746 | 0.940356 | 2116779 | 2135036 |
| 12/qft/grid | target-artifact | 558.213 | 560.606 | 1.004287 | 10118879 | 10118779 |

The ordinary QFT6/grid and QFT12/grid cases meet the required 5% reduction;
repetition6/ring improves less and does not meet that individual speed condition.
The declared aggregate requires two of three, not all three. The target-artifact
controls remain under 1.05. Ordinary traced peak increases are respectively
6,531, 12,436 and 18,257 bytes, within both 10% and 128 KiB. No extra allowance
is added after observation. The largest target-artifact ratio is 1.004287;
values close to one establish neither significance nor exact performance parity.

Retain the candidate for full CI review. Both resource controls pass their
frozen gates, with their distinct baselines clearly identified. This is evidence
for a bounded compilation-local analysis tradeoff, not for improved CZ counts,
depth, routing quality, the pending controlled-gate extension or the entire
compiler objective. Full exact-head CI and ordinary protected integration remain
required; the PR stays draft until those finish.

## Complete archives and independent replay

The [inspected index](evidence/scoped-commutation-inspected-screen.json) and
[fresh index](evidence/scoped-commutation-fresh-screen.json) reconstruct their
original 3,707,970-byte and 3,579,193-byte reports byte-for-byte. They retain
104 and 106 distinct normalized outputs respectively in three hashed chunks per
draw. Every one of 480 original observations remains present. The
[fresh producer-free replay](evidence/scoped-commutation-replay.json) validates
all rows, fixture hashes, source pins, output hashes, layouts and returned
metrics without importing FlagQuantum, Qiskit or Cirq. Maximum semantic residuals
are 6.072657351995283e-14 and 3.3382154094629775e-14 against unchanged 1e-8.
Three-probe rows retain their explicitly limited coverage.

Both resource records retain driver and protocol source, nine samples per
variant, allocation observations and 144 payload verifications each. The
[initial resource archive](evidence/scoped-commutation-resources.json) isolates
reuse with a disabled-scope baseline; the exact-main record addresses its wrapper
overhead limitation. There are 288 verified execution payloads across the two
different controls. Tracing covers complete adapter execution, excluding adapter
construction, and does not measure native allocations or RSS.

## Verification scope

Fourteen new workflow/lifecycle tests and 149 expanded commutation, cancellation,
scaling, optimization-level and native physical-evidence tests pass. The sparse
default tier passes 241 tests with 42 deselected. Complete changed commutation and
pipeline modules plus the new lifecycle test module pass strict Linux-targeted
mypy, Ruff and Black. An import-boundary test admits only the two additional
standard-library context modules while retaining its exact ban on simulation
imports. This is scoped local evidence, not a claim that full repository CI or
the complete historical evidence corpus has passed.

The source stays at the published implementation. No public API, default,
serialized schema, dependency, maturity or layout change is introduced. Formal
exclusive Linux qualification remains deferred and no machine or service was
restarted. Existing user workflows and their complete operators stay unchanged.
