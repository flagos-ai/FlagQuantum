# Rejected shared CZ classification and synthesis

Status: rejected by the declared complete-workflow resource admission. The final
branch restores production code and failure-injection tests exactly to integrated
control `5a84f706ab9aeaab0ad3a5472bac292616fe39fe`. Candidate commit
`84880e05e76d70c24f8f98f2a7ff2c6f2ff6e031` remains in branch history for inspection.
This is a `single_device_fast_path` development experiment, not native-cost
routing admission, qualified SDK performance evidence or an SDK ranking.

## Mechanism and decision

The candidate shared the existing Qiskit-derived Weyl classification between
public two-qubit synthesis and private native block analysis. It avoided emitting
local factors for a known non-reducing CZ proposal only when the complete raw
instruction bound fit the existing limit. Otherwise full emission preserved the
priority of instruction-growth refusal. Reducing proposals still completed
synthesis, cleanup, complete-block checks and independent physical replay.
No new dependency, public contract, tolerance, budget or cross-call cache was
introduced. The public synthesis signature, return annotation and docstring were
unchanged, and the public API snapshot check passed.

This addressed the duplicate decomposition in the
[earlier rejected precheck](COMPILER_CZ_COST_PRUNING.md), but stage savings did not
meet the fixed complete-workflow admission. The first complete resource run is
retained, without replacement runs or threshold changes:

| Target-artifact control | Baseline median ms | Candidate median ms | Ratio | Baseline / candidate traced peak bytes |
| --- | ---: | ---: | ---: | ---: |
| QFT, 6 wires, grid | 105.226 | 90.589 | 0.8609 | 2198504 / 2191340 |
| Repetition, 6 wires, ring | 115.555 | 116.327 | 1.0067 | 2567021 / 2558541 |
| QFT, 12 wires, grid | 560.860 | 554.980 | 0.9895 | 10118150 / 10118150 |

The predeclared requirement was at least 5 percent lower median compilation time
on **all three** controls and no larger traced peak on any. Two controls failed
the time requirement, so this implementation is withdrawn. These shared-host
observations measure Python allocation peaks and adapter compilation time, not
process RSS, statistical significance or dedicated Linux qualification.

## Fixed controls and independent acceptance

The unchanged SDKBench source is
`9aa9260920984e2ca05bcb293abb885fccd9c144`. Candidate source hashes, the original
protocol, exact drivers and patch are retained in the
[source archive](evidence/shared-cz-synthesis-sources.json) and candidate commit.
The source protocol was committed before the first full control; it was not
published before execution. Inputs had already been inspected, so there is no
held-out or prospective campaign claim.

The complete set covers three fixed nonces, widths 6/8/12,
QAOA/Ising/hopping/QFT/repetition, line/ring/grid/sparse topologies and both
ordinary-restored and target-artifact workflows. All 360 baseline/candidate
pairs, or 720 executions, passed the unchanged acceptance. Normalized operations,
layouts, metrics, adapter audits, rewrite records and refusal counts were exactly
equal. Timestamp-bound artifact identities are not compared for equality.

The [screen index](evidence/shared-cz-synthesis-screen.json) references bounded
operation and row files with byte counts and SHA-256 hashes. Deduplication is an
archive representation only. Every original report can be reconstructed with its
original byte hash. The [fresh replay](evidence/shared-cz-synthesis-replay.json)
checks all outputs without importing FlagQuantum, Qiskit or Cirq: 240 full
operators and 480 three-probe acceptances. Larger-width probes do not prove full
operator equivalence. The largest accepted residual is approximately
`6.073e-14`.

The [resource archive](evidence/shared-cz-synthesis-resources.json) retains two
warmups and nine alternating retained samples per variant and control. Both
variants use AST-cloned functions, the same cached alias patch list and equal
context lifetimes. Fixture preparation and independent acceptance remain outside
`adapter.execute` timing and allocation tracing. Numerical libraries use one
thread, Python is 3.12.14, and no local tests, profiling or artifact transfers ran
concurrently with resource samples.

The [stage archive](evidence/shared-cz-synthesis-stages.json) includes all six
unchanged earlier reduction controls, two warmups and twenty alternating samples.
Both variants perform exactly one Weyl decomposition in every control. All four
reducing proposals and both refusal results agree; independent complete-operator
checks pass for the four proposals. The two generic refusals avoid local-factor
emission, but reducing cases include time and allocation regressions. These local
savings do not override the failed complete-workflow gate.

## Verification and limitations

The candidate passed 291 focused tests and source Ruff/mypy checks. The required
full default tier reached the 300-second supervisor limit at approximately 41
percent and was terminated; it is **not** a passing full-suite result. The source
archive preserves supervision diagnostics. An initial naive precheck failed an
existing resource-priority test; its patch and failing log are also preserved.
That failed approach was removed before the complete controls. The restored
implementation also passes all 291 focused tests. All 13 new evidence artifacts
pass revision/origin checks; English, local links and repository hygiene pass.
The full-tree origin check reports 16 existing historical declaration conflicts
in unrelated MPS evidence. Those declarations were not changed to suppress the
findings.

The final change retains evidence and this rejection record, with production
source and tests restored. Formal Linux resource qualification remains deferred
because the reserved host lacks `memory.peak`. No host, service, kernel or driver
was changed or restarted.

## Existing user workflow

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile

circuit = fq.Circuit(4).h(0).cx(0, 3)
result = compile(circuit, coupling_map=CouplingMap.line(4))
assert result.metadata["routing"]["mapping_restored"] is True
```

Expected: an equivalent, topology-legal circuit with restored output placement.
The evidence supports development of native target compilation behind existing
APIs; the rejected optimization does not alter this workflow.

```sh
python -m pytest tests/unit/test_native_block_optimization.py \
  tests/unit/test_compiler_sabre_native_cost.py \
  tests/unit/test_compilation_target_cz_resynthesis.py \
  tests/unit/test_compilation_two_qubit_synthesis.py \
  tests/integration/test_native_block_physical_evidence.py -q
```

## Follow-up sparse-product pilot

A separate [rejected pilot](evidence/native-block-sparse-product-pilot.json)
multiplied sparse matrix rows using one or two nonzero coefficients instead of
all four. Both variants retained the same matrix reader and every operator
column; the candidate additionally declined nonfinite coefficients. This was a
development-only replacement of the private block product, not a production
change or a proven new support boundary.

The first fixed three-control resource run used two warmups and nine alternating
samples per variant. Candidate/baseline medians were 0.9941, 1.0002 and 1.0098;
all three failed the unchanged 5 percent admission requirement. Traced peaks were
equal or slightly lower. The last saved outputs agree with baseline and pass
independent acceptance, but no complete application screen or general semantic
validation was run after this early rejection. The source freeze, exact driver,
all resource samples and pilot limitations are retained. Do not divide these
medians by earlier runs or claim a speedup from fewer arithmetic operations.

## Follow-up native-preview translation templates

A separate [rejected pilot](evidence/native-preview-template-pilot.json) translated
each source instruction and the inserted SWAP template once per experimental
selection, then remapped those native leaves to each candidate's physical
operands. The existing placement cache stayed call-local. Production routing,
native replay, final artifact emission, growth limits and independent SDKBench
acceptance remained unchanged. This was a reuse of the existing translator,
not a new upstream algorithm or a default native-CZ policy.

The first completed fixed resource control compared the previous and proposed
experimental consumers on integration
`39713983207cef13a7d2ea473a523236b43e4dd4`. Both sides used parsed function
definitions with equal patch lifetimes, two warmups and nine alternating samples.
The unchanged cases and nonce were already inspected; no local test suite ran
during these samples.

| Experimental target-artifact control | Baseline / candidate median ms | Ratio | Baseline / candidate Python peak bytes |
| --- | ---: | ---: | ---: |
| QFT, 6 wires, grid | 205.134 / 204.349 | 0.9962 | 2268622 / 2266354 |
| Repetition, 6 wires, ring | 203.571 / 199.841 | 0.9817 | 2600145 / 2601851 |
| QFT, 12 wires, grid | 564.557 / 558.645 | 0.9895 | 10118935 / 10118885 |

All three time ratios fail the existing 5 percent admission requirement, and
the repetition allocation peak increases. Withdraw the candidate rather than
retain another cache on these observations. The medians compare two experimental
consumers; they do not establish a gain over production or qualified performance.

Eleven focused preview/native-budget tests pass on the candidate, including a
new reversed-operand, multiple-layout operator control. The resource driver
checks the last baseline/candidate operation sequences and independently accepts
both for all three cases: full operators at width six, three probes at width
twelve. The archive retains the uncommitted candidate source, tests, exact
drivers, every timing sample, metrics and original report reconstruction hashes.
It does not retain emitted operations for a fresh offline replay. No complete
application quality screen or general semantic certification followed this early
resource rejection. Neither the pilot implementation nor its implementation-
specific translation-count assertion remains in executable repository code.

A subsequent [diagnostic profile](evidence/native-cost-current-profile.json)
records one production and one experimental execution of each control on the
unchanged integration above. All six independent acceptances pass. Each of the
two six-wire experimental executions calls native route selection three times
and preview realization six times. The twelve-wire experimental execution never
enters native route selection or preview realization: completed-trial filtering
has no alternative to evaluate. A preview-only optimization cannot improve that
control; its small timing difference does not establish an effect of the pilot.
This is a limitation of the attempted scope, not a reason to waive the fixed gate.

All six profiles also record 70 complete IR content-hash calculations. These
counts identify repeated work to investigate; they do not establish which
validation can safely be shared. Independent replay and mutable-content checks
remain necessary. The profile ran alongside repository-hygiene tests. Its timing
fields include profiling overhead and shared-host load, and cumulative times
overlap; use the call counts for diagnosis, not those times as latency evidence.

The SDKBench source pin in the template archive uses the existing external
dependency declaration in `evidence-revision-origins.toml`. Its repository URL
and full revision stay in the archive; the separate inline provenance vocabulary
is reserved for repository and producing-host history. The metadata repair
changes no archived source, measurement, or reconstruction digest.

## Follow-up Jacobi update pilots

The [Jacobi pilot archive](evidence/jacobi-update-pilots.json) retains two
withdrawn private numerical experiments on integration
`39713983207cef13a7d2ea473a523236b43e4dd4`. Neither changes the default compiler,
search budgets, native-growth limits, independent replay, or public contracts.

The first computed each symmetric mirrored pair once. It reproduced 1,000
exactly symmetric inspected inputs bitwise, and 245 focused tests passed. One
Qiskit anchor refused this host's 2.6.2 installation because the repository
certifies lanes 2.0 and 2.5; that comparison is not a pass. The first resource
attempt stopped at its first output-equality preflight, without a completed
resource report. Its partial timing arrays were not persisted, so no timing
conclusion is drawn from that attempt.

An untimed diagnosis found real M2 mixtures are not always exactly symmetric:
212 of 252 solver inputs differed across the diagonal, by at most
2.7755575615628914e-17. Mirroring changed 50 eigenpair outputs. Both complete
six-wire circuits passed the independent operator tolerance and retained CZ
count and depth, but their emitted angles differed. This invalidates the
micro-optimization's exact-output premise. Three original input/eigenpair
witnesses are retained for numerical replay; theoretical symmetry is not an
exact floating-point invariant.

The second proposal traversed the two halves together while calculating them
separately. All 231 focused synthesis, native-block and route-preview tests
passed; 1,003 inspected inputs, including the three real counterexamples,
matched the baseline eigenpairs bitwise. Its first completed resource control
used the unchanged three cases, two warmups and nine alternating samples, with
equal patch lifetimes and no concurrent local test suite.

| Experimental target-artifact control | Baseline / candidate median ms | Ratio | Baseline / candidate Python peak bytes |
| --- | ---: | ---: | ---: |
| QFT, 6 wires, grid | 189.089 / 201.922 | 1.0679 | 2268662 / 2268580 |
| Repetition, 6 wires, ring | 200.908 / 200.104 | 0.9960 | 2600195 / 2600145 |
| QFT, 12 wires, grid | 557.216 / 555.749 | 0.9974 | 10118935 / 10118935 |

Every time ratio fails the existing requirement of at least 5 percent improvement
on every control. Python peaks do not grow. The last output sequences and layouts
agree, and all six independent acceptances pass: full operators at width six,
three probes at width twelve. Emitted circuits were not retained for fresh
offline replay. No full application screen follows early rejection. The archive
preserves raw samples, source snapshots, tests, witnesses, and exact drivers;
both candidate implementations and their added tests were restored out of the
executable tree. These shared-host observations establish neither qualified
latency nor an SDK ranking.
