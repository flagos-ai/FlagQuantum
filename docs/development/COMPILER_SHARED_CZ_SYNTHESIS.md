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
