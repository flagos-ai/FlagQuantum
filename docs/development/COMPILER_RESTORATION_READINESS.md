# Readiness-guarded physical layout restoration

Owner: Compiler. Classification: `single_device_fast_path`.
Status: rejected and withdrawn. Production code is byte for byte identical to
the declared integrated baseline. A separate test improvement below is retained. The candidate implementation
and its independent tests remain at published freeze
`3561f271169257334dbbed54c65045e611a83c6a`.

The rejected graph-path experiment at
`0692118febfd7a8f5e3ee17ee32121fc913d494c` showed that fewer restore SWAPs
can increase delivered dependency depth. Untimed causal tracing reproduces
20 inspected outputs and audits exactly: QAOA12/grid changes its forward route,
whereas QFT12/sparse retains the identical forward route in both workflows but
changes the restore from 22 to 18 SWAPs and increases final depth. Preserving
forward routing alone therefore does not solve the observed problem.
The [trace](evidence/restoration-selection-diagnosis.json) records every route
materialization, plan, forward hash and restore sequence. No timing claim follows.

## Algorithm and proof boundary

Retain the existing spanning-tree leaf restore. On cyclic graphs, propose the
same leaf order with paths in all remaining graph edges. Fixed vertices are
excluded, so they cannot be displaced; removing a tree leaf keeps each remaining
component connected. Accept the proposal only if it has strictly fewer SWAPs
and its output readiness is componentwise no later for **every** incoming wire
readiness vector. Ties, tree topologies, unreachable component exchanges and
unused device-wire exclusion retain the existing behavior.

A unit-duration SWAP sets both wire times to one plus their maximum. Composing
these operations gives a max-plus linear map: every output is a maximum of
input times plus integer delays. Compare one input column at a time, representing
an absent dependency by -1. If every candidate coefficient is at most its
baseline coefficient, output readiness cannot increase for any input times.
The same implication holds for any common positive SWAP duration, including
three serial CZ layers. The proof retains two linear-size columns, not a dense
matrix or numerical circuit state. It has at most
`n_wires * (candidate_swaps + baseline_swaps)` scalar edge updates, capped at
65,536 per comparison. Exhausting that budget retains the existing tree result.
There is no cross-call cache, random search or new dependency.

This proves a boundary property for the restore sequences. It does **not** prove
nonregression after forward-trial selection, native lowering, block synthesis,
local rotations, heterogeneous durations, or device scheduling. Changing restore
prices can still change the selected forward route. Complete-workflow gates
below remain necessary; an isolated proof is not permission to claim global
quality or admit a failing candidate.

Qiskit's `ApproximateTokenSwapper.map` was inspected for mature graph-token
swapping practice. It delegates multi-trial randomized swapping to rustworkx;
this candidate extends the existing FlagQuantum leaf walk rather than porting
that implementation or adding its dependency. Existing Qiskit/Cirq Apache
notices remain with their separately reused planner mechanisms. FlagQuantum IR,
target contracts, evidence schemas, public signatures/defaults and physical
identity-output requirements remain unchanged.

Tests compare the guard with independent clock execution for all pairs of
three-wire SWAP sequences of length at most three and all arrivals in
{0, 1, 7}; test delayed-input counterexamples and a direct transposition versus
a palindromic path; enumerate all five-wire ring permutations for token replay
and readiness; retain historical tree-count references; and check exhausted
proof budgets. The delayed-input witness initially used all-zero arrivals,
which did not exhibit its delayed-input violation; it was corrected to the
actual delayed arrival before any complete-workflow observation.

Local verification passes 1,485 selected compiler unit/integration, routing
conformance, consumer replacement, target-artifact and golden-path tests. Ruff,
Black and strict mypy pass all five changed Python modules. This is not the
whole repository default/runtime tier; required full exact-head CI remains
outstanding.

## Frozen prospective admission

Publish candidate and protocol before complete-workflow observations. Baseline:
integrated main `b6e606ea29298912a2c363b423253b3e4f3f30c5`. Load its original
restoration function in the same compiler globals. Verify all AST nodes outside
restoration, its private guard/budget and the path helper docstring are identical.
SDKBench remains clean at `9aa9260920984e2ca05bcb293abb885fccd9c144`.
Freeze source/driver hashes, dependency versions, both adapter configurations
and compiler seed 11. Use ordinary-restored and target-artifact workflows.

Each draw runs all 60 inputs: widths 6/8/12, five application families, four
topologies, both variants and workflows (240 executions). First use nonce `01`
repeated 32 times. After publication generate exactly one fresh 32-byte nonce,
freeze it before observation, and never redraw or tune to it. Every output must
satisfy unchanged numerical, topology, identity-layout and capacity acceptance.
Every paired CZ count and two-body dependency depth must be nonincreasing;
at least one pair in each draw must strictly improve one metric. Every producer
build and compilation count must remain one, execution count zero, and all
other invariant audit fields must match. Native baseline CZ count, accepted
rewrite count and serialized artifact bytes may differ with the compiled output;
retain them and validate that they are nonnegative integers. Unlike the prior
rejected protocol's equality requirement, this new prospective rule recognizes
those three fields as derived quantities before observing this candidate.
No prior result is reclassified. No SDKBench adapter or evaluator changes.

Recheck all 480 serialized outputs in a fresh process without producer SDK
imports. Width six uses full operators; larger widths use three probes only.
After both quality gates pass, measure inspected QFT6/grid, repetition6/ring and
QFT12/grid, both workflows, two warmups, nine alternating retained rounds and one
separate Python-allocation-traced round per variant. Fresh adapters, equal patch
lifetimes and one numerical thread; no concurrent tests. Independently verify
all 144 outputs, hashes, metrics and audits against quality references outside
timing. Every median time ratio must be at most 1.05. Every Python peak must be
at most both 1.10 times baseline and baseline plus 128 KiB. No classical speedup
is required: the intended benefit is less delivered quantum work. Python tracing
does not measure native allocations, RSS or cgroup peak memory.

Any failed gate rejects admission. Keep first outcomes, unsupported cases and
tradeoffs; no favorable rerun, omitted case, widened threshold or fixture tuning.
Quality draws have 900-second process-group budgets; resource/replay phases
have 300 seconds. Reports are capped at 16 MiB and created exclusively. Keep
sources immutable during observations. Exact-head full CI is required before
merge. Formal Linux qualification remains deferred; never restart a machine,
upgrade a kernel or infer cross-SDK ranking/scalability from these local controls.

Frozen driver `fq-restoration-readiness-control-20261123.py`, SHA-256
`16ea1315d0be5d64ae79dd98c0d25a85ec1a507d9996990d3444254ed2b47d11`.
Replay `fq-restoration-readiness-replay-20261123.py`, SHA-256
`e876ed82f8bb2fb99de3504d55230984847b046275a9ac14be3c0fe14859864a`.
Sources are embedded in their resulting reports.

## Existing supported workflow

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, route_to_topology

circuit = fq.Circuit(6).h(0).cx(0, 3).cx(1, 4)
routed = route_to_topology(circuit, CouplingMap.ring(6), strategy="sabre")
assert routed.metadata["routing"]["final_logical_to_physical"] == tuple(range(6))
assert routed.metadata["routing"]["mapping_restored"] is True
```

Expected: legal routing with physical identity output. Short contributor check:

```bash
python -m pytest -q tests/unit/test_compilation_restoration_readiness.py tests/unit/test_compilation_restore_paths.py tests/unit/test_compilation_sabre_layout.py tests/unit/test_compilation_sabre_trials.py tests/team/compiler/test_routing_conformance.py
```


The candidate-specific test command above applies to the published freeze;
that test module was removed with the rejected implementation.

## First outcomes

Both frozen draws complete: all 480 executions pass unchanged numerical,
topology and physical identity-layout acceptance. Producer-free replay accepts
all 480 serialized outputs again, with maximum residuals
6.072657351995283e-14 and 2.6832735025352902e-14 against 1e-8. The scope is
160 full-operator and 320 limited three-probe observations. Invariant audit
counts match; derived artifact/rewrite counters remain explicitly recorded.

The inspected draw has three quality violations; the fresh draw has two.
The first draw changes five payload pairs, the second changes eight (including
one equal-cost payload change); every payload remains in the archive. Neither
failure is excused by the readiness proof or the improvements elsewhere.

| Draw / input | Workflow | Baseline CZ / depth | Candidate CZ / depth |
| --- | --- | ---: | ---: |
| inspected repetition6/grid | ordinary-restored | 72 / 59 | 72 / 57 |
| inspected repetition6/grid | target-artifact | 62 / 51 | 66 / 51 |
| inspected QFT8/sparse | ordinary-restored | 182 / 105 | 164 / 108 |
| inspected repetition12/sparse | target-artifact | 70 / 58 | 68 / 62 |
| fresh repetition6/grid | ordinary-restored | 72 / 64 | 72 / 65 |
| fresh repetition6/grid | target-artifact | 68 / 60 | 68 / 61 |
| fresh QAOA12/grid | ordinary-restored | 180 / 79 | 162 / 59 |
| fresh QAOA12/grid | target-artifact | 176 / 77 | 158 / 57 |

No resource phase was run because its prerequisite quality gates failed.
After withdrawal, 115 focused restoration, routing, artifact and golden-path
tests pass; production and existing test files match the declared baseline.
There is no favorable rerun, changed threshold, omitted case or fixture
exception. This is a within-FlagQuantum development control, not a cross-SDK
ranking. A max-plus proof for homogeneous SWAP dependencies cannot certify
post-selection target synthesis. Future work must compare delivered native
cost/depth without bypassing retained topology replay or paying uncontrolled
repeated synthesis costs. This proposal is not enabled for any workflow.

[Inspected index](evidence/restoration-readiness-inspected-screen.json),
[fresh index](evidence/restoration-readiness-fresh-screen.json) and
[producer-free replay](evidence/restoration-readiness-replay.json) retain the
complete first observations, source hashes, protocols and exact drivers. Each
index lists hashed payload chunks and original top-level key order for exact
reconstruction of the original pretty-printed report plus trailing newline.
Original report SHA-256 values:
`b4ef5b516fb5a5c25d913f62e75dbd42ac20cd4d9786f31a1f7d39735a4f0357`
and `fa25e114f104959b5992bf6a100c1fba1ae80cfac88b2e6fb243c71d9ce347bd`.


## Retained layout-input coverage

After withdrawing the optimization, the layout-plan reuse scenario is extended
independently to include both identity input (seed 3) and a searched nonidentity
physical input (seed 0). Its exact GF(2) operator oracle initializes each physical
wire with its declared logical input bit. The nonidentity case explicitly differs
when incorrectly interpreted as identity-input execution, so it exercises the
previously absent semantic boundary rather than merely accepting the old sample.
The existing single-search assertion, estimate/emission count agreement and
independent exact operator check remain mandatory. Automatic deployment's
identity-input contract is unchanged.

All 83 selected trial, layout, routing conformance, target-artifact and golden-path
tests pass after this retained change. Ruff, Black and strict mypy pass the whole
changed test module. Compiler production sources remain identical to the baseline;
no claimed CZ/depth improvement is delivered by this test change.
