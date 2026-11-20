# Native SABRE relative scoring

Status: implementation and development validation; no SDK parity or superiority
claim. Execution classification: `single_device_fast_path`. Compiler owns the
change; it introduces no public API, dependency, IR or serialization change.

## Upstream adaptation

The inspected source is Qiskit 2.6.2, revision
`0c26a5a158c4d62b2b303a8c2f3c6eb2b14653b4`,
[`State::choose_best_swap` in route.rs](https://github.com/Qiskit/qiskit/blob/0c26a5a158c4d62b2b303a8c2f3c6eb2b14653b4/crates/transpiler/src/passes/sabre/route.rs).
Its candidate loop accumulates changes in layer distance instead of summing every
interaction's distance for every proposed SWAP. The current-layout absolute
score is common to all candidates when no endpoint decay multiplier is applied.

FlagQuantum adapts this mechanism inside its existing planner. For each blocked
frontier, it indexes weighted front/extended interactions by physical endpoint.
A proposed SWAP changes only interactions incident on its endpoints. An
interaction touching both endpoints is visited once. Existing frontier candidate
selection, dependency graph, extended-layer BFS order, lookahead decay, source
anchors, device admission, layouts and final restoration remain authoritative.
This is an implementation adaptation of relative scoring, not the complete
LightSABRE implementation or Qiskit routing pipeline. No Cirq algorithm is added
in this change; Cirq local cleanup and routing progress recovery have separate
implementations and review records.

The IBM copyright and Apache-2.0 modification notice accompany `sabre.py`.
FlagQuantum's repository LICENSE supplies the license. No upstream DAG, layout,
Rust extension, backend object or new runtime dependency crosses the boundary.
The private scorer is replaceable under the existing planner without modifying
its callers; tests replace it with an exhaustive absolute-score oracle.

## Numerical behavior and bounded work

The scoring objective is unchanged mathematically. Relative differences can have
different floating-point rounding from absolute summation. If relative costs
are within `4 * term_count * ulp(max(1, current_absolute_magnitude))`, compare
those candidates again using the original absolute evaluation order. Sorted
candidate order resolves exact ties. This avoids introducing a new heuristic
tolerance that deliberately changes routing quality. It is not a proof of
bit-identical plans for every floating-point input.

On devices of twelve or fewer owned qubits, retain the existing full-rescoring
loop: incidence construction can cost more than it saves on small layers. This
threshold is an internal implementation choice, not a supported SDK option or
a workload-specific benchmark shortcut. Larger devices use the same scoring
objective with indexed evaluation; no fixture IDs are consulted.

If there are F front and E extended interactions, index construction stores
O(n + F + E) references. Each candidate reads the union of incident interactions
instead of all F+E terms. Near ties still require full rescoring. The existing
extended-layer limit, search stop condition, topology checks, and caller resource
limits remain in force. This change does not make a global planner memory or
wall-time guarantee and does not resolve all routing efficiency gaps.

## Reproduction and acceptance

The development timing probe loads the immutable pre-change planner source from
`1e2fbd3f45687afc2ba0e057f2f0f01c67a9e53e` and runs both implementations
in one process on the same generated circuits and topology. Each circuit has
four seeded shuffled perfect matchings on a line, at widths 6, 12, 16, 24, 32
and 64. One warmup and five retained samples alternate implementation order.
Raw samples, environment, source hashes, SWAP counts and complete-plan equality
are emitted. Run on a quiet host, separately from tests:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   python -m benchmarks.compiler_sabre_scoring > /tmp/sabre-scoring.json
```

This measures planner calls, not target lowering, end-to-end compilation, memory,
QPU performance or comparison with another SDK. The circuits are inspected
development inputs, not prospective application evidence. Preserve every row
and any regression. Source lookup needs the retained Git revision locally.

Focused correctness checks:

```bash
python -m pytest tests/unit/test_compilation_sabre_scoring.py   tests/unit/test_compilation_sabre_routing.py   tests/unit/test_compilation_sabre_layout.py   tests/team/compiler/test_routing_conformance.py -q
```

Acceptance includes every four-wire permutation and pair combination under
weighted lookahead, repeated interaction terms, exact ties, an empty candidate
set, complete-plan replacement against absolute scoring on sixteen-wire grids,
and a fourteen-wire workflow comparing complete output states and shared-angle
gradients. Existing routing tests cover symbols, classical dependencies,
barriers, disconnected topologies, layout restoration and source evidence.

The user workflow remains:

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile

source = fq.Circuit(16).h(0).cx(0, 15).cx(3, 12)
compiled = compile(source, coupling_map=CouplingMap.grid(4, 4))
assert compiled.metadata["routing"]["mapping_restored"] is True
```

The scorer changes only internal planning cost. It does not reduce CZ counts by
construction, introduce multi-trial search or paired-SWAP lookahead, admit free
artifact layouts, change protected defaults, or certify comparative performance.
Independent unchanged SDKBench resource-qualified campaigns remain required.

## Validation record (2026-11-19)

The focused routing and artifact workflow selection passed 182 tests, including
complete-state and shared-parameter-gradient comparisons. Whole-package mypy
passed for 1,133 source files with Python 3.12, including the development probe.
The protected public API snapshot, scoped repository language check, local
Markdown links and compiler ownership check passed.

An initial broad unit run detected the new internal parameter name `wires`
through the repository vocabulary census. The implementation now calls that
parameter `qubit_pairs`; the pinned census and its acceptance criteria were not
changed. The corrected scorer and complete vocabulary test file passed 79 tests.
The development probe retained all 18 complete baseline plans. That broad run and a concurrent integration run were stopped before
completion to apply the correction. Neither is passing-suite evidence. Complete
CI checks remain required before integration.

The local timing probe shares a host with other active workloads. Its timing
samples are diagnostic only, and no speedup estimate is promoted here. A quiet,
resource-controlled run is required to establish the performance effect.


## Integration status

PR #1232 integrated this scorer at
`15ba2367e70d76ad374297098687fac1b1949259` after its required checks.
The interrupted local attempts above remain historical non-passing evidence.
The combined native compiler and prospective qualification scope are recorded
in [the upstream audit](COMPILER_UPSTREAM_REUSE.md); merging does not establish
a performance advantage over another SDK.
