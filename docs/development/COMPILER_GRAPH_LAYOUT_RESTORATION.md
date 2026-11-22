# Graph paths for physical layout restoration

Owner: Compiler. Classification: `single_device_fast_path`.

Status: experimental candidate; complete-workflow admission is pending.

## Requirement and implementation

A physical restore must return every logical qubit to its own physical wire.
The existing spanning-tree leaf walk can miss a real device edge: swapping
tokens 3 and 4 on a six-wire ring takes nine tree SWAPs although they are adjacent.
The candidate retains that tree walk and tries a second walk in the remaining
induced coupling graph, using the same leaf order. It returns the second walk
only when it uses strictly fewer SWAPs. Trees keep their original single walk;
ties keep the original trace. No unused device wire becomes workspace.

Removing a tree leaf preserves connectivity among the remaining vertices of
that component. A path excludes fixed vertices, so already restored tokens
cannot move. Both walks terminate after one removal per circuit wire. The
second walk uses at most one breadth-first search per displaced leaf, linear
temporary storage, and no cache or numerical matrix. There is no claim of an
optimal solution on general graphs or a global depth guarantee. Its shorter
restore changes route pricing and can therefore change the selected forward
route; complete-workflow quality must be checked separately.

Qiskit's `ApproximateTokenSwapper.map` was reviewed: it delegates randomized
multi-trial graph token swapping to rustworkx, based on Miltzow et al.'s token
swapping work. This candidate instead extends the existing FlagQuantum leaf
walk without a dependency or a new random search. It is not a port of that
upstream implementation. Existing Qiskit/Cirq source notices remain attached
to the separate planner mechanisms they cover. No public signature, default,
result field, evidence schema, target admission or layout contract changes.

The initial focused run exposed eight historical tree-count assertions and
one test that assumed a caller-selected `sabre_layout` had identity input.
Historical counts are retained exactly by replacing the graph walk with its
tree-only implementation in that test's reference context. The new candidate
still must beat those bounds. The caller-selected strategy test now initializes
its independent GF(2) reference with the plan's declared input placement; this
strategy has always allowed nonidentity input, unlike automatic deployment.
No automatic-workflow reference is changed. The corrected focused selection
passes all 95 tests. New tests enumerate all four-wire graphs and permutations
against a breadth-first token-state oracle, and cover cliques, cyclic components,
padding exclusion and the adjacent-token ring case.

The expanded local selection passes 1,484 compiler unit/integration, routing
conformance, consumer replacement, target artifact and golden-path tests.
A further historical 92-SWAP assertion retains its tree-only reference in the
same manner. Strict mypy checks all five changed Python modules; Ruff and Black
also pass. Earlier wider collection attempts lacked synthesis examples in the
sparse checkout; those are setup failures, not passing test evidence. Required
full repository tiers remain outstanding in CI; this selected run is not their
replacement.

## Prospective admission protocol

Publish this implementation and protocol before complete-workflow observations.
Baseline: integrated main `dbe944ca181c01c2e4d98dc221b3c39075c99dd8`.
Only the restoration implementation differs between control variants. Load the
baseline restoration functions from that revision into the same module globals;
verify that every other AST node is identical. Both variants use unchanged
SDKBench `9aa9260920984e2ca05bcb293abb885fccd9c144`, compiler seed 11,
and its ordinary-restored and target-artifact adapter configurations.

Freeze source hashes, driver hash, versions and adapter options. Run all 60
inputs (widths 6, 8, 12; five families; four topologies), both workflows and both
variants: 240 observations per draw. First use nonce `01` repeated 32 times;
after publication generate exactly one fresh 32-byte nonce and freeze it before
observing its draw. Retain every result and failure without redraw or tuning.

Every output must satisfy unchanged numerical, topology, identity-output-layout
and capacity acceptance. Every paired full CZ count and two-body dependency
depth must be nonincreasing; at least one pair per draw must strictly improve a
quality metric. Audit records must remain equal. Serialized outputs may differ.
Width six uses full operators; widths eight and twelve use three limited probes
and do not prove complete operator equality. Recheck all serialized outputs in
a fresh process without importing a producer SDK. No SDKBench code changes.

After both quality draws pass, measure the inspected QFT6/grid,
repetition6/ring and QFT12/grid cases in both workflows: two warmups, nine
retained alternating rounds, one separate Python-allocation-traced execution
per variant. Use one numerical thread, fresh adapters and equal patch lifetimes.
Validate all 144 outputs, hashes, metrics and audits against the frozen quality
results outside timing. Retain every sample. All six median wall-time ratios
must be at most 1.05; Python peaks must satisfy both 1.10 times baseline and
baseline plus 128 KiB. The intended benefit is less delivered quantum work,
not a required classical speedup. Python tracing does not measure native
allocation, process RSS or cgroup memory. Ratios are descriptive local evidence.

Any failed gate rejects this candidate. Preserve its first outcome and do not
rerun for a favorable outcome, omit cases, relax limits or tune to the fresh
draw. Quality phases have 900-second process-group budgets; resources and
producer-free replay have 300 seconds each. Each raw report is capped at
16 MiB and created exclusively. Sources stay immutable during observations,
with no concurrent tests. Required exact-head CI remains necessary for merge.
Formal Linux qualification remains deferred; no cross-SDK ranking, scalability
or hardware-support claim follows. Never restart a machine or upgrade a kernel.

Frozen driver: `fq-graph-restoration-control-20261122.py`, SHA-256
`1225f208545621ad00f879ce1279654da1518360ea38d7d30273c7bafebe74f0`.
Producer-free replay: `fq-graph-restoration-replay-20261122.py`, SHA-256
`89513e707f24a524471984dae07080fb904817abe8b4bdcc0a0606bae31629c6`.
Both full sources are embedded in their respective reports.

## Existing user workflow

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, route_to_topology

circuit = fq.Circuit(6).h(0).cx(0, 3).cx(1, 4)
routed = route_to_topology(circuit, CouplingMap.ring(6), strategy="sabre")
assert routed.metadata["routing"]["final_logical_to_physical"] == tuple(range(6))
assert routed.metadata["routing"]["mapping_restored"] is True
```

The supported route still reports and physically restores identity output.
Shortest contributor check:

```bash
python -m pytest -q tests/unit/test_compilation_graph_restoration.py tests/unit/test_compilation_restore_paths.py tests/unit/test_compilation_sabre_layout.py tests/unit/test_compilation_sabre_trials.py tests/team/compiler/test_routing_conformance.py
```
