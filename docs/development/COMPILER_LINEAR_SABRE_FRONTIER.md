# Linear SABRE ready-front maintenance

Owner: Compiler. Classification: `single_device_fast_path`. Status: resource
admission pending. This is an internal implementation change, with no new API,
default, dependency, schema, routing trial, gate reorder or capability claim.

## Problem and implementation

SABRE previously removed every executable instruction from its ready list using
`list.remove`. A layer with F independent executable instructions therefore paid
O(F squared) list shifting. Parallel local layers need no routing search, but
still paid this cost on every layer.

Keep the existing executable comprehension. When every ready instruction can
execute, replace the frontier with an empty list. Otherwise, filter it once
using a temporary set of executable indices. Retain blocked instructions before
newly released successors, preserving their order and every existing tie.
When no instruction can execute, the current frontier already consists entirely
of blocked two-qubit operations and needs no second filter. Maintenance becomes
O(F), with O(F) temporary integer references and no cache or retained IR.

This applies the standard linear stable-partition pattern to FlagQuantum's
existing planner. It is not a new Qiskit or Cirq source port. Existing upstream
notices and SABRE scoring, restoration, pruning and recovery are unchanged.
It does not resolve the CZ/depth tradeoffs of commuting-frontier scheduling.

## Frozen resource control

Compare against integrated main `5b9cf3878664393e8667ca175ec8669f37fb3bc9`.
Load that exact SABRE module from Git and substitute only `_plan_sabre_trial` at
the same boundary as the candidate. Use fresh line couplings outside timing,
equal substitution lifetimes, one numerical thread and no concurrent local
test suite. Source programs are constructed outside timing. Observe the full
trial including distance/dependency construction, or the complete public router
including route selection, emission, restoration and evidence, as stated below.

| Width | Layers or CX instructions | Workload | Boundary |
| ---: | ---: | --- | --- |
| 256 | 1,024 layers | Adjacent disjoint CZ pairs per layer | Single trial |
| 512 | 512 layers | Adjacent disjoint CZ pairs per layer | Single trial |
| 6 | 64 instructions | Random ordered CX pairs, local seed 706 | Single trial |
| 12 | 128 instructions | Random ordered CX pairs, local seed 706 | Single trial |
| 6 | 64 instructions | Same random program | Public SABRE router |
| 12 | 128 instructions | Same random program | Public SABRE router |

Use two warmups and nine alternating retained samples per variant, plus one
separate traced Python allocation peak per variant after garbage collection.
Verify every warmup, timed and traced result outside timing against the same
complete-plan or serialized-router-output hash. Preserve all samples and all
failures. Run once under a 300-second process-group supervisor; do not rerun to
improve a median. Driver SHA-256:
`be46c59d5893a685a7ee8de9dd9da49ff98890136210c420c5fdb09eacafada1`.
Archive its complete source together with the first observations.

Admission requires both wide trial median ratios at most 0.95, all six ratios
at most 1.05, and every candidate Python peak at most 1.05 times baseline and
at most baseline plus 128 KiB. All observed outputs must be exactly identical.
These are prospective engineering gates on a shared, unqualified macOS host;
they establish neither statistical significance, native allocation/RSS limits,
formal SDKBench qualification nor an SDK ranking. Formal Linux measurement is
still deferred; no machine restart or kernel change is part of this work.

## Correctness and supported workflow

New scenarios cover 4-, 32- and 128-qubit parallel layers and a mixed frontier
whose distant CX waits while local instructions release successors. Existing
routing tests cover complete linear operators, tensor parameters and gradients,
classical dependencies, disconnected failures, direction, layout restoration,
pruning, recovery and deterministic ties. Keep those assertions unchanged.

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, route_to_topology

circuit = fq.Circuit(4).cz(0, 1).cz(2, 3).cz(0, 1).cz(2, 3)
routed = route_to_topology(circuit, CouplingMap.line(4), strategy="sabre")
assert routed.metadata["routing"]["inserted_swap_count"] == 0
assert routed.metadata["routing"]["mapping_restored"] is True
```

Run focused acceptance with:

```sh
python -m pytest tests/unit/test_compilation_sabre_frontier_order.py \
  tests/unit/test_compilation_sabre_routing.py \
  tests/unit/test_compilation_sabre_progress.py \
  tests/unit/test_compilation_sabre_layout.py \
  tests/unit/test_compilation_sabre_trials.py \
  tests/team/compiler/test_routing_conformance.py \
  tests/team/compiler/test_barrier_routing.py -q
```
