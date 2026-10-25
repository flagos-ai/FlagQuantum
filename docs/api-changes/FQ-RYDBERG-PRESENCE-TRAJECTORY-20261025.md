# One native Rydberg presence trajectory: shared baseline

The user explicitly approved all five decisions in
[proposal 092](../development/API_CHANGE_PROPOSAL_092_RYDBERG_PRESENCE_TRAJECTORY.md)
on 2026-10-25. Document PR #845 merged as
`7bd873856c13275c140e4be622560ab008bea60f` after complete applicable CI/GPU.
Each later PR requires its own conditional merge authorization. This baseline
does not activate the proposed Core method, native source 1.4, Lindblad plan
1.8, root input union or result-plan annotation.

## Replacement boundary

`tests/unit/test_rydberg_presence_contract.py` defines four implementation
factories: source, planner, runner and reader. Its independent physical consumers
must survive replacement unchanged. The disposable
`tests/unit/rydberg_presence_reference.py` composes the existing Rydberg frontend,
authentic native plan 1.7 validation/capture, decoded interval Hamiltonians,
Liouvillian, integrator, instrument probabilities, private single-index draw and
canonical ExecutionResult/MeasurementResult. It assembles the full native grid
once and does not restart native time at the observation.
The source-operation index is its canonical timed position, not an append count;
the reference reuses existing control-operation keys to locate that position.

The source double holds an observation outside its unchanged source 1.0.
Its serialization is explicitly `test.rydberg_presence_recipe`, containing
an authentic old native plan, not a fake ProgramRecord 1.4 or LindbladPlan 1.8.
No fake plan is attached to ExecutionResult. Production compatibility tests
keep the current method absent, new native reader admission closed and old
Lindblad plans refused at the root. These stage assertions are removed only
when their real owning domain changes; frozen physical consumers are retained.

The existing execution-plan candidate records the approved future root join
as `contract_baseline`, `production_implemented=false`. Its exact guard and five
negative mutations prohibit silently changing versions, adding result fields,
claiming implementation or expanding feedback. This metadata does not make
LindbladPlan an ExecutionPlan subtype or introduce a registry or new result type.

## Independent evidence

The oracle constructs Hamiltonians, loss jumps and row-vectorized generators
from declared scalar physics, without source/model/kernel matrices. Full
complex densities are compared to independent matrix exponentials with
atol=1e-10, rtol=0; fixed Kraus algebra uses atol=1e-12, rtol=0. Time-step
convergence is checked separately. A moving interacting pair checks roster
permutation and named-atom identity. Initial ground/Rydberg populations,
g/r coherence and lost-sector coherence, both forced branches, efficiency
zero/one, zero contributions, weighted unread equivalence, post-observation
boundary rows and earlier forward rows are covered. Lost-sector coherence is
a finite-basis algebra witness, not a hardware preparation claim.

Output names, order and axes, private Python/Torch RNG invariance, immutable
input/output snapshots, explicit seeds and exact same-environment JSON replay
are consumer assertions. The existing 4,096 single-index draw witness retains
its declared 1,856..2,240 interval. Full interval bindings change across the
observation and remain independent of later caller edits; the moving-pair
conditional density is checked against the same scalar physical oracle.
No random label is promised across builds.

The legacy fixture copies five complete source strings and six complete plan
strings captured before any 092 implementation. Stored artifacts retain exact
JSON bytes on every compatible reader. The sampled Ising artifact retains its
sealed Torch-build refusal on incompatible builds. Source reconstruction is
exact on every host; newly computed native matrices additionally retain exact
plan bytes on the captured Darwin/arm64/Torch profile. Cross-host numerical
results use the declared physics budget rather than decimal-string identity.
The original evidence capture is never overwritten.

True new-source/plan rehashed tampering, sealed options/environment admission,
dense resource ceilings, deployment refusal and root dispatch must be tested
against the real owning-domain implementations when those stages land. A test
recipe cannot certify production readers. The independent consumers and old
artifacts remain in place throughout Core, Simulation and Runtime delivery.

## Runnable existing public workflow

```python
import torch
import flagquantum.lindblad as fql
from examples.train_rydberg_controls import control_plan

plan = control_plan(torch.tensor([0.8, 1.2], dtype=torch.float64), intervals=8)
result = fql.run(plan)
replay = fql.run(fql.LindbladPlan.from_json(plan.to_json()))
print(plan.to_dict()["version"])
print(result.populations.shape == (9, 9))
print(torch.equal(result.populations, replay.populations))
```

The expected output is `1.7`, `True`, `True`. This existing unconditional
workflow supports the baseline without claiming native intermediate execution.
Run the replaceable consumers with:

```bash
python -m pytest tests/unit/test_rydberg_presence_contract.py tests/unit/test_rydberg_presence_compatibility.py -q
```

Feedback, multi-event/multi-shot trajectories, hardware, detector latency,
false positives, heating, erasure decoders and gradients through a random choice
remain outside this delivery. No maturity or full M3 completion claim changes.
