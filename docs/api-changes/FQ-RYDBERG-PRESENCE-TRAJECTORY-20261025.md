# One native Rydberg presence trajectory: staged delivery

The user explicitly approved all five decisions in
[proposal 092](../development/API_CHANGE_PROPOSAL_092_RYDBERG_PRESENCE_TRAJECTORY.md)
on 2026-10-25. Document PR #845 merged as
`7bd873856c13275c140e4be622560ab008bea60f` after complete applicable CI/GPU.
Each later PR requires its own conditional merge authorization. Baseline #854,
Core #869, Simulation #875 and Runtime #879 have landed separately. Runtime joins
strict native source 1.4 to sampled LindbladPlan 1.8 and `fq.run(plan)`.
It preserves the existing result classes and adds only the approved root input
and result-plan type annotation. Runtime merged as
`21b8e7f6e34a638e5120a0f4d7fff64fe196adb2` after complete CI `40255122443`
(attempt 2), GPU `40255122490`, the three Python matrix jobs and every applicable
check succeeded. Main smoke `40258387265` then succeeded. The local default and
integration suites passed 15,995 and 2,481 cases respectively, with unchanged
skip policy and zero XML failures/errors. The smoke result does not certify a
full main CI, release, hardware path or all M3 gates.

## Replacement boundary

The four factories in `tests/unit/test_rydberg_presence_contract.py` now select
real RydbergProgram, Lindblad planning, root execution and Lindblad JSON reading.
All independent physical consumers and oracle functions remain unchanged.
Their 29 AST hashes were compared before and after replacement. The disposable
test reference was deleted. Real source/version/options/environment tampering is
checked against the production reader, alongside original byte captures.

Simulation owns full-schedule assembly, fixed Kraus actions, prefix evolution
and selected continuation. Runtime owns sealed options, environment compatibility,
private drawing and canonical result assembly. One original absolute grid and
all interval bindings are retained. No new IR, solver, registry or result class
is introduced. The candidate records the approved `runtime` stage and exact
signature guard; this does not promote capability maturity.

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

Real source/plan rehashed tampering, sealed options/environment admission,
dense resource ceilings and root dispatch are checked in
[the Runtime tests](../../tests/unit/test_rydberg_presence_runtime.py).
[Compatibility consumers](../../tests/unit/test_rydberg_presence_compatibility.py)
retain old artifacts and refuse relabelled profiles. The independent physical
consumers now select real Core construction, planning, execution and JSON
reading through their four factories; the disposable reference is gone.

## Runnable public trajectory workflow

```python
import torch
import flagquantum as fq
import flagquantum.lindblad as fql
from examples.train_rydberg_controls import moving_pair_program

program = moving_pair_program().measure_presence("a", at=0.5, efficiency=0.8)
plan = fql.plan(
    program, program.basis_state(("g", "g")),
    torch.linspace(0, 1, 129, dtype=torch.float64),
    parameter_values=[{"omega": 1.2}] * 128,
    outputs=fq.expectation(program.survival_operator(), name="survival"),
    options=fq.ExecutionOptions(shots=1, seed=17),
    return_density_matrices=True,
)
result = fq.run(fql.LindbladPlan.from_json(plan.to_json()))
print(plan.to_dict()["version"], result.state.shape)
print(result.expectation("survival").shape)
print(result.measurement("mid_circuit_measurement").value.shape)
```

Expected: `1.8 torch.Size([1, 9, 9])`, `torch.Size([129])`, and
`torch.Size([1, 1])`. Earlier trajectory rows are forward states; the observation
boundary row is post-observation. `no_flag` includes undetected loss. `fql.run`
continues to return EvolutionResult for old profiles and refuses sampled 1.8.

Run the replaceable consumers with:

```bash
python -m pytest tests/unit/test_rydberg_presence_contract.py tests/unit/test_rydberg_presence_compatibility.py -q
```

Feedback, multi-event/multi-shot trajectories, hardware, detector latency,
false positives, heating, erasure decoders and gradients through a random choice
remain outside this delivery. No maturity or full M3 completion claim changes.
