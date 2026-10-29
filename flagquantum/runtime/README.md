# Runtime

Turn a quantum program or accepted execution plan into an observable execution
attempt. The design goal is consistent program, gradient, and result semantics
across local, distributed, and remote resources, with a direct local fast path.

Runtime owns selection, execution lifecycle, coordination, recovery, and result
assembly. Compiler transforms programs; Simulation computes numerical updates;
Compute and Remote adapt resources through their contracts.

## Follow the local path

```text
fq.run(program) → plan → execute accepted plan → Simulation → ExecutionResult
```

`fq.run(plan)` must execute the supplied plan without planning or compiling it
again. The result retains that plan and reports the observed execution route.

The experimental IsingBraidProgram uses this same path for the reviewed
four-sigma vacuum sector. `_ising_plan.py` is native admission behind the existing
plan codec, not another plan class or execution manager. Native plans alone use
version 1.1 for braid-only sources, 1.2 for one unread observation, or 1.3
for one sampled observation, seal matrix outputs and have `analysis=None`; circuit plans
retain version 1.0 and their actual CircuitAnalysis. Dispatch authenticates the
profile from the sealed payload and delegates finite F/R actions to Simulation.
Source snapshots, matrix snapshots and exact environment preflight also apply
to JSON replay. No qubit count or elapsed time is inferred from fusion labels.
The planner and reader bind native plan/source versions exactly, before output
allocation or source decoding. Version 1.2 requires exactly one ordered
`dephase_pair()`; its unread channel may yield a mixed density. No intermediate
outcome, event, classical bit or feedback record is produced. Rehashing cannot
bypass the version or source semantics.

Run `python -m examples.ising_braid_execution` for direct execution, exact JSON
replay and explicit terminal QND charge sampling. It reports plan 1.1, null qubit
count, state shape `(1, 2, 2)`, `P(psi): 0.5`, exact replay and 2,000 terminal
shots without changing caller RNG state. This bounded CPU complex128 route has
no gradients, alternate backend, remote submission, general multi-shot execution
or circuit calibration. Each requested matrix expectation retains its own existing
MeasurementResult; no native result wrapper is introduced.

| Change | Entry point |
| --- | --- |
| Selection and estimates | [planner/](planner/README.md) |
| Plan representation and validation | [execution_plan.py](execution_plan.py), [execution_plan_contract.py](execution_plan_contract.py) |
| Plan execution and dispatch | [plan_execution.py](plan_execution.py), [execution.py](execution.py) |
| Representation lifecycle | [executors/](executors/README.md) |
| Shared rank coordination | [distributed/](distributed/README.md) |
| Stochastic work and recovery | [trajectories/](trajectories/README.md) |
| PyTorch model and optimization loop | [module.py](module.py), [training.py](training.py) |

## Verify a change

From the repository root:

```bash
python -m pytest tests/integration/test_cpu_vertical_slice.py tests/team/runtime -q
python tools/check_architecture.py
python tools/check_dependency_policy.py
```

Preserve the accepted plan, reject unsupported requirements before numerical
execution, and report any authorized fallback. For training changes, verify
parameter updates and recovery equivalence as well as forward values.

[Execution and precision details](IMPLEMENTATION.md) cover dynamic feedback,
precision admission, and internal compatibility paths.

## One sampled Ising trajectory

Native 1.3 requires explicit `ExecutionOptions(shots=1, seed=...)` on CPU.
`_ising_plan.py` composes the existing terminal probability checks and private
CPU draw with Simulation's prefix and selected-branch continuation. It returns
one conditional density and existing expectation records, then appends one
`mid_circuit_measurement` record with current pair IDs and original branch
probabilities. It does not create final samples, classical bits or feedback.
The existing environment protocol 1.1 binds `rng_torch_version`; reading or
executing on a different Torch build fails before numerical work or a draw.
Old plans retain their exact 1.0 environment mappings and byte captures.

```python
import torch
import flagquantum as fq
from flagquantum.experimental.simulation import IsingBraidProgram

source = IsingBraidProgram().exchange(1).measure_pair().exchange(0)
plan = fq.plan(source, options=fq.ExecutionOptions(shots=1, seed=17),
               outputs=fq.expectation(source.projector("psi"), name="psi"))
result = fq.run(fq.ExecutionPlan.from_json(plan.to_json()))
index = int(result.measurement("mid_circuit_measurement").value[0, 0])
print(plan.schema_version)
print(abs(float(result.expectation("psi")[0]) - index) < 1e-12)
print(torch.equal(result.state, fq.run(plan).state))
```

Output: `1.3`, `True`, `True`. The selected label itself is build-dependent.
Run `python -m pytest tests/unit/test_ising_sampled_contract.py tests/unit/test_ising_sampled_runtime.py tests/integration/test_ising_sampled_execution.py -q`.
The 216-byte numeric input/probability/sample subtotal is not a process-RSS bound;
state storage remains 64 bytes. This single fixed-sector CPU path adds no
multi-trajectory batching, gradient, feedback, other sector or hardware claim.


One native Rydberg presence trajectory is planned by `flagquantum.lindblad.plan`
with explicit `ExecutionOptions(shots=1, seed=...)` and run by `fq.run(plan)`.
Runtime owns 1.8 options/environment admission, private drawing and canonical
ExecutionResult/MeasurementResult assembly. Simulation retains numerical work.
CPU complex128, world-size one, one/two actors and no gradients are supported;
old Lindblad profiles remain outside root execution. See
[the approved delivery record](../../docs/api-changes/FQ-RYDBERG-PRESENCE-TRAJECTORY-20261025.md)
for a complete example and axes. Run the presence contract, compatibility and
runtime unit tests when modifying this path.

## Experiment files

`fq.save(path, program, result=result)` and `fq.load(path)` persist CircuitIR and
recorded numeric results as `.fqx` experiments without executing a program.
`experiment.py` owns archive validation and result binding; `_experiment_codec.py`
owns bounded JSON and tensor encoding. Historical plans remain data; existing
live execution-plan validation is unchanged. See the [format reference](../../docs/reference/EXPERIMENT_FILES.md)
and [CPU example](../../examples/save_experiment.py).
