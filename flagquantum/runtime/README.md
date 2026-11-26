# Runtime

Training and fq.Module implementation live in `training/`; checkpoints in `checkpointing.py`; execution/trajectory results and count statistics in `results/`; profile definitions in `profiles/`; measured MPS profiling in `observability/`; evolution composition in `evolution/`. Remote deployment planning lives in `planner/deployment.py`. Per-trajectory seed ownership remains in `trajectories/rng.py`, where it belongs to execution lifecycle.

Turn a quantum program or accepted execution plan into an observable execution
attempt. The design goal is consistent program, gradient, and result semantics
across local, distributed, and remote resources, with a direct local fast path.

Runtime owns selection, execution lifecycle, coordination, recovery, and result
assembly. Compiler transforms programs; Simulation computes numerical updates;
Compute and Remote adapt resources through their contracts.

The approved [Module capture boundary](../../docs/development/MODULE_REUSABLE_EXECUTION_CONTRACT.md)
permits only `module.py` to call the Compiler-owned builder capture entry.
The replacement keeps one current capture and existing live binding maps.
JAX tracing consumes its canonical IR instead of invoking the original Python
builder again. Runtime retains lifecycle, consumed snapshots and constant guards;
Simulation continues to consume Core IR without Compiler artifact types.
`execute().runtime` reports `builder_lowering`, an exclusion reason when the
legacy JAX builder is needed, and `jax_kernel_build_count`. That count tracks
kernel objects, not individual lazy XLA compilations or derivative variants.

## Follow the local path

```text
fq.run(program) → plan → execute accepted plan → Simulation → ExecutionResult
```

`fq.run(plan)` must execute the supplied plan without planning or compiling it
again. The result retains that plan and reports the observed execution route.
`_run.py` composes admission, planning and accepted-plan execution; execution.py
retains numerical backend dispatch and re-exports the existing local run entry.

## External initial states

Runtime admits per-call initial_state separately from the Circuit definition.
The first slice supports ordinary static Circuit sources, exact unbatched CPU
PyTorch statevectors and probability outputs. `_initial_state.py` owns
admission, minimum memory preflight and one differentiable input copy;
`execution_plan_contract.py` owns the versioned input requirement. Compiler
uses its existing optimize=False route for arbitrary-input legality; Simulation
still computes through Circuit.state. No external amplitudes enter the IR or
plan, and existing default-zero plans keep their initialization assumptions.

```python
import torch
import flagquantum as fq

circuit = fq.Circuit(1, dtype=torch.complex128).x(0)
initial = torch.tensor([0, 1], dtype=torch.complex128)
result = fq.run(circuit, initial_state=initial)
torch.testing.assert_close(
    result.statevector(), torch.tensor([[1, 0]], dtype=torch.complex128)
)
restored = fq.ExecutionPlan.from_json(result.plan.to_json())
replayed = fq.run(restored, initial_state=initial)
assert replayed.plan.identity == result.plan.identity
```

Use a compatible new tensor on every replay. Missing input and supplying input
to a default-zero plan fail explicitly. Precision must match; no normalization,
coercion, backend substitution or input detachment is implicit. The memory
budget covers a minimum state-plus-input-copy subtotal, not kernel workspace,
autograd or process RSS. Dynamic/QEC external inputs, other representations,
accelerators, shots, noise and remote state loading remain unsupported.
Gate parameters remain part of the sealed program: after an optimizer changes
them, run the source Circuit to create a new plan. Replaying a live plan with
changed gate values fails before copying the input or executing kernels.

For FQX, save `result.plan` together with `result`; the record preserves the
input requirement and outputs, without separately storing the initial tensor.
A marked plan without a result is refused before writing. Loading never
executes and does not recover the original input or graph.

For a small input-admission change, run
`python -m pytest tests/api_contract/test_execution_initial_state.py -q`.
Its replacement scenarios run the same numerical consumer with the reviewed
fake and real input binder; public scenarios cover gradients, optimizer updates,
JSON/FQX replay, memory preflight and unsupported requests.

## Named runtime angles

`inputs` binds existing `fq.Parameter` names per attempt. The first slice accepts
finite Python floats or matching real scalar CPU tensors on ordinary unbatched
static Circuits. The plan retains the symbolic template and requirements, not
the caller's values or autograd graph. Compatible changes reuse that plan:

```python
import torch
import flagquantum as fq

program = fq.Circuit(1, dtype=torch.complex128).ry(0, fq.Parameter("theta"))
theta = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
first = fq.run(program, inputs={"theta": theta})
torch.autograd.backward(first.statevector()[0, 1].abs().square())
torch.testing.assert_close(theta.grad, torch.sin(theta.detach()) / 2)
second = fq.run(first.plan, inputs={"theta": 0.8})
```

Every call supplies all names. A marked plan without inputs fails; an existing
unmarked plan refuses new inputs. `fq.plan` remains unchanged. Initial quantum
states, register type declarations and runtime angle arguments have distinct
semantics: `inputs` and `initial_state` cannot be combined in this first slice.
Only exact CPU PyTorch statevector and probability outputs are admitted. There
is no silent approximation, precision conversion of input tensors, other-engine
fallback or remote binding. Optimization is disabled for this symbolic route.
State memory is preflighted, but the estimate does not bound autograd, temporary
kernel memory or process RSS.

`_runtime_inputs.py` owns admission and per-attempt binding. It reuses Core's
Parameter/ParameterExpression and Instruction validation, the existing option
resolver, planner and numerical engine. A simple binding repair should remain
in that file. Run `python -m pytest tests/api_contract/test_execution_runtime_inputs.py -q`;
the scenarios include independent analytic references, optimizer updates,
replacement of the review binder, JSON/FQX replay and failure before kernels.
Save `result.plan` with `result` for FQX; a marked plan-only save is refused.
Loading restores requirements and recorded outputs, never prior inputs or graphs.

## PyTorch training and named binding

`Module` and `run(inputs=...)` use `_bind_named_instructions` in
`builder_compilation.py` to materialize ordinary symbolic gate parameters.
Core owns expression evaluation and angle validation. Runtime admission stays
with each entry point: the narrower `run` profile does not restrict `Module`.
Circuit subclasses retain their existing `bind_parameters` overrides.

Callable Module builders keep their existing FX tensor graph and reusable
task-local slots. These preserve arbitrary supported Torch feature transforms,
batching, higher derivatives, and cache invalidation without expanding the
serializable ParameterExpression vocabulary. This is shared named binding,
not a common execution-plan or backend capability claim. Training remains
`prediction = model(x)`, loss, `backward()`, and `optimizer.step()`; model weights
remain registered PyTorch parameters, not a user-managed runtime input dictionary.

For the shortest hybrid training workflow, run `python examples/quick_start.py`.
For initialization, freezing, optimizer membership, and current checkpoint
behavior, read [Training parameters](../../docs/guides/TRAINING_PARAMETERS.md)
and run `python -m examples.local.train_parameters`. Qualified v3 checkpoints
restore freeze flags and verify optimizer names/order. `builder_compilation.py`
keeps the private source-identity codec beside reusable builder snapshots; only
Module reads the existing Compiler capture digest. Historical v1/v2 files
retain caller-owned flags. Run `python -m pytest
tests/unit/test_training_parameter_state.py -q` for the local codec workflow.
For a binding change, run
`python -m pytest tests/unit/test_module_runtime_binding.py -q`. The tests replace
the binder through the same public consumers and compare values, gradients,
Adam updates, and checkpoint recovery with independent analytic references.

## Native Ising execution

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
| PyTorch model and optimization loop | [module.py](module.py), [training](training/__init__.py) |

## Verify a change

From the repository root:

```bash
python -m pytest tests/integration/test_cpu_vertical_slice.py tests/team/runtime -q
python tools/architecture/check_architecture.py
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
