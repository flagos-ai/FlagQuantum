# Runtime planner

This directory is the authoritative home for execution-mode and backend
selection, resource estimates, topology-aware candidate construction, noisy
backend selection, calibration inputs, and the final Runtime selection result.

It does not transform programs, execute numerical kernels, manage provider
lifecycle, or own the stable serialized `ExecutionPlan`. The package entry
compiles through the existing Compiler boundary and assembles the existing plan
product from resolved Runtime policy. Plan types and serialization live in
`runtime/execution_plan.py` and `runtime/execution_plan_contract.py`.

## Ten-minute change path

- Change stable `fq.plan` orchestration in `__init__.py`.
- Change backend cost decisions in `backend_selection.py`.
- Change memory estimates and execution policy in `estimates.py` and
  `execution_policy.py`. A tensor-network estimate is the larger of the state a
  contraction materializes and its working set; the working set has one
  definition, `estimates.estimate_tensor_network_working_set_bytes`, shared by
  backend selection, the selection context, and the execution policy, and it is
  an uncalibrated width-based proxy rather than the executor's measured
  `peak_size`. That proxy is not an upper bound on the peak, so it admits runs
  that cannot fit; do not present it as a capacity guarantee. What does bound an
  admitted run is a declared `memory_limit_bytes`, which the tensor-network
  executor passes down as a hard peak budget for every contraction the run
  performs and which raises rather than being overrun.
- Change candidate evidence in `candidate_plans.py`, `candidates.py`, and
  `providers.py`.
- Change selection ranking in `selection_result.py`.
- Change noisy-backend decisions in `noise_selection.py` and their calibration
  schema in `noise_calibration.py`.
- Change final stable/noisy plan assembly in `__init__.py`.

Run the backend-selection, selection-context/result, execution-policy,
execution-plan, and CPU vertical-slice tests. An ordinary policy change should
remain in this directory.

## Initial-state admission

Circuit plans currently serialize operations and execution policy, not a live
`Circuit(inputs=...)` tensor. The shared planning entry therefore refuses every
explicit tensor input before compilation or representation selection. This
applies to `fq.plan(circuit)`, local `fq.run(circuit)` and `circuit.run()`; an
explicit zero tensor is also refused rather than treated as a default input.
Dynamic planning uses the same check. Default-zero execution and state
preparation expressed as gates, including in-program resets and QEC rounds,
retain their existing semantics and serializable plans.

For custom-state execution, use `fq.run(circuit, initial_state=initial)` with a
Circuit constructed without inputs. This bounded CPU PyTorch statevector route
uses `_initial_state.py`, the existing option resolver and plan_advanced with
optimize=False. Its effective plan carries a versioned input requirement, not
amplitudes. Existing `fq.Circuit(..., inputs=initial).state()` remains supported
with its autograd path. The guard does not make Circuit.to_ir carry a live
state tensor or admit external inputs to dynamic, remote or other engines.

`_validated_plan_program` owns this check for both static and dynamic requests.
Later representation and lowering decisions consume the admitted program; they
must not introduce their own interpretation of construction-time inputs.

`fq.plan` produces an executable `ExecutionPlan`; `fq.run(plan)` validates and
executes its accepted decision without replanning. `Circuit.runtime_plan()`
explains runtime selection through a separate result type. Neither entry makes
an existing zero-input plan reusable with arbitrary external states. Compiler
passes currently use the zero-input assumption, including removal of leading
resets in `compiler/optimization/zero_state_reset.py`.

Run the following for admission, gradient preservation and gate-preparation
JSON replay checks:

```bash
python -m pytest tests/integration/test_custom_initial_state_admission.py \
  tests/api_contract/test_dynamic_execution_planning.py -q
```
