# API Change Proposal 084: Unified dynamic execution through `fq.run`

## Status

**Stable promotion completed on 2026-10-23. The local delivery sequence and all
acceptance criteria are complete, and the API owner authorized the exact scope
with `approve DYNAMIC-EXECUTION-UNIFIED-RUN-STABLE-PROMOTION`.**

The original implementation authorization was:

```text
approve DYNAMIC-EXECUTION-UNIFIED-RUN-CONTRACT
```

The stable promotion changes no public runtime behavior. It retires the
candidate contract after the separately reviewed contract, planning, execution,
and performance stages completed. The root export set and every existing public
signature remain unchanged.

The approved stable construction contract remains
[`contracts/dynamic-circuit-v1.json`](../../contracts/dynamic-circuit-v1.json).
The stable execution contract is
[`contracts/dynamic-execution-unified-run-v1.json`](../../contracts/dynamic-execution-unified-run-v1.json).
It supersedes and retires the candidate contract without requiring a user-code
change. This proposal covers execution only. It does not merge `DynamicCircuit`
into `Circuit`, promote provider integrations, or expose the native dynamic
result.

## Problem

Users can construct a stable dynamic program but must leave the stable execution
model to run it:

```python
import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit

circuit = DynamicCircuit(2).h(0).measure(0, classical_bit=0)
circuit.conditional("x", 1, classical_bit=0)

native = fq.experimental.dynamic.run_dynamic(circuit, shots=128, seed=7)
result = native.to_execution_result()
```

This path has three product problems:

1. the stable builder does not use the stable `fq.run` entry;
2. callers must know an experimental result type and perform a projection;
3. `fq.plan(circuit)` and `fq.run(fq.plan(circuit))` reject the same program, so
   simply special-casing direct `fq.run` would split the execution model.

The existing pieces are sufficient for a narrow convergence. `DynamicCircuit`
already lowers to CircuitIR v1, dynamic instructions already round-trip, the
local executor already produces deterministic shot records, and
`DynamicExecutionResult.to_execution_result()` already defines a candidate
projection into `fq.ExecutionResult`. The missing work is one reviewed stable
contract across planning, execution, result normalization, and failure behavior.

## Decision

### 1. One stable execution entry

After implementation, the shortest supported local workflow is:

```python
import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit

circuit = DynamicCircuit(2).h(0).measure(0, classical_bit=0)
circuit.conditional("x", 1, classical_bit=0)

result = fq.run(circuit, shots=128)
assert isinstance(result, fq.ExecutionResult)
assert result.samples is not None
assert result.samples.shape == (128, 2)
```

`fq.run` gains no parameter. `DynamicCircuit` is already a `Circuit` subtype, so
the public function signature and root export set remain unchanged. The change is
an additive stable behavior: a dynamic CircuitIR is no longer categorically
refused when it satisfies this contract.

`fq.experimental.dynamic.run_dynamic` remains experimental and callable during
the migration period. It is not re-exported, renamed, or declared stable by this
proposal.

### 2. Planning precedes stable execution

Direct and explicit planning must be equivalent before the behavior is exposed:

```python
options = fq.ExecutionOptions(shots=128, seed=7)
plan = fq.plan(circuit, options=options)

automatic = fq.run(circuit, options=options)
explicit = fq.run(plan)

assert automatic.plan is not None
assert explicit.plan is not None
assert automatic.plan.identity == explicit.plan.identity == plan.identity
```

The existing `ExecutionPlan` and serialized schema `1.0` are reused. No dynamic
plan class, scheduler object, public strategy enum, or plan schema version is
added. The plan's existing program payload carries the dynamic instructions and
its existing resolved-options payload carries shots and seed. Planning preserves
those instructions rather than canonicalizing them into a static program.

The plan records a single-process local trajectory decision and remains an input
to `fq.run`. A stale, edited, mismatched, or environment-incompatible plan fails
closed under the existing plan identity and preflight rules; it is never silently
replanned.

### 3. Exact option boundary for the first stable slice

Dynamic local execution requires one positive shot count. It may be supplied
either as `fq.run(..., shots=N)` or as `ExecutionOptions(shots=N)`, using the
existing duplicate-input `TypeError` when both are supplied. Planning requires
`ExecutionOptions(shots=N)` because `fq.plan` has no direct `shots` keyword.

`ExecutionOptions.seed` is the stable seed input. An identical program, effective
plan, seed, supported Torch version, CPU runtime, and noise model must reproduce
the same samples. No cross-version or cross-device bitwise guarantee is made.

The first stable slice accepts:

- `mode` unset or `"auto"`;
- `backend` unset, `"auto"`, or `"pytorch"`;
- `device` unset, `"auto"`, or a CPU device;
- `target` unset, `"auto"`, or `"samples"`;
- `shots`, `seed`, and `memory_limit_bytes` under their existing validation;
- `require_gradients` unset or false;
- `allow_approximate` and `allow_backend_fallback` unset or false.

`batch_size` and `precision` must be absent because those properties are owned by
the already constructed `DynamicCircuit` in this slice. A conflicting or
unsupported option raises `ValidationError` or `CapabilityError` before execution
with a reason naming the field. The stable entry never substitutes another
backend, drops classical control, or converts the program to a static circuit.

The existing `noise_model` argument is accepted only for the dynamic noise subset
already documented and tested by the local executor. `outputs` must be `None`:
mid-circuit measurements and final sampling are defined by the dynamic program
and this first slice does not compose an independent output graph with them.
`compiler`, remote `target`, `target_qubits`, and `name` remain outside this local
contract and fail before provider submission.

### 4. Canonical result projection

The stable return type is exactly `fq.ExecutionResult`. The projection has these
semantics:

- `samples` is the final computational-basis sample tensor with shape
  `(shots, n_qubits)` for an unbatched program;
- `measurements[0]` is a `MeasurementResult(kind="sample")` for those final
  samples;
- when available, `measurements[1]` is
  `MeasurementResult(kind="mid_circuit_measurement")` containing the classical
  register in ascending classical-bit order;
- both measurement records carry the effective positive shot count;
- `state` and `value` are `None`; per-shot final trajectory states are not one
  canonical statevector and do not become a stable result field;
- `plan` is the exact effective `ExecutionPlan`, including for direct automatic
  execution;
- `runtime` reports mode, shots, seed, and whether mid-circuit measurements are
  available;
- local execution does not invent provider provenance.

`DynamicExecutionResult`, `final_states`, `feedback_traces`, provider-specific
metadata, and executor strategy remain outside the stable contract. Internal
execution may continue to use that result and its projection, but the stable path
must not expose it by type, attribute, or import.

### 5. Failure occurs at the earliest knowable boundary

| Condition | Stable behavior |
| --- | --- |
| Missing, duplicated, non-integer, boolean, or non-positive shots | Existing `TypeError` or `ValidationError` before planning/execution |
| Condition reads a bit not written by an earlier measurement | `ValidationError` during construction or plan validation |
| Gradients requested or parameters require gradients | `CapabilityError` before execution |
| Distributed world size is greater than one | `CapabilityError` with `dynamic_conditional_execution_unsupported` during planning |
| Unsupported dynamic noise or option | Typed refusal naming the unsupported capability |
| Dynamic plan identity/program/environment mismatch | Existing plan preflight error; no replanning |
| Remote target lacks verified dynamic capabilities | Refused before submission; remote support is a later proposal |

An error must not consume RNG state, submit a task, run a static substitute, or
return a partial `ExecutionResult`.

## Compatibility

This is additive behavior on the already stable `fq.run`, `fq.plan`,
`ExecutionOptions`, `ExecutionPlan`, and `ExecutionResult` surfaces. It changes no
signature, field order, default, root export, CircuitIR schema, ExecutionPlan
schema, or current static-circuit behavior.

Existing experimental callers continue to work:

```python
native = fq.experimental.dynamic.run_dynamic(circuit, shots=128, seed=7)
```

No deprecation is introduced in this proposal. A later proposal may deprecate
direct user access to `run_dynamic` only after the stable path has shipped,
provider and diagnostic consumers have a replacement, and a documented release
window has elapsed.

`DynamicCircuit` and `Circuit` remain distinct construction entries. Any future
class convergence requires its own compatibility analysis and approval.

## Delivery sequence

1. **Contract and conformance baseline.** After explicit approval, add a
   machine-readable additive behavior contract and failing conformance tests for
   automatic/explicit plan identity, result projection, option precedence, and
   fail-closed behavior. No runtime implementation in this step.
2. **Dynamic planning.** Reuse ExecutionPlan 1.0, preserve dynamic instructions,
   record the trajectory decision, enforce the option/distribution boundary, and
   prove serialize/restore/run equivalence.
3. **Local `fq.run` activation.** Route both direct programs and restored plans
   through the same internal execution boundary, attach the effective plan, and
   return only `ExecutionResult`.
4. **Performance and non-regression.** Measure planning overhead, trajectory and
   batched execution, memory-bound strategy selection, and unchanged static fast
   paths. Performance evidence changes no stable result or option semantics.
5. **Remote capability proposal.** Add provider capability negotiation and
   deployment only after the local contract is complete. No real-QPU support is
   inferred from local or adapter conformance.

Each implementation step is a separately reviewed PR with a runnable user-facing
example. Protected-contract changes land before Runtime implementation, following
the repository's integration workflow.

### Local performance budget

Stage 4 uses the maintained `dynamic_unified_run` benchmark. It separately
measures planning, execution of an existing plan, automatic planning plus
execution, the existing native dynamic oracle, both memory-selected local
trajectory strategies, and a static sampling sentinel. The gate requires:

- planning p95 no greater than 50 ms;
- stable explicit-plan median no greater than 3.0 times the native oracle;
- automatic plan-and-run median no greater than 2.0 times explicit-plan run;
- static automatic median no greater than 2.0 times static explicit-plan run.

These deliberately bounded wrapper budgets detect routing and serialization
regressions without claiming kernel parity. The payload is classified as local
single-process evidence and explicitly forbids distributed, accelerator, QPU,
and release-performance claims. Reproduce it with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
flagquantum-benchmark run dynamic_unified_run \
  --batched-shots 4096 --trajectory-shots 64 --static-shots 4096 \
  --warmup 2 --iterations 9 --planning-iterations 25 \
  --json-output \
  benchmarks/results/comparison/dynamic_unified_run_cpu_arm64_20261023.json
```

The recorded macOS arm64 observation used Python 3.12.14, PyTorch 2.14.1,
FlagQuantum 0.3.1, one process, and one CPU thread. Planning p95 was 0.347 ms.
Stable explicit-plan/native median ratios were 1.123 for the batched strategy and
1.099 for the memory-forced trajectory strategy. Automatic/explicit ratios were
0.982 and 0.988, and the static sentinel ratio was 1.441. All four budgets
passed. These values characterize that observation only; the versioned runner
and budget verdict, rather than the numbers themselves, are the regression
mechanism.

## Promotion record

| Stage | Merged evidence | Outcome |
| --- | --- | --- |
| Contract and conformance | PR #733 | Protected candidate contract and failing-first semantic checks established. |
| Dynamic planning | PR #734 | Dynamic instructions and plan identity survive ExecutionPlan 1.0 serialization. |
| Local stable execution | PR #752 | Direct and explicit-plan execution converge on `fq.ExecutionResult`. |
| Performance and non-regression | PR #754 | Local overhead budgets, both strategy paths, correctness, and the static sentinel pass. |

The promotion itself changes only contract status and documentation. It does not
modify Runtime code or infer remote, accelerator, distributed, or QPU support.

## Acceptance

- [x] API owner approves the exact decision with the recorded directive.
- [x] A target-branch-protected machine contract describes the additive behavior
      without replacing existing stable snapshots merely to pass CI.
- [x] `fq.run(circuit, options=options)` and
      `fq.run(fq.plan(circuit, options=options))` return results carrying the same
      plan identity.
- [x] Dynamic plan JSON round-trips without losing measurement, reset, conditions,
      instruction order, shots, seed, or plan identity.
- [x] The stable path returns `fq.ExecutionResult` and exposes no native dynamic
      result type.
- [x] Result samples and classical registers match the existing seeded dynamic
      executor on trajectory and batched paths.
- [x] Invalid def-use, gradients, distributed execution, unsupported noise,
      unsupported options, and incompatible targets fail before execution or
      submission with typed diagnostics.
- [x] Static circuit planning and execution retain their current signatures,
      results, identities, and measured fast-path performance.
- [x] CircuitIR `1.0`, ExecutionPlan `1.0`, the root export set, and existing
      `ExecutionResult` fields remain unchanged.
- [x] Required unit, integration, distributed-CPU refusal, API-contract,
      serialization, documentation, package, and supported Python checks pass.

## Non-goals

- Merging `DynamicCircuit` into `Circuit`.
- Stabilizing or exporting `run_dynamic` or `DynamicExecutionResult`.
- Adding a new result, plan, condition-expression, scheduler, strategy, or public
  metadata type.
- Supporting gradients through sampled classical feedback.
- Claiming distributed dynamic execution, GPU parity, QPU support, or provider
  equivalence.
- Silently flattening, approximating, rerouting, or statically substituting a
  dynamic program.
