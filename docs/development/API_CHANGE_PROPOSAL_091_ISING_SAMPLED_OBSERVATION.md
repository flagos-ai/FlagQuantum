# Proposal 091: one sampled Ising pair-charge trajectory

Status: **approved for staged implementation on 2026-10-24**.
The user explicitly approved all five decisions and conditionally authorized
merging the document PR #818. That PR merged after complete applicable CI/GPU.
Proceed first with the replaceable baseline; each later PR still needs its own
merge authorization. This approval record does not activate a production API.

## Five decisions for review

1. Add `IsingBraidProgram.measure_pair(self) -> IsingBraidProgram`: one ideal
   sampled QND observation of current positions 0/1, followed by the remaining
   declared exchanges. Reuse `Operation("ising.measure_pair", physical_data=None)`.
   Permit one observation total: either this operation or `dephase_pair()`, not
   both. The builder remains fluent and returns no symbolic classical bit.
2. The initial execution profile is **one conditional trajectory**: existing
   ExecutionOptions must explicitly provide `shots=1` and an integer seed in
   `[0, 2**64-1]`. Reuse Runtime's private CPU generator/draw authority and
   Simulation's ordered density/Kraus actions. No second runner, controller,
   instrument class, registry, feedback loop or training surface is added.
3. Only the sampled operation selects native source/plan **1.3**. Preserve exact
   braid-only 1.1, unread 1.2 and all other old artifacts. Add the sole new
   native-1.3 environment requirement `rng_torch_version` under the existing
   execution_environment protocol version 1.1, bound to the exact
   `str(torch.__version__)`; reject a mismatch before execution. Seed/shots stay
   in existing options, not a second sampling payload.
4. Return the existing ExecutionResult with a **conditional** `(1,2,2)` CPU
   complex128 state and conditional requested expectations. Append one existing
   MeasurementResult of kind `mid_circuit_measurement`, empty qubit tuple and
   `(1,1)` int64 outcome index. Do not fabricate final samples, a bit register,
   an event timestamp or feedback. Retain existing numeric resource preflight.
5. After explicit approval, establish a replaceable contract baseline and old
   captures before domain-owned implementations. Every subsequent PR needs its
   own merge authorization and complete latest applicable CI/GPU. Approval of
   this proposal is not a blanket merge grant.

## Why one trajectory first

[089](API_CHANGE_PROPOSAL_089_ISING_UNREAD_OBSERVATION.md) implements an unread
channel: its state includes both alternatives and it selects no outcome. A
sampled trajectory must instead normalize one nonzero branch, continue from
that conditional state and expose the actual observed label. Returning an
unconditional mixture beside a claimed selected trajectory would be misleading.

One trajectory preserves the existing leading state axis and requested output
axes. It avoids prematurely choosing a multi-shot density axis, empirical versus
exact ensemble state, selected-branch gradients or feedback schema. Multi-shot
native execution is explicitly deferred, not silently implemented by repeated
unseeded calls. Callers may construct separate existing plans with distinct
explicit seeds when they need independent trajectories; that is not a throughput
or batching claim.

Existing qubit [dynamic execution](../../flagquantum/runtime/dynamic/execution.py)
already has statevector trajectories, classical bits and feedback. Its
[canonical result projection](../../flagquantum/runtime/dynamic/result.py)
reuses ExecutionResult/MeasurementResult. Fusion coordinates are not qubits, so
this proposal neither feeds a fusion density to that qubit runner nor creates a
parallel DynamicExecutionResult. It reuses the native executor and the same
canonical measurement-kind vocabulary without adding a classical register.
Concurrent [post-routing dynamic work](API_CHANGE_PROPOSAL_089_DYNAMIC_POST_ROUTING_CAPABILITY_REMATCH.md)
and [Quafu integration](API_CHANGE_PROPOSAL_090_QUAFU_DYNAMIC_VERTICAL_SLICE.md)
retain their separate scope. This proposal makes no provider capability claim.

## Proposed user workflow (unavailable before implementation)

```python
import torch
import flagquantum as fq
from flagquantum.experimental.simulation import IsingBraidProgram

program = IsingBraidProgram().exchange(1).measure_pair().exchange(0)
plan = fq.plan(
    program,
    options=fq.ExecutionOptions(shots=1, seed=17),
    outputs=fq.expectation(program.projector("psi"), name="psi"),
)
result = fq.run(fq.ExecutionPlan.from_json(plan.to_json()))
observed = result.measurement("mid_circuit_measurement")
index = int(observed.value[0, 0])
print(plan.schema_version)
print(observed.metadata["outcome_labels"][index] in ("1", "psi"))
print(abs(float(result.expectation("psi")[0]) - index) < 1e-12)
print(torch.equal(result.state, fq.run(plan).state))
```

Proposed output: `1.3`, `True`, `True`, `True`. No particular random label is
promised across PyTorch builds. Within the reviewed environment, repeating the
sealed seeded plan must return identical indices, conditional state and outputs.
The first exchange gives both labels nonzero probability; the final outer
exchange preserves their populations, so the conditional P(psi) equals the
selected index. The new method does not exist today.

## Exact source and physics semantics

Retain the fixed four sigma anyons, total vacuum, left-associated tree, Nayak
gauge and two fusion coordinates of 085/089. Track current anyon positions under
existing exchanges: observing positions 0/1 is not observing the initially named
pair after it has moved. Keep the original source/context immutable.

At the declared operation boundary, use the existing fixed projectors:

```text
P1 = diag(1,0), Ppsi = diag(0,1)
Jj(rho) = Pj rho Pj
pj = real(trace(Jj(rho)))
j = one private seeded draw from the checked branch probabilities
rho_conditioned = Jj(rho) / pj, with pj > 0
rho_final = the remaining ordered braid actions applied to rho_conditioned
```

The corresponding unread continuation is the weighted sum of the conditional
continuations. A zero-weight branch contributes zero to that sum and must never
be divided by zero. If an invalid selected index or nonpositive selected weight
reaches conditional continuation, raise ExecutionError without returning a
partial result. There is no public forced-outcome/postselection argument.

The operation's operands, results, parameters and conditions are exact empty
tuples; literal and physical_data are literal None. No arbitrary Kraus argument,
callback, name, outcome, classical-bit destination or timing parameter is added.
Supplying arguments raises TypeError. A second sampled or unread observation,
in either insertion order, raises ValueError atomically. Existing programs that
do not contain the new operation keep their existing behavior.

## Execution, RNG and environment

Planning and JSON reading perform no draw and consume no Torch, CUDA or Python
RNG state. During execution Runtime creates a private CPU generator with the
explicit seed and reuses the existing probability/draw machinery in
[_terminal_instruments.py](../../flagquantum/runtime/_terminal_instruments.py)
and [measurements.py](../../flagquantum/runtime/measurements.py). No call to global
manual_seed is permitted. Simulation owns selected Kraus application,
normalization and suffix evolution through the existing ordered action kernel.
The old terminal sampler's signatures, metadata and seeded behavior stay intact;
its terminal-instrument record must not be relabelled as a native source event.

Refactor private helpers only where both current callers actually need them.
Do not add a new public instrument record, RNG service or numerical engine.
Probability-input validity uses the existing complex128 terminal checks, including
its 1e-10 validity tolerance. That is not the physics error budget: independent
small-word density/probability/weighted-continuation oracles retain 1e-12, rtol=0.
Record the existing probability correction when roundoff requires draw-weight
normalization; normalize the selected physical branch by its original positive
trace, not by its corrected draw weight.

Native plan 1.3 includes `environment_requirements.rng_torch_version` as the exact
PyTorch version string. Only this profile selects version 1.1 of the existing
flagquantum.execution_environment protocol; its other fields keep their meaning.
This is a versioned extension of the existing mapping, not a second environment
type or registry. Old native/circuit plans retain environment protocol 1.0 and
its exact bytes; native 1.1/1.2 reject the extra RNG requirement or protocol 1.1. Readers and preflight refuse a 1.3 RNG
version mismatch with the existing environment_incompatible reason before kernel
or draw. Same-version execution on arbitrary hosts is not a cross-platform
bitwise reproducibility promise. Source, output, options and environment remain
sealed by the existing fingerprints and native identity authority.

## Result contract and selection

Keep all public signatures, dataclass fields and result schema versions. In this
new native profile:

| Existing result member | Meaning |
| --- | --- |
| `state` | Conditional final density, shape `(1,2,2)`, CPU complex128, unit trace. |
| `value` and expectation measurements | Exact conditional expectations, existing `(1,O)`/`(1,)` axes and output ordering. |
| `samples` | None: there was no automatically requested terminal measurement. |
| `measurements` | Existing requested expectation records first; one native intermediate record appended last. |
| `plan` | The accepted native 1.3 plan, with `analysis=None`. |

The intermediate record has kind `mid_circuit_measurement`, `qubits=()`,
`shots=1`, value of shape `(1,1)` and dtype int64. Its metadata is:

- `stage`: `intermediate`;
- `outcome_labels`: ordered `["1", "psi"]`;
- `probabilities`: original pre-observation weights shaped `(1,2)` as JSON lists;
- `probability_correction`: the existing maximum draw-weight correction;
- `source_operation_index`: zero-based location in the captured source;
- `pair_anyon_ids`: the two IDs at current positions 0/1 at that location.

Runtime retains mode density_matrix and reuses existing `shots`, `seed` and
`mid_circuit_measurements_available` metadata keys from dynamic result vocabulary.
No native classical-bit array or feedback trace is created. The sample index is
an ordered charge label, not a qubit count or an inferred detected-loss flag.

Existing `measurement()` name-over-kind resolution is unchanged. The proposed
example uses a nonconflicting output name. If an expectation is deliberately
named `mid_circuit_measurement`, callers can use `result.measurement(-1)` to
select the documented appended observation. Do not reserve a previously valid
output name or change global selector semantics for this profile.

## Compatibility, resources and refusals

Native source/plan 1.3 is valid only with exactly one sampled operation. Native
1.1 remains braid-only; 1.2 remains exactly one unread operation. Strictly reject
relabelled sources, mismatched source/plan versions, duplicate/mixed observations,
extra fields/data, SSA operands/results/conditions and rehashed inconsistencies.
The global circuit EXECUTION_PLAN_VERSION remains 1.0. Lindblad plan 1.3 belongs
to its own schema and is unchanged. Preserve old source/plan JSON bytes and IDs;
old readers refuse new native 1.3 rather than migrating it implicitly.

Retain CPU complex128, batch_size=1, world_size=1, exact density mode and existing
state/expectation targets. Reject shots=None or shots other than 1, seed=None or
seed outside the stated range, gradients, noise, alternate backends/devices,
approximation/fallback, terminal-samples target, cost calibration and Remote.
Global ExecutionOptions validation still owns malformed option types; valid
options outside this sampled profile fail with CapabilityError before numerical
work. A saved plan's overrides remain forbidden.

State storage remains 64 bytes. Reuse the terminal sampler's numeric preflight:
for one density, two fixed projectors, two probabilities and one index its
accounted input/probability/sample subtotal is `64 + 128 + 16 + 8 = 216` bytes.
A non-None memory_limit_bytes below 216 must fail before prefix evolution; the
existing dense-matrix ceiling and output checks also apply. This subtotal is not
a process-RSS or temporary-workspace bound. Do not change the old state-only
64-byte admission for 1.1/1.2 or report fusion coordinates as qubits.

## Acceptance and domain delivery

1. Before implementation, capture exact old circuit/native 1.1/1.2 and
   Rydberg/Lindblad bytes, IDs, result axes and seeded terminal behavior. Establish
   replaceable source/planning/execution/JSON consumers; the reference must not
   fake a production native 1.3 plan or claim the new method exists.
2. Use independent fixed F/R and projectors for both initial channels, zero-weight
   branches, noncommuting prefixes/suffixes and current-pair identity. Compare
   complete complex conditional densities, probabilities, trace and weighted
   unread equality at atol=1e-12, rtol=0. Validate actual conditional expectation
   values, not only a sampled label histogram.
3. Check same-seed direct/planned/JSON exact replay, plan identity changing with
   seed or source order, frozen outputs/source, and unchanged Torch/Python/CUDA
   RNG. Use eigenstate certainty and a predeclared shared-sampler witness:
   4,096 equal-weight draws with private seed 17 must put the psi count within
   192 of 2,048 (six ideal-binomial standard deviations). This tests the existing
   draw authority, not multi-shot native admission. Distinct-seed single-trajectory
   consumers separately verify selected-state consistency. Do not hardcode one
   PyTorch-specific label sequence or widen a bound to hide a numerical defect.
4. Prove source/plan and RNG-environment mismatches fail after rehashing, old
   readers reject 1.3, all existing options/device/resource refusals remain, the
   215/216-byte admission boundary follows the shared numeric accounting, and
   canonical dynamic/circuit/terminal consumers remain unchanged.
5. After the baseline is accepted, split Core source/codec, Simulation ordered
   conditional math, and Runtime admission/draw/projection deliveries. Replace
   only factories and delete obsolete fakes; keep independent consumers fixed.
   Run supported-Python focused tests, required default/integration and strict
   API/architecture checks, then complete latest CI/GPU before each separately
   authorized merge. No public snapshot or candidate contract changes occur
   before explicit approval.

## Runnable current-API reference

This caller-composed terminal draw followed by explicit conditional mathematics
runs today. It proves reuse and the physical distinction; it is not native
sampled source support, plan 1.3, a contract baseline or feedback execution.

```python
import math
import random
import torch
import flagquantum as fq
from flagquantum.experimental.simulation import IsingBraidProgram
from flagquantum.operators import SuperOperator
from flagquantum.runtime import sample_terminal

program = IsingBraidProgram().exchange(1)
rho = fq.run(program).state
projectors = [program.projector(c) for c in ("1", "psi")]
instrument = {c: [p] for c, p in zip(("1", "psi"), projectors)}
torch_before, python_before = torch.random.get_rng_state().clone(), random.getstate()
observed = sample_terminal(rho, instrument, shots=1, seed=17)
index = int(observed.value[0, 0])
branches = [SuperOperator.from_kraus([p]).apply(rho) for p in projectors]
weights = [float(torch.trace(branch[0]).real) for branch in branches]
assert weights[index] > 0
phases = torch.exp(torch.tensor([-1j, 3j], dtype=torch.complex128) * math.pi / 8)
continuation = SuperOperator.left_right_multiply(torch.diag(phases), torch.diag(phases).mH)
conditional = continuation.apply(branches[index] / weights[index])
average = sum((continuation.apply(branch) for branch in branches), torch.zeros_like(rho))
unread = fq.run(IsingBraidProgram().exchange(1).dephase_pair().exchange(0)).state
torch.testing.assert_close(average, unread, atol=1e-12, rtol=0)
repeat = sample_terminal(rho, instrument, shots=1, seed=17)
print(round(float(average[0, 1, 1].real), 12))
print(abs(float(conditional[0, 1, 1].real) - index) < 1e-12)
print(torch.equal(observed.value, repeat.value))
print(torch.equal(torch_before, torch.random.get_rng_state()) and python_before == random.getstate())
```

Executed output: `0.5`, `True`, `True`, `True`. The first value is the weighted
unread P(psi); each selected trajectory instead has its conditional value 0 or 1.
The last two checks show private-seed repetition and unchanged caller RNG.

This proposal does not authorize multiple native measurements, sampled feedback,
atom-loss inference, general instruments, multiple trajectories per run,
continuous gradients, hardware execution or a complete M3/M4 claim.
