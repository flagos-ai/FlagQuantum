# Proposal 092: one seeded native Rydberg presence trajectory

Status: **approved for staged implementation on 2026-10-25**.
The user explicitly approved all five decisions and conditionally authorized
normal merging of document PR #845 after its latest complete applicable CI/GPU
and repository rules pass. Proceed first with the replaceable baseline; each
later PR still requires its own merge authorization. This approval record does
not activate a production API.

This document changes no API, source/plan reader, candidate contract, snapshot,
capability level or production code. Approval of 081/083/089/091 and merges of
existing examples did not approve the changes below; this proposal received its
own explicit approval. Each later PR requires its own merge authorization and
latest complete applicable CI/GPU verification.

## Five approved decisions

1. Add `RydbergProgram.measure_presence(self, atom: str, *, at: float,
   efficiency: float) -> RydbergProgram`: one declared observation of one named
   atom inside the physical schedule. Extend the existing ProgramRecord
   operation/payload family, not the IR hierarchy or a detector registry.
2. Admit one CPU complex128 conditional trajectory, explicit `shots=1` and a
   Python integer seed. Retain the full native schedule, time grid and interval
   bindings; reuse existing Liouvillian assembly, integrators and Kraus actions
   on either side of the observation. Refuse gradients through this profile.
3. Use source 1.4 only for this operation and Lindblad plan 1.8 only for this
   sampled profile. Capture the existing ExecutionOptions representation and
   execution-environment protocol 1.1 with `rng_torch_version`. Preserve exact
   old source/plan bytes and strict source/version/environment admission.
4. Extend the existing root `fq.run` input union with LindbladPlan, admitting
   **only** this new 1.8 profile through that branch. Return the existing
   ExecutionResult/MeasurementResult with explicit final-state/time-output axes.
   Extend only ExecutionResult.plan's annotation to include LindbladPlan;
   add no result fields/classes and do not make Simulation depend on Runtime.
   Existing `fql.run` retains EvolutionResult and refuses sampled 1.8 with an
   instruction to use `fq.run(plan)`.
5. Establish a replaceable contract baseline after approval, then deliver
   Core, Simulation and Runtime-owned changes separately. Preserve the
   independent physical consumers during replacement; reject unsupported
   profiles before numerical execution/drawing. Approval is not a blanket
   merge grant.

## Existing owners and the result boundary

| Existing owner | Reuse and constraint |
| --- | --- |
| Core RydbergProgram and ProgramRecord | Named atom/site roster, timestamped controls, strict native source codec and validation. Append order is not physical time. |
| Simulation `rydberg_model.native_model` | Assemble all interval Hamiltonians and loss jumps once from the full grid and original bindings. Its existing grid starts at zero and covers every control boundary. |
| Simulation Liouvillian and `advance_interval` | Advance the prefix and suffix with the selected existing integrator and unchanged convergence checks. No new integrator, generator or Hamiltonian builder. |
| Operators SuperOperator | Apply the declared trace-decreasing Kraus branch; Simulation validates and normalizes its positive original trace. |
| LindbladPlan and native-plan validator | Seal the full source, matrices, bindings and numerical decisions in one existing plan type; no nested ExecutionPlan or second identity. |
| Runtime probability/draw and environment authorities | Reuse terminal-instrument probability checking, private CPU generator/draw, and the exact RNG-build gate already used by 091. |
| Runtime ExecutionResult/MeasurementResult | Retain canonical result ownership and all fields. Simulation returns numerical arrays, never Runtime records. |

The current EvolutionResult is defined in Simulation. Adding Runtime
MeasurementResult fields there would invert the existing dependency boundary;
a second trajectory result class would duplicate a canonical result. This
proposal instead joins an existing dynamics plan to the existing root executor.
It adds neither a new runner entry nor a result envelope. Ordinary circuit
imports/execution must retain their lazy CPU fast path without eagerly importing
Lindblad planning, initializing distributed execution or querying a provider.

This is a protected root-input/plan/result-annotation change and needs explicit
review. It is not covered by approval of Ising sampling. The root planner does
not gain time/initial-state arguments: the existing dynamics planner already
owns them. No contract type is added to stand between its plan and Runtime.

## Source operation and physical ordering

The proposed fluent method has no defaults for detector efficiency or time:

```python
# Proposed only; absent from the current implementation.
def measure_presence(
    self, atom: str, *, at: float, efficiency: float
) -> RydbergProgram: ...
```

The atom must be one declared atom ID, not a qubit index or inferred site.
Time and efficiency must be finite real scalars, excluding bool and trainable
parameters; efficiency is in `[0, 1]`. One observation total is allowed.
Reuse one `Operation("rydberg.measure_presence", physical_data=...)` with a
frozen payload containing exactly atom, at and efficiency in the existing
physical-payload family. Existing pulse/transport records cannot encode these
three meanings without inventing fields. Core construction performs the same
immediate leaf versus snapshot/full-history checks as existing Rydberg controls.

At snapshot/planning, require `0 < at < control_horizon` and exactly one distinct
matching internal output-grid point under the existing time-precision rules.
Do not insert a grid point, shift a control or restart native time at zero for
the suffix. A grid must still contain all ordinary physical control boundaries.
At a shared control boundary, evolve the preceding interval, apply the
instrument, then evolve the following interval. The grid row at the observation
stores the **post-observation** state; earlier rows remain forward pre-observation
states, not a retrodicted trajectory conditioned on future information.

An actor's ID and roster/basis position survive transport and observation.
A commanded path is not evidence of trap activity, actual position or physical
survival. The first profile retains one/two actors and the current ordered
`g/r/lost` product basis; it does not create an explicit-basis or other-species
instrument API.

## Instrument, loss and continuation

For the selected actor in local `g,r,lost` order, use exactly:

- `K_no_flag = diag(1, 1, sqrt(1 - efficiency))`;
- `K_flag = diag(0, 0, sqrt(efficiency))`.

Embed those operators in the declared atom roster; other factors are identity.
Ordered labels are `no_flag`, then `flag`. Efficiency means detection of the
existing lost sector, with zero false positives. It is not a physical survival
probability. A no-flag outcome includes undetected loss when efficiency is below
one; it must never be labelled an all-present result or a detected erasure.

For branch j, reuse `J_j(rho) = K_j rho K_j^dagger` and original
`p_j = real(trace(J_j(rho)))`. Check completeness/density/probabilities with the
existing complex128 validation budget. Reuse the established correction of
roundoff-sized probabilities only for drawing; normalize the selected state
by its **original positive trace**, not the corrected draw probability. Reject
a selected zero-probability branch and preserve the unobserved lost sector.
Efficiency zero is legal: its flag branch contributes zero, never `0/0`.

The deterministic weighted sum of both normalized continuations must reproduce
the unread instrument followed by the same suffix. An unread channel must not
be reported as the selected trajectory, and finite-shot frequencies cannot
replace this deterministic physical check. Observation is an instantaneous
ideal QND instrument on this finite model, not a detector-latency or event-stream
simulation. Existing Markov loss continues before and after the boundary.

## Plans, controls and fail-closed behavior

Only sources containing the new operation use native Program IR 1.4. Old
Rydberg controls stay 1.0; Ising 1.1/1.2/1.3 keep their exact meanings/bytes.
Only the new sampled native Lindblad plan uses 1.8. Existing native 1.7,
Lindblad 1.3-1.6 and circuit/Ising plans remain unchanged and backward-readable.
Rehashed mismatches, a 1.0 source relabelled 1.4 or a no-observation plan labelled
1.8 are refused; numerically plausible matrices cannot replace source meaning.

The new plan has the existing native top-level fields schema, version, request,
decision, program, bindings and identity, plus exactly options and
environment_requirements. Options reuse the existing ExecutionOptions encoding,
not a second sampling-options protocol. The existing plan identity covers the
whole new payload. Environment requirements reuse protocol 1.1 and the exact
`str(torch.__version__)` RNG requirement; no runtime handle is serialized.
Readers and execution preflight reject a build mismatch before advancing or
drawing. Old plans do not gain an options/environment field.

The 1.8 planner requires an explicitly supplied ExecutionOptions object with
`shots=1` and `type(seed) is int`, `0 <= seed <= 2**64-1`. It may not inherit
seed/shots from defaults or environment. Support only CPU complex128,
world-size one and one unbatched input state/density; retain the existing
complex128 default. Refuse requested gradients and any live input/control/output
tensor requiring gradients before capture. Ordinary native plan 1.7 keeps its
existing trainable behavior. Retain the existing integrator choices and their
numerical failure/accuracy requirements; do not silently substitute a solver.

Outside the two explicit sampling controls, existing unsupported Lindblad
options stay unsupported, including memory_limit_bytes. Reuse the shared dense
allocation ceiling before numerical capture/trajectory retention or draws.
The instrument's existing numeric subtotal is `48 * D**2 + 24` bytes for one
state, two Kraus matrices, two float64 weights and one int64 draw (456 bytes for
one actor, 3,912 for two). Count requested retained time densities separately
as `16 * T * D**2`; this is a numeric subtotal, **not** peak workspace or RSS.
No memory/scalability/performance claim follows from those numbers.

Root execution of a sealed 1.8 LindbladPlan rejects options, measurements and
noise_model overrides with TypeError, just as existing sealed root plans do.
`fq.plan` is unchanged. Existing `fql.run` accepts its old profiles unchanged
and refuses 1.8 with CapabilityError before execution; users call `fq.run(plan)`
for the new sampled profile. Root execution continues to refuse old Lindblad
profiles rather than silently broadening their support. Existing
Deployment/Remote front doors retain their own fail-closed admission.

## Canonical results and exact proposed signature deltas

Append LindbladPlan to the input type union of `fq.run`; do not add parameters,
change defaults or change its ExecutionResult return type. Extend only the
ExecutionResult.plan type annotation from its current union to
`ExecutionPlan | RuntimePlanContract | MPSProductionPlan | LindbladPlan | None`.
Field order/defaults, MeasurementResult, EvolutionResult and their existing
public imports remain unchanged. No result-class relocation is required.

For the new profile only:

- `state` is final conditional CPU complex128 density `(1,D,D)`;
- `value` and `samples` are None; no second implicit expectation packing;
- requested expectations appear in existing ordered `expectation` records,
  with `(T,)` values and existing output index/name metadata;
- if return_density_matrices=True was planned, one existing `density_matrix`
  record holds the forward trajectory `(T,D,D)`; otherwise it is absent;
- one `mid_circuit_measurement` record is last, with empty qubits, shots=1 and
  CPU int64 index `(1,1)`. Metadata contains exactly outcome_labels,
  probabilities `(1,2)` (original weights), probability_correction, atom_id,
  time_index, time, time_unit, stage="intermediate" and source_operation_index.
  No classical bit or inferred trap flag;
- runtime metadata reuses mode/backend/device/precision, single-device/world-size,
  exact-plan and seed/shots fields, adds the captured time_grid, and explicitly
  records state_axes and expectation_axes. A retained density record declares
  axes `time,basis_row,basis_column` in its metadata.

Use existing output naming/index selection. Reserve `mid_circuit_measurement`
and, when a density trajectory is requested, `density_matrix` against conflicting
expectation names before execution. Existing named expectations remain usable
through `result.expectation(name)`. The final state is not a time batch, and the
trajectory time axis is not an independent-sample axis.

## Migration and refusal classes

No existing caller must migrate: ordinary `fql.plan/run`, native 1.7 training,
root circuits/Ising and terminal sampling keep their current behavior. New
callers opt into the source operation, explicit sampling controls and the
`fql.plan` to `fq.run` artifact join together. They cannot relabel an old plan
to acquire sampling or supply a preselected outcome through a public parameter.

| Trigger | Required failure, before partial results escape |
| --- | --- |
| Wrong atom/scalar type, including bool/tensors for time/efficiency | TypeError under the existing Core leaf rules. |
| Unknown atom, invalid finite/range value, repeated observation or invalid source horizon | IRValidationError under Core construction/snapshot validation. |
| Missing/ambiguous internal grid boundary | Existing EvolutionValidationError with invalid_native_request at times, before integration. |
| Missing explicit sampling options, shots other than one, unsupported profile/device/batch/gradients/options | CapabilityError before numerical capture/execution; wrong seed type is TypeError and out-of-range seed is ValueError. |
| Corrupt/rehashed source, matrices, bindings, options or version | SerializationError from LindbladPlan validation. |
| Incompatible sealed RNG environment | Existing ExecutionPlanContractError with reason_code=environment_incompatible at reading/preflight, before kernel/draw. |
| Overrides on a sealed plan | TypeError. |
| Malformed draw, out-of-range index or selected zero-probability branch | ExecutionError, with no partial result returned. |
| Sampled plan passed to fql.run, or an old Lindblad profile passed to the new root branch | CapabilityError before execution. |

The proposal adds no alias-only entry, migration wrapper, compatibility registry
or result constructor. Candidate/snapshot admission must enumerate these exact
deltas after approval; regenerating a snapshot cannot approve them.

## Proposed user workflow (not implemented)

```python
import torch
import flagquantum as fq
import flagquantum.lindblad as fql
from examples.train_rydberg_controls import moving_pair_program

program = moving_pair_program()
program.measure_presence("a", at=0.5, efficiency=0.8)  # Proposed method.
plan = fql.plan(
    program,
    program.basis_state(("g", "g")),
    torch.linspace(0, 1, 129, dtype=torch.float64),
    parameter_values=[{"omega": 1.2}] * 128,
    outputs=fq.expectation(program.survival_operator(), name="survival"),
    options=fq.ExecutionOptions(shots=1, seed=17),
    return_density_matrices=True,
)
result = fq.run(fql.LindbladPlan.from_json(plan.to_json()))  # Proposed 1.8 admission.
observed = result.measurement("mid_circuit_measurement")
print(result.state.shape)                 # (1, 9, 9)
print(result.expectation("survival").shape)  # (129,)
print(observed.value.shape)               # (1, 1)
```

This proposed workflow must not be run as a claim of current support. The
current root entry does not accept LindbladPlan and the source method is absent.

## Runnable current-API reference

The following executable reference uses existing APIs only. It samples a
snapshot of an already evolved native schedule, then explicitly evolves an
independently known numeric suffix. It is a mathematical/reference composition,
not native intermediate execution, a new runner or a serialized sampled plan.
It never slices/restarts the native source or changes its clock.

```python
import math
import torch
import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.experimental.simulation import RydbergProgram
from flagquantum.operators import SuperOperator
from flagquantum.runtime import sample_terminal

program = RydbergProgram(atoms=(("a", "home"),), sites={"home": (0.0, 0.0)}, length_unit="um", time_unit="us", c6=0.0, loss_rate=0.4, minimum_separation=1.0, drive_channels=("laser",))
program.pulse("a", start=0.0, duration=0.5, amplitude=1.2, detuning=0.0, phase=0.0, channel="laser")
program.pulse("a", start=0.5, duration=0.5, amplitude=0.0, detuning=0.7, phase=0.0, channel="laser")
plan = fql.plan(program, program.basis_state(("g",)), torch.linspace(0, 1, 129, dtype=torch.float64), method="runge-kutta", return_density_matrices=True)
r = fql.run(fql.LindbladPlan.from_json(plan.to_json()))
assert r.density_matrices is not None
before = r.density_matrices[64]
eta = 0.8
branches = {"no_flag": (torch.diag(torch.tensor([1.0, 1.0, math.sqrt(1 - eta)], dtype=torch.complex128)),), "flag": (torch.diag(torch.tensor([0.0, 0.0, math.sqrt(eta)], dtype=torch.complex128)),)}
observed = sample_terminal(before, branches, shots=1, seed=17)
index = int(observed.value[0])
unnormalized = [SuperOperator.from_kraus(ks).apply(before) for ks in branches.values()]
weights = [float(torch.trace(x).real) for x in unnormalized]
conditional = unnormalized[index] / weights[index]
# Explicit numeric reference for the independently known second interval.
h = torch.diag(torch.tensor([0.0, -0.7, 0.0], dtype=torch.complex128))
jumps = []
for level in (0, 1):
    jump = torch.zeros((3, 3), dtype=torch.complex128)
    jump[2, level] = 1
    jumps.append((jump, 0.4))
suffix = fql.run(h, conditional, torch.linspace(0, 0.5, 65, dtype=torch.float64), dimensions=(3,), collapse_operators=jumps, method="runge-kutta", return_density_matrices=True)
assert suffix.density_matrices is not None
print(round(weights[1], 8))
print(abs(weights[1] - eta * (1 - math.exp(-0.2))) < 1e-10)
print(abs(float(torch.trace(suffix.density_matrices[-1]).real) - 1) < 1e-10)
print(torch.equal(observed.value, sample_terminal(before, branches, shots=1, seed=17).value))
```

Executed output is `0.1450154`, `True`, `True`, `True`. The first number is the
observed flag probability, not physical loss `1-exp(-0.2)` and not a hard-coded
sampled label. The analytic survival/efficiency relation and conditional trace
are checked independently of the chosen outcome.

## Executed current-API preparation

On Python 3.10, 3.11 and 3.12, an independent row-vectorized generator/matrix
exponential was compared with existing native evolution and explicit numeric
conditional continuations. Each version ran four initial states, 12 instrument
cases, 19 positive conditional branches and five zero contributions without
normalizing zero. The maximum conditional full-density error was
`1.0970205391191358e-12`, below the declared `1e-10`/rtol=0 integration budget.
Weighted branches recovered the corresponding unread continuation. A coherent
finite-basis lost-sector input also witnessed nontrivial unread back-action;
that algebra test is not a claim of preparing such coherence in hardware.

These probes use only existing APIs and establish the reference mathematics.
They do not implement or validate the proposed source 1.4, plan 1.8, root
admission or result projection. The future replaceable baseline must add their
own frozen consumers after explicit approval.

## Replaceable acceptance before implementation

After explicit approval, first capture old source/plan strings and install a
replaceable reference under the existing test factories; do not publish a fake
as Program IR 1.4 or LindbladPlan 1.8. Freeze independent consumers before domain
implementation, then replace factories and delete the disposable reference.

Acceptance must cover:

1. One actor with independent driven/loss prefix and noncommuting detuned
   suffix; physical survival `exp(-gamma*t)` versus observed flag/no-flag
   probabilities and conditional full complex densities. The runnable probe
   above is preparation, not this future implementation baseline.
2. A moving two-actor interacting example from the existing native schedule,
   with roster-permutation/identity checks and an independent finite-dimensional
   density reference. Reuse both initial g/r populations and coherences; test
   a lost component, efficiency zero/one and a zero-weight branch explicitly.
3. Both conditional branches, weighted unread equivalence and original-trace
   normalization. Keep fixed channel-algebra errors at 1e-12/rtol=0, and the
   independent complex128 time-integration comparison at 1e-10/rtol=0 on a
   sufficiently resolved declared grid; numerical convergence is separate.
4. Explicit time-order/boundary tests, post-observation grid-row semantics,
   requested output names/axes, private RNG invariance, snapshots and exact
   same-environment numeric JSON replay. Do not hard-code a label across builds.
5. The existing 4,096-draw private-helper witness keeps its 1,856..2,240 interval;
   use a separately declared binomial budget for any unequal-probability draw
   experiment. Statistical acceptance cannot compensate for a density error.
6. Rehashed source/plan/options/environment tampering, gradient/batch/device/
   multiple-observation/endpoint/control-grid refusals before kernel/draw,
   existing deployment refusal and old numeric/Ising/circuit/native bytes.
7. A normal circuit import/execution still follows the existing local CPU fast
   path, without eager dynamics imports or distributed initialization.

The shared baseline must precede Core source/codec, Simulation interval/branch,
and Runtime plan/result integration PRs. Each implementation PR owns one primary
domain and preserves independent consumers. The Lindblad public facade changes
are part of the reviewed Runtime-facing contract, not permission for one giant
cross-domain implementation.

Feedback, multi-event trajectories, multi-shot native batches, event latency,
false positives, destructive detection, species calibration, motional heating,
remote jobs, device support, erasure-decoder records and gradients through random
choices remain outside this proposal. It does not declare M3 or M4 complete.
