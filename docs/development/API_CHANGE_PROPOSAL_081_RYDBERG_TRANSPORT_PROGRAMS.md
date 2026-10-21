# Proposal 081: Native Rydberg transport programs on the shared Core boundary

## Status and decisions requested

**Proposed; not approved for implementation.** This document changes no API,
IR, plan schema, capability status or implementation. The inspected baseline is
main `085c7b94157bd63699d2f274532776d6ab5dbb09` on 2026-10-21. The design and
current-API loss/detection reference are reviewed separately in
[#687](https://github.com/flagos-ai/FlagQuantum/pull/687); that PR's review or
merge is not approval of this proposal. Proposals 073/078/079/080 remain approved
for their existing scopes.

The following decisions require an explicit owner decision together:

1. Add one experimental `RydbergProgram` frontend, backed by the existing Core
   `ProgramRecord`, with explicit atom/site identity, timed moves and pulses.
2. Add a typed physical-control profile to that same internal Program IR. Keep
   CircuitIR 1.0, existing SSA value kinds and unimplemented level exits unchanged.
3. Let the three existing high-level Lindblad entries consume that profile on
   CPU, reusing their mathematical state spaces, binding, kernels and results.
4. Record original transport meaning and interval bindings in native-only
   Lindblad plans 1.7; preserve old plan bytes and identities.
5. Admit the explicitly recorded midpoint approximation of imposed linear paths,
   with separate geometry/solver convergence evidence and no motional claims.
6. Provide deterministic loss and end-time presence effects. Do not claim sampled
   detection, event feedback, physical hardware execution or completion of M3.
7. Establish a replaceable contract baseline before domain implementations.
   Each implementation/merge PR still needs its own authorization and verification.

Do not implement by regenerating snapshots or treating a documentation merge as
approval. Follow [public API protection](PUBLIC_API_PROTECTION.md).

## Existing boundaries that must be reused

Core already owns `ProductStateSpace`, parameters and their expression/binding
rules, the scalar JSON codec, and the Program IR's functions, blocks, operations,
verification and diagnostics. Simulation already owns Lindblad normalization,
Liouvillian, three integrators, trajectory outputs and the initial-state adjoint.
The high-level Lindblad facade already owns plan/run composition. Outputs remain
`fq.expectation` and `EvolutionResult`, as approved in proposal 080.

`CircuitSchedule` describes logical dependency layers, not physical time.
`PhysicalCircuitPlan` describes routed qubit circuit evidence, not atom paths.
The internal `quantum` and `target` level exits are declared but unimplemented.
Do not call them implemented, relabel a SWAP as transport, or attach semantic
motion information to ignored metadata. `ProgramArtifactV2/V3` currently target
circuits/static text; they are not a native analog-program serialization path.

The existing register adapter remains Remote-owned vendor discovery/preflight.
Core must not import it. Its coordinate extraction is not an atom/site identity
model, and this proposal does not make it submit an analog program.
Factor the finite Cartesian-pair validation already used there into one small
Core-owned primitive, then have the existing Remote adapter and native profile
consume it. Remote retains vendor field parsing, unit spellings, messages and
its existing handling of coordinate sequences; prove unchanged profiles and
refusals through its old tests. Do not copy its parser into the new frontend or
import Remote from Core. This replacement is a separate Remote-owned PR after
the Core boundary lands, not an unrelated edit hidden in a numerical PR.

## Proposed user surface

Only the existing `flagquantum.experimental.simulation` namespace gains the
feature export `RydbergProgram`. There is no new root export, top-level package,
backend registry, general model manager, training loop or numerical core.
The implementation frontend is Core-owned, under `flagquantum/core/rydberg.py`;
it imports no Runtime, Compiler, Simulation or provider implementation. Tensor
helpers construct small mathematical inputs, not numerical evolution engines.

The proposed signatures are:

```python
# Proposed surface only; not executable on the inspected baseline.
class RydbergProgram:
    def __init__(
        self, *, atoms: Sequence[tuple[str, str]],
        sites: Mapping[str, tuple[float, float]],
        length_unit: Literal["m", "um", "nm"],
        time_unit: Literal["s", "ms", "us"],
        c6: float, loss_rate: float, minimum_separation: float,
        coordinate_frame: str = "program_local",
        transport_channels: Sequence[str] = (),
        drive_channels: Sequence[str] = (),
        geometry_sampling: Literal["interval_midpoint"] = "interval_midpoint",
    ) -> None: ...

    @property
    def atom_ids(self) -> tuple[str, ...]: ...

    @property
    def dimensions(self) -> tuple[int, ...]: ...

    def move(
        self, atoms: str | Sequence[str], *,
        source: str | Sequence[str], destination: str | Sequence[str],
        start: float, duration: float, channel: str,
    ) -> RydbergProgram: ...

    def pulse(
        self, atoms: str | Sequence[str], *, start: float, duration: float,
        amplitude: float | Parameter | ParameterExpression,
        detuning: float | Parameter | ParameterExpression = 0.0,
        phase: float | Parameter | ParameterExpression = 0.0,
        channel: str,
    ) -> RydbergProgram: ...

    def commanded_position(self, atom: str, time: float) -> tuple[float, float]: ...
    def basis_state(self, occupations: Sequence[str]) -> torch.Tensor: ...
    def projector(self, occupations: Mapping[str, str]) -> torch.Tensor: ...
    def survival_operator(self) -> torch.Tensor: ...
    def no_flag_operator(self, efficiency: float) -> torch.Tensor: ...
    def to_ir(self) -> ProgramRecord: ...
```

Methods mirror the existing fluent construction style: `move`/`pulse` append
Core operations and return the same frontend. Its single source is a Core
record, not a second command list or IR. `to_ir` returns an immutable, validated
snapshot. Later frontend edits do not modify an earlier record or plan.

Proposed use after approval and implementation:

```python
# Proposed API: deliberately not claimed as runnable today.
import math
import torch
import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.experimental.simulation import RydbergProgram

program = RydbergProgram(
    atoms=(("a", "left"), ("b", "near")),
    sites={"left": (0., 0.), "near": (2., 0.), "far": (4., 0.)},
    length_unit="um", time_unit="us", c6=64., loss_rate=0.2,
    minimum_separation=1., transport_channels=("aod",),
)
program.move("b", source="near", destination="far",
             start=0., duration=0.5, channel="aod")
initial = (program.basis_state(("g", "g"))
           + program.basis_state(("r", "r"))) / math.sqrt(2)
plan = fql.plan(
    program, initial, torch.linspace(0., 0.5, 129, dtype=torch.float64),
    outputs=(fq.expectation(program.survival_operator(), name="survival"),
             fq.expectation(program.projector({"a": "r", "b": "r"}), name="rr"),
             fq.expectation(program.no_flag_operator(0.8), name="no_flag")),
    method="krylov-exponential",
)
result = fql.run(plan)
```

The numbers above specify an illustrative effective model, not a species or
device calibration. The final survival and RR probabilities are exp(-0.2) and
exp(-0.2)/2. The profile retains the actual linear path and its approximation;
it is not the two-distance discontinuous reference from #687.

## Identity, geometry and resources

- `atoms` is an ordered sequence of `(atom_id, initial_site_id)` pairs. That
  order is the subsystem order, with the first atom most significant. IDs are
  nonempty strings, unique within their role, never coerced or renumbered.
  Atom and site namespaces are separate; equal spellings across roles are legal.
- Sites are named 2D Cartesian points in the declared coordinate_frame, a nonempty
  program-local frame identifier; no cross-frame transform is inferred. All sites use the
  same length unit; the time grid and operations use the declared time unit.
  C6 has angular-frequency times length-to-the-sixth units, loss_rate has inverse
  time units, and phases are radians. No Hz/rad/s conversion or species inference
  occurs. Changing units requires changing all related physical inputs.
- Geometric inputs accept finite ordinary Python real numbers, excluding bool.
  minimum_separation is positive; loss_rate is nonnegative; C6 may be signed or
  zero. Duplicate site coordinates, duplicate actor assignments, unknown sites
  and malformed rosters are refused. Physical real literals normalize to floats;
  data is copied into immutable tuples. Expression operators remain Core's
  existing operators, without a new expression language or algebraic rewriting.
- A move group has matching nonempty actor/source/destination sequences. A
  single string denotes one ID, not its characters. Actor duplicates are refused.
  Group pairs canonicalize by the declared subsystem order.
- Paths are straight Cartesian interpolation over `[start, start+duration]` and
  hold at their source/destination before/after the move. start is nonnegative
  and duration positive. Source occupancy at start must match the declared prior
  history. Explicit timestamps determine history, not Python append order.
- All actors in one group read the same pre-move occupancy. Check continuous
  minimum pair separation, including overlaps and endpoints, against every
  potentially present actor. Sharing only endpoints does not prove path safety.
  Loss uncertainty does not renumber actors or make a potentially unsafe path safe.
- Two overlapping moves of one actor are invalid. Each declared channel carries
  one operation/group at a time; abutting intervals are allowed. Channels are
  named, unique, declared by role and not shared ambiguously across roles.
  A multi-actor group is one simultaneous operation. This is a simulation resource
  model, not certification of a vendor's AOD/bus geometry or timing constraints.
- Pulses may address a moving actor. They use the declared, spatially uniform
  effective field on their target group. There is no inferred Doppler shift,
  beam waist, acceleration heating, crosstalk or transfer infidelity model.
  At most one pulse may address an actor at a time, even on distinct channels;
  disjoint target groups on distinct channels may be simultaneous. This avoids
  silently combining incompatible rotating-frame drive declarations.

At a shared boundary the position is continuous and the following interval reads
the completed move. Refill, pickup/dropoff controller operations, conditional
moves and runtime presence-dependent branching are outside this first profile.
commanded_position reports the imposed path, not an observation or proof that
the actor is still present. Physical presence remains a separate quantum output.

## Typed extension of the existing Program IR

The additive internal fields are exactly:

```python
# Proposed additions; not existing declarations.
Operation.physical_data: RydbergTransport | RydbergPulse | None = None
ProgramRecord.physical_context: RydbergContext | None = None
```

The three new frozen, deeply immutable internal records contain:

| Record | Fields in order |
| --- | --- |
| RydbergContext | atoms, sites, length_unit, time_unit, c6, loss_rate, minimum_separation, coordinate_frame, transport_channels, drive_channels, geometry_sampling |
| RydbergTransport | atoms, source, destination, start, duration, channel |
| RydbergPulse | atoms, start, duration, amplitude, detuning, phase, channel |

`atoms`/`sites` in the context are the immutable pairs described above; operation
actor/site/channel references are strings/tuples of strings. Time and physical
constants are finite reals. Pulse scalars reuse Core `Parameter` and
`ParameterExpression`, whose leaves here must be real literals or parameters.
Live tensors enter through existing interval bindings, not mutable tensor leaves
inside the source IR. The existing Core scalar codec represents expressions;
there is no new expression AST or binding evaluator.

Only opcodes `rydberg.move`/`rydberg.pulse` carry these records, matching the
corresponding exact record type. All ordinary Operation fields are empty/None
on those control operations. No data is hidden in metadata or an identity gate.
The context identifies a homogeneous ground/Rydberg/absorbing-loss model, with
local order `(g, r, lost)`. Product dimensions are computed by the existing
`ProductStateSpace`; they are not a second stored state-space model.

These opcode meanings are scoped by the typed physical context. Do not add them
to the global circuit operator manifest or change ordinary opcode_signature
lookup. A pre-existing CircuitIR custom operation whose name happens to equal
one of these strings retains its explicit-matrix semantics and old bounded-level
refusal; it does not acquire transport meaning. Test this collision explicitly.
Only the physical-profile verifier admits the typed native operations.

This physical-control profile has one function `main`, one entry block and a
return terminator, no edges, SSA arguments or conditional regions. It is a
declarative timed-control model for an external Lindblad initial state, not a
qubit state converted into a three-level SSA value. Existing SSA value kinds,
linearity, measurement and circuit operations retain their current contracts.
Mixing those operations into this profile is explicitly unsupported.

The existing `verify_program_record` dispatches to the profile's verifier rather
than a new registry. It validates identities, sources, continuous paths, channel
conflicts, parameters and unsupported structure. It returns existing diagnostic
records with codes `physical.invalid_context`, `physical.invalid_operation`,
`physical.occupancy_conflict`, `physical.path_collision`,
`physical.channel_conflict`, or `physical.unsupported_structure` as applicable.
The frontend raises `IRValidationError` for a verified-invalid program, retaining
field/operation context in its message. Wrong argument types raise TypeError.
Structurally valid but unimplemented program structure raises CapabilityError;
it must not be mislabeled as supported or approximated into a flat program.
The generic lower_to_level entry accepts a native frontend/record for the
program exit, returning the existing LevelConversion with its validated record
and no canonical circuit payload. Invalid native data uses INVALID_INPUT and
the diagnostics above. circuit_to_program and the circuit round-trip helper
remain circuit-specific; do not force native input through a fake CircuitIR.

Canonicalization sorts named sites/channels and timed operations deterministically,
preserving atom/subsystem order and simultaneous group meaning. Existing
Operation/ProgramRecord equality and hashes must remain exact when both new
fields are None. Physical records include their full typed meaning in equality
and hashing. Appending None to the old dataclass hash tuple is not compatible;
capture the actual old values before implementation and preserve them.
Capture built-in hashes per supported Python with a fixed hash seed; no equality
of Python hash integers across Python versions is promised. SHA-based numeric
plan identities retain their existing separate compatibility requirements.

Conversions to CircuitIR 1.0 refuse native timelines with an actionable existing
unsupported diagnostic; they do not emit SWAPs or discard paths. Existing
`REALIZED_EXITS=(program,)` is unchanged. No QuantumIR/TargetIR completion is
claimed, and no new IR hierarchy, pass registry or artifact kind is created.

## Physical model and numerical consumption

With hbar=1, on the declared local basis the model is

`H(t) = sum_i [Omega_i(t)/2 * (exp(i*phi_i)|g><r| + h.c.) - Delta_i(t)*n_i]
        + sum_(i<j) C6/r_ij(t)^6 * n_i*n_j`.

Each actor has two jumps, `|lost><g|` and `|lost><r|`, at the declared loss_rate.
The lost label is an absorbing occupancy marker, not an atomic energy level.
Interactions and drives act as zero on that marker; density remains normalized
over all sectors. This is explicitly state-independent Markov trap loss, not
spontaneous decay back to the ground manifold.

The high-level `fql.plan`, `fql.run` and `fql.adjoint_gradient` accept the frontend
or its Core record. Their names, argument order/defaults and result types do not
change. Other Hamiltonian forms retain all existing semantics. The first numerical
target is CPU with one or two actors; larger valid records fail with
CapabilityError before exponential-size tensor construction. No array-size,
hardware or distributed-capacity claim follows from this pair implementation.
The four tensor helpers enforce the same guard before allocation. Geometry and
IR queries may represent larger rosters without admitting numerical execution.

Times must contain at least two values, start at zero, increase strictly, include
every move/pulse boundary and cover all operations. For each output interval, positions are evaluated at
its midpoint; active pulses and bindings define its Hamiltonian. The recorded
`interval_midpoint` model is a controlled approximation of continuous imposed
paths, not an automatic fallback or exact motion integration. It requires its
own grid-convergence evidence, separate from integrator error.
Use one Core-owned eight-ULP boundary comparison: binary64 for source schedule
validation, and the requested real precision after normalizing boundaries and
the grid for planning. An ambiguous match to multiple grid points, or distinct
required boundaries collapsing at that precision, is refused. Do not use raw
decimal-float equality or silently move a boundary to a different interval.

`parameter_values` keeps proposal 073's per-interval mapping semantics, including
key normalization, missing/extra names, scalar validation and PyTorch graphs.
Parameters/expressions may set amplitude, detuning and phase. Geometry, times,
channel assignments, C6 and loss_rate are fixed inputs in this profile; requests
to differentiate them are refused rather than silently detached. Initial-state
batches and gradients retain their existing meanings. B=1 keeps its batch axis;
all batch members share the physical program, not a hidden geometry batch.
The adjoint still returns a cotangent of the normalized initial density matrix,
not of a raw ket or a physical coordinate. Native time-grid gradients retain
the existing unsupported_time_gradient refusal.

The native route derives product dimensions; supplied dimensions must match.
n_qubits and basis_labels must be omitted. Non-None collapse_operators are
refused because this model already owns its jumps. No option previously refused
by Lindblad is silently accepted, including memory_limit_bytes, shots, seed,
allow_approximate and fallback controls. Unsupported devices/profiles are refused.
Explicit construction of this recorded approximation is distinct from admitting
an unimplemented general approximation-policy option.
An otherwise valid native program with an inconsistent grid or semantic override
raises EvolutionValidationError with code invalid_native_request and the affected
argument as field. Existing parameter-binding codes and option/device categories
are retained; this new code applies only to native input consistency.

Numeric assembly consumes Core geometry/parameter semantics and feeds the existing
Liouvillian/integrators. Reuse the existing private pre-bound interval path; do
not invent synthetic parameter names to force numeric moving geometry through
the public named-parameter binder. Reuse the same initial-state adjoint loop,
factoring a private pre-bound entry if necessary. There is no second solver,
normalizer, trainer, resource scheduler or binding engine.

## Readout and unsupported event semantics

Tensor helpers return CPU complex128 mathematical inputs of the declared product
dimension. They use existing product ordering and do no evolution:

- basis_state takes one of g/r/lost per actor and returns its product basis vector.
- projector selects named actors' states; unspecified actors contribute identity.
  Unknown IDs/states are refused; an empty selection is identity.
- survival_operator projects every actor onto the span of g/r.
- no_flag_operator uses the independent effect diag(1,1,1-eta) per actor, with
  finite eta in [0,1] and no false positives. eta=0 gives identity; eta=1 gives
  survival. This is an explicit detector-effect assumption, not a sampled detector.

All measurement uses existing `fq.expectation` and named EvolutionResult accessors.
The helpers yield ordinary matrix readout inputs; they do not create native
detector instructions or add detector provenance to the transport source profile.
Unconditional RR, physical survival and no-flag selection remain distinct.
Conditioning is explicit user arithmetic with its denominator, never an implicit
renormalization by the solver. No hidden loss truth is exposed to a decoder as
if it had been detected. Counts, erasure records, mid-program instruments,
feedback, motion heating, quantum motional modes, device emission and QPU
submission remain outside this profile and retain separate M3/M4 gates.

Circuit, deployment and provider front doors refuse native records before
execution/submission, directing the caller to Lindblad. Extend the existing
Core circuit normalization guard; do not spread provider-specific checks or
submit a digital approximation. Low-level mathematical Lindblad entries do not
gain program dispatch; their existing mathematical inputs remain unchanged.

## Native-only plan 1.7

Existing LindbladPlan/EvolutionPlan/EvolutionResult fields and accessors remain.
Plans 1.3/1.4/1.5/1.6 keep exact old JSON and identities. Every native plan uses
1.7, even a static one, because dropping its physical source would lose meaning.

The 1.7 top-level fields are exactly schema, version, request, decision, program,
bindings and identity. schema remains `flagquantum.lindblad_plan`. request and
decision use the existing product-space, pre-bound interval representation and
numerical decision fields. No physical attributes are added to numerical states.

program encodes the actual Core record, with schema `flagquantum.program_ir`,
version `1.0`, functions and physical_context. The graph is the same Core
function/block/operation model; its physical_data/context fields encode the
records above, with a `kind: rydberg` context discriminator. No canonical circuit
payload is invented. The shared scalar codec handles symbolic pulse fields.
bindings is null for a fully numeric source, otherwise the exact normalized
per-interval name-to-real-value mappings at the requested precision.

The existing plan identity covers all fields, including source and bindings.
Deserialization validates the Core source, dimensions, grid, binding and source
consistency with captured Hamiltonians/jumps. Re-derived numeric entries allow
only 16 times real-dtype machine epsilon in absolute/relative reconstruction
comparison; this is an encoding check, not a relaxed physical accuracy budget.
Inconsistent native source raises SerializationError, even if its outer identity
has been recomputed. Unknown nested native fields, model kinds, opcodes and units
are refused. Old unknown-field rejection remains in place.
Native scalar encodings must survive decode/re-encode through the existing Core
codec in canonical JSON form; reject coercing/noncanonical native encodings
without changing the accepted legacy circuit encoding rules.

run(plan) uses captured numeric intervals and never recompiles/rebinds the physical
source. Semantic overrides remain refused. JSON restores numeric state, not an
autograd graph; gradients_not_available remains the existing refusal when a
serialized plan requires a live graph. Live training rebuilds a plan after each
optimizer update. No second result envelope or public fingerprint API is added.

## Runnable current-API convergence reference

This is executable on the inspected main without the proposed frontend. It
isolates a linear separation path's interaction phase, with loss disabled;
#687 separately checks loss, detection and conditioning on the full nine-state
space. Here r(t)=2+4t, C6=64, T=0.5. The exact continuum phase is 0.096875.

```python
import math
import torch
import flagquantum as fq
import flagquantum.lindblad as fql

interaction = fq.Parameter("interaction")
rr = torch.diag(torch.tensor([0., 0., 0., 1.], dtype=torch.complex128))
coherence = torch.zeros((4, 4), dtype=torch.complex128)
coherence[0, 3] = coherence[3, 0] = 1
initial = torch.tensor([1., 0., 0., 1.], dtype=torch.complex128) / math.sqrt(2)
exact_phase = 64 / (5 * 4) * (2**-5 - 4**-5)
errors = []
for intervals in (32, 64, 128):
    times = torch.linspace(0., 0.5, intervals + 1, dtype=torch.float64)
    midpoints = (times[:-1] + times[1:]) / 2
    values = 64 / (2 + 4 * midpoints)**6
    plan = fql.plan(
        [{"operator": rr, "coefficient": interaction}], initial, times,
        dimensions=(2, 2), parameter_values=[{"interaction": v} for v in values],
        outputs=fq.expectation(coherence, name="coherence"),
        method="krylov-exponential",
    )
    actual = fql.run(plan).expectation("coherence")[-1].item()
    frozen_phase = (values * torch.diff(times)).sum().item()
    assert abs(actual - math.cos(frozen_phase)) < 1e-10
    errors.append(abs(actual - math.cos(exact_phase)))
print(f"Continuum phase: {exact_phase:.8f}")
print("Second-order geometry convergence:", 3.9 < errors[0] / errors[1] < 4.1
      and 3.9 < errors[1] / errors[2] < 4.1)
print("Frozen-model quantum error below 1e-10: True")
```

Expected output after running the assertions:

```text
Continuum phase: 0.09687500
Second-order geometry convergence: True
Frozen-model quantum error below 1e-10: True
```

## Delivery and acceptance gates

1. **Explicit approval and baseline.** Record the owner decision; capture old
   ProgramRecord/Operation signatures, equality/hash values and ten old plan
   fixtures before edits. Add the approved contract claims and a replaceable
   fixture boundary. Use small fixed reference adapters over the existing engine,
   not a complete second transport compiler or fake simulator.
2. **Core-owned construction/semantics.** Implement the typed extension and one
   frontend; retain all old circuit/SSA conversion and hash fixtures. Test named
   identity, source history, permutations of simultaneous groups, continuous
   crossings, holding actors, channel overlap, empty/invalid references and loss
   without renumbering. Replace the relevant baseline factory without changing
   its consumers. Integrate ownership/export admission before ordinary code.
3. **Existing geometry consumer replacement.** Change only the Remote-owned
   coordinate normalizer to read the admitted common primitive; retain old vendor
   parsing and all profile/error fixtures. This proves a concrete second consumer
   without making Remote own native physics.
4. **Simulation-owned consumption.** Add numeric model assembly and the thin
   existing Lindblad facade dispatch. Reuse owned normalizers, forward/adjoint
   loops, plans and results. Replace baseline plan/run consumers. No ordinary
   implementation combines unrelated primary domains.
5. **Independent physics.** Compare the frozen model to independent dense
   Liouvillian exponentials at the existing precision budget. Compare moving
   geometry to the analytic integral above and establish second-order refinement
   independently of solver error. Check pulse sign/phase conventions, blockade
   trends, loss survival laws, zero loss, eta=0/0.8/1, unconditional versus selected
   outputs, B=1/B=2 axes and absence of resurrection.
6. **Gradients and replay.** For the supported controls, use independent analytic
   and finite-difference references and an actual PyTorch optimizer update.
   Preserve the existing RK4/Krylov reference budgets, CN's 1e-6 budget and its
   separate discrete-map convergence checks; do not hide known Krylov AD
   conditioning limits. Verify initial-state adjoints against an independent
   dense discrete map. Check input mutation after planning, exact numeric replay,
   altered-source rejection, unsupported-gradient refusal and all old plan bytes.
7. **Front doors and claims.** Prove refusals before kernels, provider imports or
   submission; do not claim native support from the reference script. Keep the
   roadmap's event, heating, hardware and scalability gates open. Run the existing
   repository checks, supported-Python tests and full applicable CI/GPU workflows.
   Each PR includes a runnable current user example and expected output.

Approval of this contract is not a blanket merge grant. Each PR must separately
have user merge authorization, successful latest complete verification and
satisfied repository rules. No protected API, schema, candidate contract or
snapshot changes merely because this document or #687 merges.
