# Native Rydberg transport program authorization

## Owner decision and scope

On 2026-10-22 the repository owner replied "Approved" (English rendering) to
the explicit request to approve proposal 081 for implementation. This approves
all seven decisions in [proposal 081](../development/API_CHANGE_PROPOSAL_081_RYDBERG_TRANSPORT_PROGRAMS.md),
including its replaceable-baseline-first delivery order. The earlier document
merge #690 is not this authorization. Proposals 073/078/079/080 retain their
already approved scopes. This approval grants no blanket PR merge permission.

The admitted boundary is one experimental RydbergProgram frontend over the
existing Core ProgramRecord, typed physical records, existing CPU Lindblad
plan/run/adjoint consumption, and native-only plan 1.7 source capture. It reuses
Core scalar binding/encoding, mathematical state spaces, Simulation's
Liouvillian/integrators/results and the existing PyTorch path. No second IR,
transport compiler, binder, solver, result envelope or backend registry is
authorized. Old CircuitIR 1.0 and plan 1.3/1.4/1.5/1.6 identities remain protected.

The fixed imposed path uses interval midpoints; geometry convergence and frozen
solver accuracy have distinct evidence. Absorbing loss and deterministic final
presence effects are not sampled detector events. Heating, quantum motion,
conditional transport, controller constraints, QPU execution and full M3
acceptance remain outside this first profile.

## Replaceable baseline

The integration source is main `10b2767f9b28516428d72c63a690419ab3268979`,
including the independently landed dynamic Program IR work. No production
source, Stable Core export, contract candidate, numerical kernel or public
plan schema changes in the baseline. In particular, plan 1.7 and the frontend
are not implemented by these references.

`tests/unit/test_rydberg_transport_contract.py` defines three fixture replacement
points: pair_planner, pulse_planner and initial_adjoint. The first uses the
executable fixed linear-path matrix example; the second assembles only a fixed
one-atom drive; the third is the existing adjoint entry. They delegate all
binding, numerical work, outputs and replay to existing owners. They do not
accept general atom programs or implement geometric/resource validation.
When native consumption lands, replace these fixtures with real native program
construction and the existing high-level entries, remove redundant adapters,
and retain the consumers' physical, gradient, discrete-adjoint and replay
assertions. Core-specific construction/refusal consumers are added with that
domain implementation, rather than duplicating a full frontend in tests.

The consumers check all three integrators, unbatched/B=1/B=2 trajectories,
absorbing loss without resurrection, detector efficiencies 0/0.8/1, independent
dense Liouvillian exponentiation, analytic linear-path phase refinement,
pulse sign/phase, analytic and finite-difference control derivatives, a PyTorch
optimizer update, independent discrete initial-state adjoints and exact numeric
JSON replay. RK/Krylov retain 1e-10 physical budgets; CN retains 1e-6 and its
initial-state adjoint is compared to the independent discrete map at 1e-10.
These numerical consumers do not prove native identity, resource scheduling,
continuous collision refusal, native serialization or native front-door support.

The pre-implementation ProgramRecord/Operation captures in
`tests/fixtures/rydberg_program_legacy.json` retain the original three Python
captures and their original file digests, taken at main
`085c7b94157bd63699d2f274532776d6ab5dbb09`. Compatibility consumers protect old
field/signature prefixes, default-absent physical additions, equality/hash
formulas and custom-matrix opcode collisions. The ten existing plan payloads
in `tests/fixtures/lindblad_output_plan_compatibility.json` are reused unchanged,
not copied or regenerated; their established consumers still check exact bytes,
identity and numeric replay. Do not regenerate either fixture to hide a failure.

## Runnable user reference

```bash
python -m examples.rydberg_transport_reference
```

Expected CPU output:

```text
Physical survival: 0.81873075
Unconditional RR: 0.40936538
Linear-path coherence: 0.81489256
Exact numeric replay: True
Native transport API: pending implementation
```

This uses g/r/lost coordinates and the prescribed pair separation r(t)=2+4t
for 0 <= t <= 0.5, C6=64 and loss rate 0.2 per atom, in consistent illustrative
units. The example exposes physical survival separately from unconditional RR.
The consumers also separate physical survival from detector-selected outcomes.
It does not claim native motion from matrix assembly.

## Follow-up gates

1. Land this baseline only after its own explicit merge authorization and full
   applicable CI/GPU success under repository rules.
2. Admit Core ownership/export entries alongside the one frontend and typed
   Program IR extension; preserve old no-context semantics and hash formulas.
   Exercise source history, group permutations, continuous crossings/holding
   actors, channel overlap and type/capability diagnostics. No global circuit
   opcode reservation for the physical profile names.
3. Replace the existing Remote Cartesian-pair normalizer with the admitted
   common primitive in a separate Remote-owned PR. Retain vendor parsing and
   all old profile/error evidence.
4. Implement native model consumption and plan 1.7 in Simulation-owned changes,
   replacing baseline fixtures without weakening consumers. Add native source
   mutation/tampering, binding/gradient and serialized-source refusals. Reuse
   existing scalar codec, forward/adjoint loops and plan/result ownership.
5. Re-run independent physics, compatibility, supported-Python and applicable
   full CI/GPU gates. Each PR includes runnable user code and expected results,
   and requires its own merge authorization. Native success does not close the
   remaining event, motional, hardware or scalability gates.

## Core implementation admission

After baseline #698 merged at `acc420c258042de5189a6edf934efdd8e2d25ec9`,
the Core delivery owns `core/rydberg.py` (frontend), `core/geometry.py` (finite
Cartesian validation), and `core/ir/program/physical.py` (typed records and
physical verification). The existing ProgramRecord/Operation gain only the two
approved optional fields. The existing level verifier/dispatcher and circuit
normalizer consume them; the global circuit opcode manifest stays unchanged.

The existing experimental surface candidate admits exactly RydbergProgram under
experimental.simulation, with no root export or change to its frozen governance
baseline. The existing internal-level candidate admits the six approved
physical diagnostic codes and the one typed profile; its existing checker must
read that admission. This is not a second profile registry or a new IR hierarchy.
Core's local README owns the shortest construction/verification change path.

Numerical native consumption, native plan 1.7 and the Remote adapter replacement
remain subsequent domain deliveries. Core tensor helpers construct small CPU
mathematical inputs; they do not execute the program or certify full M3.

The root run composition calls the same Core-owned native-input guard before
choosing any local or remote route. This closes the pre-existing path that
type-checked in Runtime before reaching Core normalization. Runtime and provider
implementations remain unchanged; there is no provider-specific native check.

Core also owns the native source encoding under
`core/ir/program/physical_codec.py`. It serializes the actual functions, blocks,
operations and typed context using the existing scalar codec, with strict
unknown-field and canonical-round-trip checks. It adds no artifact kind or
public frontend method. Simulation will consume this boundary for plan 1.7;
the source codec itself does not assemble or validate numeric Hamiltonians.

### Core-stage scope and remaining consumer checks

The Core delivery proves source construction, typed identity, continuous path
safety, scalar/source encoding, tensor inputs and refusal through Core
normalization and root plan/run/compile composition. It does not yet prove
every direct deployment/provider/client intake path. In particular, the legacy
deployment package builder performs its own inline to_ir conversion and needs
a domain-owned guard delegation before the complete front-door gate closes.
Keep that preparation separate from Core semantics and from numerical assembly.
The existing Remote coordinate replacement and consumer-intake evidence remain
subsequent domain-owned work. Native numerical enablement must retain the full
front-door/refusal gate; this source-only delivery does not close it or M3.

## Deployment intake preparation

The target-neutral package and Pauli-measurement package builders now delegate
the native-input refusal to Core before reading circuit metadata, selecting a
backend, compiling or emitting a digital circuit. This is a separate
Deployment-owned change stacked on the Core producer. It adds no frontend,
provider rule, compiler pass, numerical dispatch or public signature.

The four real-input regressions failed before the delegation: package creation
raised AttributeError on missing circuit metadata, and Pauli package planning
raised its generic circuit TypeError. They now require the existing
CapabilityError with the Lindblad direction and use side-effect spies before
backend selection, compilation and both emitters. Old circuit package and
routing-evidence consumers remain unchanged. Each PR retains its own latest
CI/GPU and merge-authorization requirements.

## Remote geometry consumer replacement

After Core #700 landed at `050be1d7b30dd5f1c508120242a1195ffc9e66c0`, the
existing Remote register parser delegates finite-coordinate validation to
Core's `finite_real`, the same primitive used by `cartesian_pair` in the native
profile. Remote retains declaration parsing, type acceptance, float conversion,
unit aliases, first-two-coordinate extraction and its existing error messages.
Validation remains sequential: a non-finite first coordinate must still fail
before a wrongly typed second coordinate; an overflowing integer conversion
retains its original OverflowError. No vendor parser or physical model moves
into Core, and there is no second validator or provider-specific native guard.

The pre-change captures cover 32 coordinate outcomes and four unchanged register
profiles. They are reused as compatibility assertions, including numeric
subclasses, ignored third coordinates and mixed-error precedence. Existing
provider roster and golden-path consumers remain unchanged. Additional real
frontend/raw-record consumers prove that both Jiuding submission clients and
the resident measurement intake reuse Core's refusal before authentication,
workspace discovery, target selection or submission. This is offline consumer
evidence, not analog hardware execution or full native numerical acceptance.
