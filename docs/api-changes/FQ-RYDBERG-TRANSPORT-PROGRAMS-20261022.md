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

## Historical runnable baseline reference

The fixed adapter below was run for the baseline and is retired by native
replacement; use examples.rydberg_native_dynamics for the current workflow.

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

## Approved Remote boundary supersession

On 2026-10-22 the owner explicitly approved
[proposal 082](../development/API_CHANGE_PROPOSAL_082_REMOTE_PROVIDER_BOUNDARY.md).
Its four decisions supersede only the generic Remote-coordinate-consumer gate
in this record and proposal 081: retire the non-submitting NeutralAtomProvider
and its generic parser/profile exports; preserve the Core geometry and native
program; correct the roster evidence; require a concrete service adapter before
claiming a Remote neutral-atom integration. PR #702's intermediate finite-check
refactor is superseded rather than merged solely to retain an unused consumer.
The original approval and captured evidence above remain historical records.
All native Simulation, physical, gradient, replay, front-door, old-plan and
remaining M3 acceptance gates continue unchanged. No blanket merge permission
or cloud execution is granted or established by this approval.

## Remote retirement implementation

The Remote-owned proposal 082 delivery deletes the generic provider module and
both namespace exports without an alias or stub. Existing Core/source/math and
Simulation files remain unchanged. Current provider-roster examples use real
vendor declarations for preflight and the existing Core frontend for local atom
geometry. Negative module/export and submission-side-effect consumers cover the
retirement. The authoritative capability/parity records and their generated
documents record the missing concrete neutral-atom service and retain their
previous maturity levels. This delivery does not enable numerical evolution or
cloud access. Separate PR verification and merge authorization remain required.

## Simulation implementation admission

After Core #700, Deployment #701 and approved Remote retirement #705, the
Simulation delivery consumes the existing typed physical ProgramRecord on the
three existing high-level Lindblad entries. Simulation owns bounded numeric
assembly in simulation/rydberg_model.py; the existing Lindblad facade owns
1.7 source/binding capture and consistency checks in lindblad/_native_plan.py.
Existing plan construction gains a private pre-bound input, and the existing
adjoint loop gains a private pre-bound entry while its public signature stays
unchanged. Interval scalar mapping validation is factored from its existing
normalizer for both old schedules and native controls, retaining Core binding.
No new IR, scheduler, binder, solver, registry or result fields are admitted.
Native-only plan 1.7 and invalid_native_request are already approved in 081;
old numerical/source fixtures and all physical/gradient/replay budgets stay.

## Simulation replacement evidence

The three baseline fixtures now construct actual native programs. Their physical,
gradient, optimizer, discrete-adjoint and replay consumer functions and decorators
are unchanged, checked by AST comparison against main f187adbb. The redundant
fixed matrix adapter is removed; the separate neutral_atom_pair_reference remains
an independently used numerical/detection reference. Existing source and old-plan
fixture files are not regenerated.

Run `python -m examples.rydberg_native_dynamics`: version 1.7, survival 0.81873075,
unconditional RR 0.40936538, no-flag 0.85353566 and exact numeric replay True.
The 40 unchanged physics consumers and 48 native consistency/gradient/refusal
cases pass locally. Three-parameter controls are compared to an independent dense
unitary for all three integrators. The focused dynamics suite passes 717 cases each on Python 3.10, 3.11 and 3.12;
strict mypy passes 784 source files and all 11 repository gates pass. Full
default/runtime and remote verification remain PR acceptance gates.

The initial physics run had eight failures: Python scalar bindings were first
converted using Torch's default float32 before requested-precision conversion.
Direct conversion at the requested precision fixed the drift; all original
assertions and budgets were retained. This is not evidence of stronger solver
accuracy. No event, motion-heating, cloud or scalability claim is introduced.

The integration evidence update records this bounded experimental consumer in
the existing continuous_time_lindblad capability limitations and regenerates its
documentation. It does not add a registry entry, raise maturity, change governance
snapshots or promote the frontend to Stable Core.

An additional snapshot test initially compared a single CN interval to the
continuous analytic derivative (difference 2.37e-6 against a 1e-6 budget). The
snapshot test now uses a refined RK grid; the separate unchanged consumers still
validate CN against its discrete map and retain its original physical budgets.

Native scalar classification now checks tensor/numeric types without a temporary
int64/default-float tensor. The prior branch raised ValueError for a finite real
binding of 2**70 before requested-precision conversion; the regression now seals
its exact real value without executing an extreme-frequency model. Fully numeric
sources also retain proposal 073's empty per-interval mapping semantics: the
shared validator accepts empty mappings, refuses unused names, and plan 1.7 still
captures null bindings. No solver or physical tolerance changes accompany these
input-boundary fixes. The final native/unchanged-physics selection passes 87 cases.

### Completed local verification and exact scope

Initial feature d858f88019 completed the full default selection: 14,540 passed,
169 skipped and 3,034 deselected in 917.92 seconds. That run collected the first
36 native cases before subsequent scalar-boundary fixes and added tests; it is
not reported as an exact-head default run of those later changes.

Final production code 6bf24b7df5 completed the runtime integration selection:
1,769 passed, 129 skipped and 15,856 deselected in 382.72 seconds. The final native
selection has 48 passing cases on each of Python 3.10, 3.11 and 3.12, including an
independent dense-unitary simultaneous-pair drive and a C6 blockade trend at the
unchanged 1e-10 exponential budget. The 40 original physics consumers passed on
all three versions after the scalar fixes. Python 3.12's broader focused dynamics
selection passed 719 cases before the final additional blockade case. Earlier
broad selections passed 717/718 cases on the other recorded feature revisions;
these are retained as historical evidence, not relabelled as the latest tree.

Strict mypy passes 784 source files; all 11 repository gates, both formatter/lint
tool versions, consumer AST equality and unchanged fixture-byte checks pass. The
last delivery commit adds only the blockade test and this evidence, with no
production-code change. Latest remote full CI/GPU, applicable checks, repository
rules and this PR's separate merge authorization remain mandatory. No scientific
budget or legacy assertion was loosened, and full M3 remains open.

## Merged bounded deliveries

The following records supersede pending-PR wording above; earlier local runs
remain historical evidence for their original revisions. Each final head had
41 successful checks and two unchanged manual-only GPU resource-pool skips,
with successful complete CI/GPU workflows before merge. Each merge's Main smoke
also succeeded. A PR merge or green reference is not full M3 acceptance.

| Delivery | Final head and merge | Complete CI / GPU | Post-merge Main smoke |
| --- | --- | --- | --- |
| [#707 native CPU evolution and plan 1.7](https://github.com/flagos-ai/FlagQuantum/pull/707) | 4f8523cd23 / ac092b003f | [39722409464](https://github.com/flagos-ai/FlagQuantum/actions/runs/39722409464) / [39722409290](https://github.com/flagos-ai/FlagQuantum/actions/runs/39722409290) | [39725500564](https://github.com/flagos-ai/FlagQuantum/actions/runs/39725500564) |
| [#714 native control training](https://github.com/flagos-ai/FlagQuantum/pull/714) | 5174ced278 / 29a2fdc2a9 | [39727069900](https://github.com/flagos-ai/FlagQuantum/actions/runs/39727069900) / [39727069929](https://github.com/flagos-ai/FlagQuantum/actions/runs/39727069929) | [39733415364](https://github.com/flagos-ai/FlagQuantum/actions/runs/39733415364) |
| [#715 commanded transport view](https://github.com/flagos-ai/FlagQuantum/pull/715) | a1dd1f3ded / 7e781ba951 | [39729165240 attempt 2](https://github.com/flagos-ai/FlagQuantum/actions/runs/39729165240/attempts/2) / [39729165411](https://github.com/flagos-ai/FlagQuantum/actions/runs/39729165411) | [39740301406](https://github.com/flagos-ai/FlagQuantum/actions/runs/39740301406) |

Training uses the existing CPU Lindblad plan/run path and PyTorch Adam for two
interval amplitudes. Its 128-interval frozen-unitary value/gradient reference
retains the original 1e-10 budget; the initial 64-interval gradient difference
2.28e-10 was fixed by refinement, not by relaxing the assertion. The SVG/PNG view
uses Core commanded_position and native source timings, not a duplicate geometry
or solver. It does not infer presence, trap activation, heating or fidelity.

The first #715 CI attempt retained 11 failed and five cancelled checks following
Action/source downloads and job timeouts. Completed workflow attempt 2 reran
failed/cancelled jobs on the same head and succeeded without code, workflow,
timeout, scientific-budget or protection changes; the negative attempt is not
reported as a pass.

The [terminal instrument proposal](../development/API_CHANGE_PROPOSAL_083_TERMINAL_INSTRUMENT_SAMPLING.md)
was documented by #716 and explicitly approved separately on 2026-10-23. Its
[authorization and baseline](FQ-TERMINAL-INSTRUMENT-SAMPLING-20261023.md)
cover the next mathematical sampling boundary; they do not retroactively turn
081's deterministic presence effects into sampled detection or close remaining
M3 event, feedback, heating, hardware or scalability gates.
