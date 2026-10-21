# Matrix expectation output requests

## Authorization and delivery

The repository owner explicitly approved proposal 080 on 2026-10-21 with:
"Approve proposal 080 and advance M2" (English rendering), in response to the question covering its
five decisions and the sequence "replaceable contract baseline, then
implementation". [Proposal 080](../development/API_CHANGE_PROPOSAL_080_MATRIX_EXPECTATION_OUTPUTS.md)
owns the exact accepted contract. Document-only #663 had already merged; its
merge is not the authorization recorded here.

This integration change establishes test consumers and a test-only request
adapter. The adapter delegates numerical planning, execution, options,
parameter binding and serialization to the existing owners. It is not an
implemented public matrix expectation API. Its only replacement point is the
consumer fixtures; the real API must replace them without changing physical,
gradient, compatibility or refusal assertions. Remove the test-only adapter
after replacement rather than retain a second implementation.

The production package, current protected signatures and existing
`observable-outputs-v1-candidate.json` remain unchanged in this baseline.
That candidate continues to enforce the current Pauli API. The implementation
change must record only the approved factory/field widening and its behavioral
rules in that existing candidate/checker, alongside the real replacement. Do
not create a second output registry or blanket snapshot exemption.

## Approved scope and compatibility

- Widen existing `expectation` and OutputRequest.observable to accept a
  full-space dense torch.Tensor, retaining the Pauli form and all other fields.
- Use the current CPU Lindblad plan/run and named results. Reject unsupported
  circuit/remote paths before execution or submission.
- Capture matrix values at planning without detaching live gradients; retain
  existing numeric plan versions and the existing trajectory-gradient rule.
- Refuse explicit/generated Lindblad name collisions with the existing
  `duplicate_output_name` category rather than dropping an output.
- Compare tensor requests by tensor object identity and matching metadata;
  preserve old Pauli/None equality and hashes. Numeric plan identity remains
  value-based under the existing encoding.

There is no new root export, numerical engine, plan version or result envelope.
The old signatures and Pauli data remain protected until the actual approved
replacement lands. Unique output names need no migration. A formerly
colliding request must supply distinct names; the correction and new tensor
refusals require release-note documentation in the implementation PR. No
existing API name is deprecated or scheduled for removal by this change.

The fixed compatibility payloads are captured from unmodified production at
main `01280756b2be8b97a7963fbdeb52c969621d078b`, before any 080 implementation.
They include qubit, mixed-product, scheduled-product and labelled static and
scheduled plans at both supported precisions. Do not regenerate them to repair
a compatibility failure.

The first 101-point CN gradient check exceeded its 1e-6 continuous-reference
budget (error 1.2597258e-6). An independent exact discrete CN rotation matched
autograd within 1.7e-14 over 51/101/201-point grids, while the continuous-error
ratios approached four. The accepted consumer uses 201 points (200 intervals),
with a retained 1e-6 budget and an explicit second-order refinement assertion.
No solver, tolerance or scientific assertion was weakened to pass the check.

## Runnable current user path

```bash
python -m examples.mixed_dimension_dynamics
```

Expected CPU output:

```text
Three-level leakage probability: 0.11757891
Coupling derivative: 0.32210884
Spin-mode dimensions: (2, 3)
Probability of |spin=0, occupation=1>: 0.11757891
```

The baseline additionally checks the same leakage as a matrix expectation,
with gradients and named results, through existing matrix execution. The
proposed public `fq.expectation(projector)` spelling is not yet available.

Each follow-up PR requires its own applicable merge/queue authorization and
latest full validation. This approval does not extend previous PR merge grants
or promote native optical, topological, accelerator or distributed support.

## Simulation implementation

After baseline #669 merged at `362b2184add084069176e6073d1c7bfe2fcfdd9a`,
the Simulation implementation replaces the test-only planner adapter with
`flagquantum.lindblad.plan`. Only the pending Core request producer remains a
test fake; its concrete type is injected into the planner's existing imported
class boundary in tests. The production request validator is not broadened.
Physical, gradient, replay and fixed compatibility assertions are unchanged.

The existing output normalizer resolves all effective names before accepting
an output, refuses generated/explicit collisions, and clones tensor operators
without detaching their graph. Numeric validation, scalar binding, the sealed
plan, all three integrators, results and serialization remain with their
existing owners. There is no new API, schema, registry or solver. The public
matrix constructor and circuit/remote refusal checks are the subsequent
Core-owned implementation, not a capability claimed by this step.

A public user example is `python -m examples.lindblad_output_names`, with:

```text
Polarization: 1.0
X expectation: 0.0
Collision refused: duplicate_output_name
```

Unique names keep their existing results. A colliding request must provide
unique names; the release notes describe this approved behavioral correction.
