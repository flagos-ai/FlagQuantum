# Native fermionic science boundaries

Status: implemented and submitted for maintainer review in PR #1109 under the
user-authorized migration and subsequent publication instructions.

## Decision

Deliver the selected OpenFermion capabilities on one feature branch with
reviewable commits. The original local-only delivery restriction was superseded
by the user's subsequent instruction to publish PR #1109. Publication does not
authorize merging. Protected stable exports and their snapshots remain unchanged.
Existing fermionic algebra remains authoritative; its public type identity,
constructor and import path remain unchanged.

Introduce a complex Pauli value in Core because Observable intentionally accepts
only real coefficients. New scientific algorithms consume the existing public
fermionic terms and the Core Pauli value. Observable conversion belongs outside
Core. Mathematical decomposition returns small rotations or factors; a separate
algorithm module lowers them to ordinary native circuits. No new runtime, IR,
backend registry or process-global encoding setting is introduced.

The first complete path is fermion input, explicit JW/parity/BK encoding,
encoded reference occupation, native measurement, and a numerical energy check.
Chemistry, symmetry and circuit synthesis extend that path independently.

## Dependencies

Core Pauli algebra uses the standard library only. Native scientific algorithms
use the existing PyTorch requirement. External chemistry imports PySCF only
inside the explicit driver boundary and returns native problem data. No migrated
production module imports OpenFermion or Cirq. The existing optional Cirq adapter
is unrelated and remains available.

Keep integral conventions, electron counts, orbital ordering, energy offsets,
encoding width, symmetry sector and output layout explicit. Static tensor
preprocessing rejects autograd inputs; differentiable evolution retains tensor
time parameters. Unsupported combinations fail before circuit execution.

## Compatibility and replacement

Reuse the existing FermionOperator and frozen-core formulas rather than copying
them into a parallel package. Preserve the existing hermitian-only JW/parity
APIs. New general encodings return complex Pauli values. Multiple encodings must
run through one unchanged energy consumer, demonstrating replacement at the
encoding boundary. Old public API tests and snapshots are checks, not files to
regenerate to approve a change.

## Acceptance

Use direct Fock-space action and determinant amplitudes as independent small
references, matrix reconstruction for decompositions, exact matrix exponentials
for evolution, and isolated imports for dependency boundaries. Comparisons
between encodings align their basis maps. Sector reduction compares the chosen
sector. Controlled operations require exact phase, not phase-insensitive fidelity.

Publish support only for combinations actually verified. This work makes no
distributed, accelerator, full-OpenFermion-parity or release-certification claim.
