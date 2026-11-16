# Native bounded unitary matrices

The user authorized native capabilities needed to migrate the five example
categories. This unreleased extension exposes `unitary(matrix, target)` and
reuses the existing checked `qreg.unitary`, matrix validation, indexed views and
absolute-phase-preserving gate decomposition. No serialized opcode, runtime,
Stable Core surface or external dependency changes. Release authorization remains
false; this is the `single_device_fast_path`.

Matrices are finite built-in numeric literals in nested lists/tuples, or an
immutable tuple of numeric tuples bound to a static imported/global name. Source
capture never calls a host matrix constructor or reads dynamic object attributes.
Complex literals preserve their imaginary parts. The shared validator requires a
square unitary of dimension 2**n, with n in 1..5. Literal/global container bounds
are checked before expanding data. Global mutable lists and tensor objects are
refused. Classical parameter matrices are not yet admitted.

A QNum or QArray target, including a contiguous QArray slice, uses logical
most-significant-first matrix order: element zero remains least significant.
An explicit list/tuple of scalar QBits specifies a local view in the same order:
position zero is its least significant matrix bit. This preserves Qmod list
semantics and makes a full array, its full slice and its ordered elements agree. Duplicate locations,
Const/uninitialized targets, malformed matrices and insufficient budgets fail
closed through existing ownership and lowering checks. Mutable gate-only helpers
may apply these operations; new inverse/control helper forms are not admitted.

## Acceptance

- Migrate unitary.qmod and quantum_volume.qmod without source imports in examples;
  preserve the latter's exact complex matrix, three iterations and explicit
  qubit order. Compare originals and natives to independent direct matrix Circuits.
- Verify all matrix columns on small registers, complex coherent inputs, absolute
  phase, offset/sliced wires, explicit-list order and helper composition.
- Reject nonunitary, nonfinite, nonsquare and oversized matrices, mutable globals,
  host calls, repeated/Const/uninitialized wires and resource-budget violations.
- Preserve all 216 original identities and frozen resource records; count complete
  Notebook workflows separately from native quantum-program coverage.
