# Literal nested constant controls

Reference: unchanged Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, addition.py, bitwise.py,
_infra/controlled.py and _infra/adjoint.py. Their four frozen blobs are asserted.

## Recursive extension (2026-11-28)

The [ECC dependency closure](ECC_NESTED_CONTROL_DEPENDENCIES.md) extends these
same owners through the literal Toffoli and ControlledViaAnd callbacks. Scalar
prefixes recurse without the original two-control cap; two through five
controls have complete source stream, complex branch, clean-work, resource and
constructor evidence. The original two-control Programs and resource table
below remain unchanged. Deeper loads own measured predicate work per loaded
bit, while Always(Add/Adjoint(Add)) remains uncontrolled. This extension closes
shared constant-load dependencies; the remaining ECC/modular parents are not
marked complete by this audit.

## Ownership and interface

The shared Bits owner constructs `Controlled(Controlled(XorK))` using the
existing positive XOR owner and exact hierarchical control propagation. The
Arithmetic owner constructs nested `Controlled(AddK)` with five literal calls:
allocate, controlled load, Always(Add), controlled unload and free. Its exact
inverse changes the unconditional Add owner while retaining matching loads.
Independent scalar ports are `ctrl2`, `ctrl`, followed by MSB-first `x`.
Caller-provided patterns are already encoded. Private constant and carry work
is zero-initialized and released; no new conjunction work replaces source
atomic Toffolis. Grouped-predicate control is a different source recipe.

The Formats catalogue owns per-compilation constructor provenance. Its AddK and
XorK handlers use the original native factories and capture their validated
encoder output, preserving one authority for signed/unsigned normalization.
Known positive control and adjoint views retain that provenance. Bare Add
adjoints used by Always reuse the existing exact adder owner. Zero, one and two
nested positive scalar controls are covered. Negative and concrete grouped or
packed predicates reuse the existing three-call predicate owner; full source
primitive streams, branch amplitudes and source resource models are compared.
The observer flattens the original composite before export so that temporary
RIGHT outputs do not exhaust a physical lifetime pool during partial export.
The original deeper-positive nesting restriction is superseded by the recursive
extension above. Symbolic data-width predicate bodies remain outside this
scoped catalogue, without being relabeled as upstream failures.
Unknown same-signature children
require a separately supplied owner, never signature-based semantic inference.
Symbolic constants retain true declarations and fail physical binding.

## Source calls, operators and erase histories

Word controls compare every available source call and operand, the actual
atomic Toffoli primitive stream and the complete complex operator. They allocate
no work and use no measurements at two controls. Deeper predicate work and
measurement histories are documented in the recursive extension. Constant
controls compare all new higher
calls, literal primitive dependencies and feedback slots against unmodified
source decomposition. Their Always(Add/Adjoint(Add)) subtree is reused only
after a complete native definition certificate. Independent Split(a)/Split(b)
creation order can differ in the existing exact inverse; full per-wire primitive
dependencies, rather than a false global schedule identity, are compared.

Unsigned, signed negative, Montgomery, zero-pattern and wider-word inputs are
covered. Arbitrary coherent complex weights span all logical control/data
values. Every small raw measurement history is checked with its unnormalized
branch amplitude and zero private output support. Native DynamicCircuit append
and trajectory execution consume the same IR. Actual source libraries at
depths zero and one roundtrip and resolve with SDK modules unavailable.

A first word-matrix test incorrectly reserved a large generic clean pool and
attempted to simulate its entire unused capacity. The process was terminated;
no passing result is claimed for that attempt. The corrected no-work word
binding supplies no clean pool and asserts its actual logical qubit extent.

## Measured primitive inventory versus source model

The three-bit encoded pattern 3 under two scalar controls has these literal counts. Native CCX
is an atomic primitive; a source default T estimate is a separate convention.

| Operation | Emitted primitives | CCX | T / T-dagger | Measurements / resets | Source model Toffoli / And | Source default T estimate | Source qubits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Nested XOR | 2 | 2 | 0 / 0 | 0 / 0 | 2 / 0 | 8 | 5 |
| Nested constant add | 47 | 4 | 4 / 4 | 2 / 2 | 4 / 2 | 24 | 10 |
| Nested exact inverse | 47 | 4 | 4 / 4 | 2 / 2 | 4 / 2 | 24 | 10 |

The constant rows also emit H=6, CX=21, S=2 and CZ=2. All four original control
sectors retain their absolute complex phases; no global-phase quotient or
sampling-only correctness claim replaces these checks.

## Full ECC dependency batch

This closes the shared nested loader, constant component and its scoped
constructor context. It does not close the full ec_add.py/ec_add_r.py parents.
Actual controlled window and ECAdd higher bindings, controlled modular and
comparison contexts, and ordinary symbolic-modulus arithmetic dependencies
remain explicitly open in NEXT_G2.md. The native source-controlled positive
ECAdd graph has 126 distinct reference nodes, 67 with available bodies in the
recorded three-bit profile; directional refusals apply at their actual leaves.
