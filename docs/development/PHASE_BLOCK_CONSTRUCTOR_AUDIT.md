# Complete Phase constructor and scalar-carrier dependency disposition

Reference: Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, Google LLC, Apache-2.0.
The unchanged source and test blobs are
`d7e6b03237f63238e511a4e7aa5056808a646593` and
`c41738844b4eb08517b4c3245d7cc47f1d6bdf1b`. Both were checked against the
GitHub contents at that exact revision and are asserted in the reference tests,
together with Qualtran 0.7.0 and the actual Kickmix 0.1 Simulator.

## Interface, storage and ownership

Phase retains the child's ordered opaque system/ancilla/resource through ports,
omitting zero-width words. Words remain MSB-first and caller-owned. The shared
`phase/phased_block_component.py` composes a zero-port scalar call and an exact
child Program. Its inverse uses explicit child and scalar inverse pairs. The
existing scalar owner allocates, restores and frees its own carrier; the child
retains its private work ownership. Logical scalar call and model remain
registerless even though physical execution uses that carrier.

`formats/legacy_phased_block_factories.py` owns scoped constructor/inverse
recipes. It delegates default fixed-gate Unitary blocks to their existing owner
and requires explicit full native pairs for other children. Resolved definition
identity, surviving arguments and native signatures are retained, with no
external SDK object in Compiler. Native positive child owners are separate
from ordinary inverse ownership, because source control specialization may use
a different graph. Both scalar and grouped controls reuse existing control,
predicate and erasure owners. Invalid payloads and unknown child owners fail
before execution rather than being inferred from signatures or resource models.

Existing Algorithms metadata and public operation entrance are preserved.
The established packed-signature owner already supplies the authoritative
signature. Source alpha/epsilon/signal metadata and callback execution stay in
Algorithms; shared typed composition and constructor resolution stay in
Compiler. Users continue to append operations to the same Circuit.

## Actual call order and existing consumer replacement

Ordinary Phase has its original scalar/block call declarations. The disconnected
source scalar can be visited differently from its insertion index; inverse
comparison therefore identifies the original atomic scalar by its empty
signature, not a guessed first/last inverse position. Parent call/operand
comparison still checks the complete source graph. The source atomic
GlobalPhase has no deeper decomposition, so the native carrier recipe is an
explicit execution boundary, not an upstream gate-by-gate claim.

The source positive-control visitor imposes an actual dependency: controlled
block first, then ZPow on the successor control. Controlling Phase.adjoint()
also visits the inverse block before negative ZPow. Generic
Adjoint(Controlled(Phase)) instead reverses the already-controlled calls. The
native factories preserve both recipes, rather than substituting equal complex
matrices for different source call chains.

This batch also fixes the existing `PhaseBlockEncoding.make_controlled_on`
forward emitter and its model call ordering. It previously emitted scalar then
child, and its test repeated that hard-coded order. It now follows the actual
source control visitor. The callback still supplies the directional child, and
the existing test derives the expected order from the unchanged upstream
decomposition. Complete phase/predicate tensors and every small raw branch still
match. No public export, signature, default, result or serialized vocabulary is
changed.

## Evidence and source boundaries

Tests cover the actual original example at shallow depths 0/1, exact generic
inverse, zero/positive/negative/fractional exponents, H/T non-Hermitian child
directions, nested phases, nonzero auxiliary bystanders, nonlinear symbolic
phase binding through the existing parameter path, and source-import-free
resolve/bind/ordinary append. A separately owned measurement-bearing default LCU
replaces the child without changing the Phase consumer. Full complex logical
operators and work zero are checked; the existing erasure certificate establishes
every raw history for measured children. Small controlled/predicate branch
enumeration is additionally covered by the existing Algorithms suite.

Original depth-2 forward and inverse serialization fail inside AutoPartition's
inhomogeneous tuple-to-array conversion. Tests retain that precise producer
failure before native decoding. No reference, serializer or exception assertion
is patched to claim deeper coverage. Arbitrary external children still require
their registered native constructor and exact ordinary/positive owners. Symbolic
storage retains existing declaration and physical-admission boundaries.

Original non-notebook tests report nine passes and one source serialization
xfail; the earlier unchanged ten-cell source notebook receipt is reused as its
separate execution scope. Scalar,
SU2, Unitary and existing Phase consumer regressions exercise the reused owners.
Actual primitive counts and private/live peaks are recorded separately from
source QEC/logical-peak models: a scalar's zero analytical cost does not erase
the two X/two phase carrier primitives or its physical workspace. Physical
carrier cleanup is proved numerically, separately from allocation release.

Eight directional rows are recorded in the external receipt
`g2-phase-block-constructor-resources-20261125.json`. The ordinary example emits
one H, two X and two phase primitives, with one private/two live wires, while
the source model reports one Clifford and logical peak one. Grouped 10 emits
four T-family, six CX, three H, four X, one S, one atomic CH, one phase and one
measurement/CZ/reset stream, with one private/four live wires. Its source model
separately reports one T, one And, five Clifford, two rotations, one measurement
and peak four. Source atomic boundaries and unsynthesized native phase gates
are retained; model T estimates are not counted as emitted T instructions.

Proposal 119 remains unapproved and its Program scalar operation is not used.
The existing carrier recipe supplies this complete Phase path under current
contracts. Reflection, Chebyshev and other former scalar-dependent roots must
be tested with that owner before being classified as blocked.
