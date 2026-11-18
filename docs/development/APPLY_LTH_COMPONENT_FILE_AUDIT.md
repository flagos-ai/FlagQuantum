# Shared-target indexed component source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `multiplexers/__init__.py` | `25c17b6a77979aac442a70b226c59c138db931a6` | Q-0581 |
| `apply_lth_bloq.ipynb` | `ff8d6499f96132421124c9a8a3b643d064cbd8fd` | Q-0585 |
| `apply_lth_bloq.py` | `438ae361b6805c2f707849ada7c0a4eed04e460f` | Q-0586 |
| `apply_lth_bloq_test.py` | `b98e5bcdff8a4a1b338bd0f51227f5a754ec780f` | Q-0587 |

The upstream initializer contains only a module docstring about coherent
selection by unary iteration. The shared implementation follows that
description: each bounded selection value routes the corresponding
controlled component onto the target, with a clean predicate lifecycle.
[Indexed component dispatch](INDEXED_COMPONENT_DISPATCH.md) documents
the native implementation. The initializer has no import or export
surface to map, so Q-0581 is accepted.

The [native tutorial](../../examples/notebooks/apply_lth_bloq.ipynb)
retains the original T/H/Z/X example, register signature and six call
identities. It appends the shared-target operation and its exact component
adjoint, then prints the primitive compilation. Selectors are big-endian
within each axis. Multiple axes select child recipes in row-major order.
The control and selectors pass through; the caller supplies clean parent
work and the child owns its own work and adjoint recipe.

## Original cases and controlled phase

The four original selector grids have extents `(2,)`, `(3,)`, `(4,)`
and `(2, 2)`. Each direction checks source register metadata, gate
dependencies, operands and measurement feedback. A coherent input covers
every legal control/selection pair and both target amplitudes. The
selected target matrix comes from the locked source leaf. Every
measurement history is projected from the emitted primitives with
zero work required in the complete output vector.

The source's positive controlled T is atomic at this boundary. The
reference reader retains the original operands and maps T or T† to the
equivalent controlled-phase angle `+π/4` or `-π/4`. This makes the
forward and adjoint comparison include the phase instruction, without
changing the Qualtran source or using an inferred decomposition. The
existing indexed-control and control-stack tests independently exercise
specialized controls, composition, restoration and error diagnostics.

| T/H/Z/X example | Emitted primitives | Declared source model |
| --- | --- | --- |
| Forward | 12 T/T†, 22 CX, 3 measurements, 1 controlled phase, 1 CH | 3 And, 12 Cliffords, 3 rotations, 3 measurements |
| Width | 6 reserved wires including 2 work wires | 6 peak live wires |

The declared cost is 45 T equivalents at four T per And and eleven T
per rotation. The primitive stream retains its controlled-phase and
controlled-H leaves; the twelve emitted T-family instructions belong to
the parent unary traversal.

## Remaining source-file boundaries

Q-0585 and Q-0587 are accepted. The locked source suite had 32 passing
tests, with expected failures for the fixture count check and
serialization. The independent source notebook passed as well.
The native test explicitly checks the fixture call declaration.

Q-0586 remains open where the source's direct zero-control value still
activates on one, its repeated multi-bit control builder cannot execute
the declared graph, or its empty inner axes have no valid traversal.
Arbitrary foreign Bloq objects and serialization remain compatibility
boundaries. The native execution interface accepts owned controlled child
recipes and explicit storage.

The new fixture, notebook test and file audit pass strict typing. Standalone
strict typing of the shared source reader reports the same 13 diagnostics
as the baseline, covering Cirq's dynamic types and existing metadata typing.

## Complete directional source capture

The existing named emitter captures the whole source in both directions. The
Ecosystem adapter preserves the default singly controlled T leaf by controlling
the exact Cirq operation returned by the source T gate. Source operands and
adjoint direction are retained. The resulting native snapshots are consumed
through the existing SelectBlockEncoding and Circuit.append entry.

The T/H and T/H/Z/X banks cover no control and both declared control values.
Each original direction is checked on every logical operator column and every
measurement history. JSON round trips, exact phases and clean work are verified.
The locked direct declaration with control value zero still activates on one.
The capture preserves that behavior, and the existing native specialized control
path has separate coverage for negative activation.

The two- and four-leaf banks emit one controlled phase and one CH in each
direction. The traversal contributes four primitive T-family gates per
measurement. The independent indexed model checks source QEC categories and peak
width in both directions. The model prices the controlled T and H as rotations.
Mixed empty axes, repeated multi-bit control and source object serialization
retain the documented boundaries above.
