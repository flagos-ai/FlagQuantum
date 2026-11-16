# Target-position dispatch source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `apply_gate_to_lth_target.ipynb` | `7b733722e6587cd31d805804d46e66f18f83a53a` | Q-0582 |
| `apply_gate_to_lth_target.py` | `77f70157ced3fcd8195951f16a34d16a8cfd709a` | Q-0583 |
| `apply_gate_to_lth_target_test.py` | `79e770fc3b13e4cfd0e8c4ffa21a14a8f100bee6` | Q-0584 |

The [native tutorial](../../examples/notebooks/apply_gate_to_lth_target.ipynb)
covers the complete original 13-cell sequence. It shows the original
two controls, three-bit selection with bound four, four target positions,
signature and wire labels, actual primitive compilation, and the six direct
call identities. The circuit runs through `circuit.append(op)`.

The compiler's target-position binding composes the shared unary traversal
with the selected physical target. Selection axes are big-endian
individually and their combined index is row-major. Each child receives
the flattened index, a positive predicate, and the exact target wire.
The caller supplies initialized parent work and distinct classical slots.
Unselected targets retain their input, and the callback supplies the
selected child and its adjoint when reverse traversal is requested.
The Pauli entry uses that binding and elides identity emission. The source
call graph still counts both controlled identity leaves.

## Original cases and execution

The entire locked suite yielded 13 passes including the independent
notebook and one expected failure for callable example serialization.
The native suite covers the width/extent pairs (3, 5), (3, 7) and (4, 5).
It compares the original `make_on` inferred source wire order and
signature to the native operands, preserves the source diagram labels,
and walks the source primitive stream with measurement feedback for
the controlled X and odd-Z examples. The two-control tutorial fixture
checks a coherent superposition of all 256 valid basis rows in all 16
raw measurement branches. The reverse traversal restores the state.
A controlled H/X callback on a nonmonotonic target bank adds a
non-Pauli witness with exact source operands, both measurement
outcomes, and complete complex output including clean work.

| Tutorial fixture | Emitted forward primitives | Declared source model |
| --- | --- | --- |
| I/Z/I/Z, two positive controls | 16 T/T†, 27 CX, 6 CZ, 4 measurements, 4 resets | 4 And, 9 Clifford, 4 measurements, 16 T equivalents |
| Wire width | 13 reserved, including four clean work wires | 12 peak live wires |

The extra reserved wire comes from the selection register's three physical
bits and its four-address extent. The third address level contributes no
live predicate on this fixture. The decorated tutorial graph and independent
cost query are verified separately from emitted counts.

## Remaining source-file boundaries

Q-0582 and Q-0584 are accepted. Q-0583 retains arbitrary foreign gate
objects and foreign serialization as open compatibility boundaries.
Native callers provide controlled child recipes and explicit wire
bindings. The locked source itself rejects symbolic iteration extents
when resolving target registers or diagram labels. The package
initializer Q-0581 remains open while its other exports are audited.
