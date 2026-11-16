# BlackBoxPrepare source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `state_preparation/__init__.py` | `024a57c221ae73f95645dbaa4247f423b037a932` | Q-0690 |
| `black_box_prepare.py` | `42aacba78eb58628d93eee0db860754827d2f42e` | Q-0691 |
| `black_box_prepare_test.py` | `8dc0acab59408715563bfd651249bc516162fea5` | Q-0692 |

Run `python -m examples.algorithms.hubbard_lcu_catalog` for the original
packed 2×2 Hubbard PREPARE. The example uses `circuit.append(op)` to
append the packed preparation alone, with the 10 through wires and two
caller-supplied clean scratch wires.

## Original source and edge cases

The original fixture wraps `PrepareHubbard(2, 2, 1, 4)`. The selection
port packs `U, V, p_x, p_y, alpha, q_x, q_y, beta` in that order. Junk
packs the two-bit `temp` word. The source has one AutoPartition child.
The native wrapper preserves the packed order and normalization across
both directions. It emits the same gate stream as the named binding.
The named Hubbard test checks the exact source parameters, underlying
gate sequence, operands and measurement feedback.

The packed source was simulated with its original operations. The
undecomposable ModAddK leaf uses its source classical permutation as an
explicit matrix. All eight native measurement histories match the full
complex source state and restore both scratch wires. The inverse
restores the all-zero input on those histories.

The original dummy oracle packs its named one-bit selection and
five-bit junk. The one-bit public signature uses bit encoding, while
its group descriptor stays opaque. The tests check packed port names,
order, bind-time operand routing, and absent normalization for an
emitter without coefficient data. Zero-width descriptors are retained
and omitted from the public signature. A zero-width selection returns
the other bound ports without invoking the child. A nonempty selection
and zero junk binds only the selection child. The original suite
reported four passes, one decomposition-only count skip, and one
expected failure for serialization.

| 2×2 Hubbard body | T/T† | CX | Measurements | Declared resource categories | Wires |
| --- | ---: | ---: | ---: | --- | --- |
| Native packed, either direction | 47 | 66 | 3 | — | 12 reserved |
| Source model | — | — | 2 | 5 CSwap, 2 And, 11 Clifford, 4 rotations | 12 logical |

## Package and remaining boundaries

The four exports in Q-0690 point to the following native owners.
Their individual source-file acceptance remains tracked under their
own parent items.

| Source export | Native owner |
| --- | --- |
| `PrepareUniformSuperposition` | `compiler/registers/preparation/uniform_preparation.py` |
| `StatePreparationAliasSampling` | `algorithms/primitives/alias_preparation.py`, dense table |
| `SparseStatePreparationAliasSampling` | `algorithms/primitives/alias_preparation.py`, sparse table |
| `StatePreparationViaRotations` | `algorithms/primitives/rotation_preparation.py`, rotation program |

Q-0692 is covered. Q-0691 remains open for general symbolic packed
layouts and importing or serializing external oracle objects. The
native wrapper binds concrete named emitters through explicit ports.
Q-0690 remains open for its exported owners.
