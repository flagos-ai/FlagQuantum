# Specialized control source file audit

The reference is unchanged Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The native audit verifies all
three original Git blobs before compiling the source notebook fixture cells.

| Frozen asset | Git blob | Ledger item |
| --- | --- | --- |
| `specialized_ctrl.ipynb` | `67c4664b39c2c1043d59312e76d9453c9a294cea` | Q-0560 |
| `specialized_ctrl.py` | `98b64995010da5863218dbf93b39bbffbd600a2f` | Q-0561 |
| `specialized_ctrl_test.py` | `8ed8dd515d1646ac5d9bcbb9e0332f3060b127df` | Q-0562 |

The [native tutorial](../../examples/notebooks/specialized_control.ipynb)
uses stored typed calls and the shared Compiler selectors. A positive
specialization keeps T and T† on the target and controls the central X only.
The two-control wrapper passes both controls through, computes the predicate
on one caller-owned clean wire, calls the positive specialization and
measurement-erases the predicate. Both outcomes preserve the complete
complex state and return the work wire to zero. The native append example
checks the locked source gates, operands, per-wire dependencies and
measurement feedback, including the reset used to reuse the work wire.

The specialized adjoint selects the positive controlled body before applying
the exact stored adjoints. Its primitive stream matches the locked
`T, CNOT, T†` source, with the phases on the target alone. The generic
wrapper in the original notebook constructs successfully. Its decomposition
and call graph fail because the uncontrolled branch passes `ctrl` and
`target` to X, which requires `q`. The native generic example implements the
intended controlled phases and CNOT and has the same complex amplitudes as
the valid specialized source. The original error is reproduced separately.

| Tutorial path | Emitted primitive counts | Locked QEC model |
| --- | --- | --- |
| Two-control specialization | 6 T/T†, 7 CX, 3 H, 1 S, 1 measured CZ feedback, 1 measurement, 1 reset | 2 T, 1 And, 2 Clifford, 1 measurement; peak 4 |
| Positive specialized adjoint | 2 T/T†, 1 CX | 2 T, 1 Clifford, 0 measurements |
| Native generic adjoint | 2 controlled phases, 1 CX | Original decomposition fails before a model can be expanded |

The original tests pass 30 cases plus the original notebook. Native
checks map all 16 grouped-control cost cases and all 12 custom-port
decomposable cases. The latter preserve the two uses of the opaque
specialized child and each predicate's separate compute and erase. Original
atomic dispatch and adjoint leaf identities are checked separately. The
opaque fixtures test graph structure or declared costs. Their placeholder
bodies carry no numerical interpretation. Existing regressions cover
actual returned-adder bindings, packed control, named selection, adjoint
masks, and direction-aware fallback with executable children.

Q-0561 remains open for generic AutoPartition return protocols and
unsupported typed register shapes and dtypes. Earlier audits also recorded
source callback failures while appending a control with `np.concatenate`
and when the selected control port is not first. Those failure cases retain
their original classifications and source blobs.
