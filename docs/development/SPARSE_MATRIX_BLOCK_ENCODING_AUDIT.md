# General sparse matrix block encoding source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.
The frozen files under `qualtran/bloqs/block_encoding/` are Apache-2.0,
copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0296 | `sparse_matrix.ipynb` | `2a85ed895beeee25d508e68a093af307ca9f1826` | Closed by the full native tutorial. |
| Q-0297 | `sparse_matrix.py` | `cacbdab733f0be4f0f28bf2a749e1bad79987bfd` | Open for generic symbolic oracle models/binding and external objects. |
| Q-0301 | `sparse_matrix_test.py` | `afcb0771b405ed26bfce7827d70d81c3d7f7a0d8` | Open for the remaining original stress, random entry and Vlasov fixtures. |

## Oracle and block ownership

Row and column oracles act on big-endian ordinal and index words. The
top-left oracle binds the ordinal directly. The banded oracle applies
Add followed by signed AddK and reverses those calls on adjoint. The
entry oracle receives the flag, row and column in that order. Uniform
entries use one Ry with angle twice arccos of the entry. Explicit entries
use the source NumPy angle transform and QROM integer conversion, then
the ordered big-endian controlled Ry sequence, then the self-adjoint
QROM again. Private QROM load and traversal wires return to zero.

The composed block binds an ancillary flag followed by the ordinal word
and has no logical resource port. It prepares the low ordinal bits,
calls the column and entry oracles, swaps the index words, applies the
row adjoint and unprepares the ordinal. The reverse path inverts the
stages and preserves their operands. The caller supplies clean temporary
work and classical slots, reused between sequential stages. Sparse
oracle logic belongs to Algorithms and delegates its gate kernels to the
existing Compiler owners. Width and sparsity checks require a concrete
valid bounded ordinal for physical binding.

## Concrete original variants

| Fixture | Physical emission | Independent locked source model |
| --- | --- | --- |
| Four-by-four uniform 0.3 | Four H, two SWAP, one Ry, no temporary wires or measurements | Six Clifford, one rotation, peak five |
| Two-by-two explicit, 10-bit angles | 16 T/T-dagger, four measurements, 11 temporary wires | Four And, 45 Clifford, ten rotations, four measurements, peak 14 |
| Eight-by-eight banded, half-bandwidth one | 72 T/T-dagger, 18 measurements, five temporary wires | 18 And, 129 Clifford, five rotations, 18 measurements, peak 12 |

Both directions of the three original concrete variants match the
source six-call graph totals, QEC categories, peak and full complex
reference output. The uniform stream checks the explicit SWAP operands,
and the explicit entry fixture checks all measured histories. The
banded fixture checks three representative histories in each direction.
All tested work bits clear. Complete banded histories and full
primitive flattening for the two measured composite variants are open.
Their direct Compiler dependencies retain independent source call and
gate verification.

The unchanged original non-notebook run has 23 passes, five mode skips
and seven expected failures. The unchanged 13-cell original and the
15-cell native tutorial pass independent kernels. The tutorial preserves
four signatures, including the original symbolic shape, and the original
six-call graph. The new append example checks the uniform projected
matrix and exact inverse. The concentrated native run passes 193 cases.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_sparse_matrix_block_encoding.ipynb
```
