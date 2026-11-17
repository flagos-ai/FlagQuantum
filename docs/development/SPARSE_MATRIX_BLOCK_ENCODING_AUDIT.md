# General sparse matrix block encoding source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.
The frozen files under `qualtran/bloqs/block_encoding/` are Apache-2.0,
copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0296 | `sparse_matrix.ipynb` | `2a85ed895beeee25d508e68a093af307ca9f1826` | Closed by the full native tutorial. |
| Q-0297 | `sparse_matrix.py` | `cacbdab733f0be4f0f28bf2a749e1bad79987bfd` | Symbolic oracle composition and the original concrete primitive streams verified. Foreign oracle binding and arbitrary measured inputs remain open. |
| Q-0301 | `sparse_matrix_test.py` | `afcb0771b405ed26bfce7827d70d81c3d7f7a0d8` | Closed with the original stress, random entry, banded index and Vlasov fixtures. |

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
All tested work bits clear. The source primitive stream replay covers
these variants in both directions. Complete banded histories remain open.

The unchanged original non-notebook run has 23 passes, five mode skips
and seven expected failures. The unchanged 13-cell original and the
15-cell native tutorial pass independent kernels. The tutorial preserves
four signatures, including the original symbolic shape, and the original
six-call graph. The new append example checks the uniform projected
matrix and exact inverse. The concentrated native run passes 193 cases.

## Original random, banded and Vlasov tests

The native signature declares the same packed system and ancilla order,
including the original symbolic fixture. The symbolic declaration can
be queried without binding wires or constructing data.

The eleven explicit tables, both banded pressure fixtures and the
Vlasov example now execute through append in both directions. For each
case, the locked tensor supplies the whole logical operator. The tests
check the projected input matrix at the original approximation tolerance
and compare all populated complex amplitudes of a coherent input across
the full logical space. All private wires return to zero on the all-zero
and alternating measurement histories. Both directions also match the
independent source QEC category vector and peak. The sparse branch
evaluator uses the exact three-CNOT SWAP identity only for evaluation.
Emitted gate counts are taken before that conversion.

The original three-column top-left example matches the projected matrix
and full coherent outputs at every measurement history in both
directions. The standalone three-bit banded oracle covers both full
operators, all 24 valid index/ordinal pairs and every small measurement
history. The complete enumerations for these bounded cases complement
the sampled long histories above.

| Vlasov emission | Locked logical model |
| --- | --- |
| 376 T/T adjoint, 94 measurements, seven controlled Ry, 19 reserved wires | 94 And, 94 measurements, 11 rotations, 413 Cliffords, 19 peak qubits |

The seven controlled Ry are the explicit entry rotations. Other
rotations include those from uniform preparation. No T synthesis cost
is inferred from the emitted rotation instructions. Both Vlasov
directions match all 2,096 instructions, including operands, rotation
angles and feedback for each of the 94 measurements. Branch execution
covers representative histories. The unchanged source slow run passes
all four tests.

## Primitive stream replay

The frozen Soquet graph provides stage order and operand wiring for the
original uniform, explicit entry, padded top-left, banded and stress
cases. The source oracle's allocation and free bind the caller's angle
word for both QROM passes. The replay uses the independent uniform
preparation, QROM, Add and signed AddK streams. Swaps and controlled
rotations preserve the original operands and angles. Measured stages
share physical workspace across calls, with classical slots bound to
the owning stage. Each wire is checked in dependency order.

## Symbolic oracle costs

The symbolic estimate composes the existing uniform preparation, index
and entry oracles, and SWAP through the six source calls. Top-left index
oracles contribute zero gates at width `2*n`. Uniform entry contributes
one rotation at width `2*n+1`. Banded index oracles compose the existing
Add and signed AddK formulas and follow the locked symbolic peak fallback.
Callers can supply independent directional formulas for other oracles.

For the original symbolic width `n` and full support, both directions
give `6*n-2` And operations and measurements, five rotations and peak
`2*n+1`. The modern Clifford count is `33*n+6`. The legacy Clifford count
is `45*n+2`. Independent symbolic sparsity also retains the unused
high-order diffusion prefix in its AutoPartition peak. A symbolic index
width with band size one has peak `Max(n+4, 2*n+1)`.

The source singleton diffusion omits its target register, and its
AutoPartition refuses the mismatched ports. Numerical symbolic binding
and foreign oracle object binding remain open. The source can also reject
some concrete-index counts with unresolved width at oracle construction
when its sparsity bound comparison has no truth value.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_sparse_matrix_block_encoding.ipynb
```
