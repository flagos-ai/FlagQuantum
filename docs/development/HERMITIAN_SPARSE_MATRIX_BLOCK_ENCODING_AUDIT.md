# Hermitian sparse matrix block encoding source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.
The frozen files under `qualtran/bloqs/block_encoding/` are Apache-2.0,
copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0298 | `sparse_matrix_hermitian.ipynb` | `117edcce9169b81c911d40885946e193e829d438` | Closed by the complete native tutorial. |
| Q-0299 | `sparse_matrix_hermitian.py` | `95b077d8eb6ac7deb814c910276b4a1d9d603bee` | Symbolic composition, structured controls, padded primitive flow and independent user oracle replacement verified. Automatic conversion of raw foreign oracle objects remains open. |
| Q-0300 | `sparse_matrix_hermitian_test.py` | `367de6a3ae0ac011a16a5cc13b63dbb77cd7ba53` | Closed for the original runnable scenarios and their notebook. Source fixture mode limits are recorded below. |

## Interface and direct dependencies

The Algorithms owner composes existing row/column and entry oracles with
the Compiler's uniform preparation, SWAP and exact Fredkin kernels.
The square-root entry oracle binds flag, big-endian row and big-endian
column, in that order. Its uniform specialization emits
`Ry(2 * acos(sqrt(entry)))` on the flag and preserves the oracle precision.
The block separately reports its declared error and sparsity scale.

The through ports are the system word and an ancilla packed as flag,
big-endian ordinal, flag. Signal preparation is identity on the initialized
ancilla. Diffusion acts on the low ordinal bits. The ordered calls are
unconditional diffusion, column and square-root entry, the index swap and
flag swap, then the three preparation adjoints in reverse order. The
optional positive control binds only the two swaps. The reflection is
self-adjoint. The block owns no logical resource port. The caller supplies
clean temporary quantum wires and classical slots, which the block reuses
between sequential calls.

## Concrete execution and resources

| Fixture | Emitted primitives and temporary work | Independent locked source model |
| --- | --- | --- |
| Full 4-by-4 uniform, ordinary | Four H, two Ry and three SWAP, no temporary wires or measurements | Seven Clifford, two rotations, peak six |
| Full 4-by-4 uniform, positive control | Four H, two Ry and three exact Fredkin streams (21 T/T-dagger), no temporary wires or measurements | Three CSWAP, four Clifford, two rotations, peak seven |
| Padded 3-by-3 in 8-by-8, ordinary | 40 T/T-dagger, ten measurements, four temporary wires | Ten And, 74 Clifford, six rotations, ten measurements, peak 12 |
| Padded 3-by-3 in 8-by-8, positive control | 68 T/T-dagger, ten measurements, four temporary wires | Four CSWAP, ten And, 70 Clifford, six rotations, ten measurements, peak 13 |
| Padded 3-by-3 in 8-by-8, equality to two on a two-bit unsigned word | 72 T/T-dagger, eleven measurements, five temporary wires | Four CSWAP, eleven And, 71 Clifford, six rotations, eleven measurements, peak 15 |

The full-support fixtures check the ordered stages, direct call counts,
bound primitive operands, all complex matrix entries, the inverse, and the
native IR round trip. The padded fixture checks every projected matrix column
at the all-zero history and every ten-bit measurement history on a coherent
complex input for each control mode. Its expected full vector comes from the
locked source tensor contraction with the work wires appended in zero. The
test covers the primitive measurements and feedback through branch replay.
The padded ordinary and positive-control fixtures also independently
replay the locked Soquet graph through the diffusion, rotation and exchange
stages. Each wire sees the same primitive operands, parameters and
classical feedback. Stage-local temporary slots retain their original
binding throughout the comparison.

## Structured controls

`make_controlled_on` wraps the existing positive binding through the
shared predicate executor. One negative control emits X around the
positive block. Multi-bit controls compute the equality predicate once,
run the positive block and erase the predicate. Control work and
measurement slots stay disjoint from the child workspace.

The full-support fixtures compare one negative bit, a mixed bit array,
a typed unsigned word, and two typed control groups. Each verifies
the locked primitive stream, source model, complete complex output and
every measurement history. The controlled reflection is self-adjoint.
The padded typed control additionally enumerates all 2,048 histories
on a correlated complex input with active and inactive control values.
It matches every output amplitude and returns all five temporary wires
to zero. Raw foreign oracle objects still need native conversion at the
optional Ecosystem boundary; native user implementations are covered below.

## Independent oracle replacement

`examples/algorithms/custom_sparse_oracles.py` implements both existing
structural protocols without inheriting the built-in oracles. Its column
map is `ordinal XOR index XOR all-ones`. Its nonuniform square-root entry
is an Ry rotation by `base + delta * (low_row AND low_column)`. The And
uses one clean caller work bit, erases it with measurement and feedback,
and exposes one classical slot. Forward and adjoint callbacks bind the
same through words and restore scratch.

User extensions of the pinned abstract source oracles implement these
operations independently in the test fixture. The original Hermitian
consumer and its source files are unchanged. Tests follow all source
Soquet connections down to primitive gates, operands, parameters, result
slots and classical feedback. They also compare the independent logical
models, including zero, Clifford, T and generic rotation classification.
Widths one and two, ordinary and positive-controlled blocks, match every
complex amplitude against the original tensor contraction and the analytic
projected matrix. The input includes nonzero ancilla and spectator
entanglement. All four erasure histories give the same normalized output,
and all sixteen histories of a block followed by its adjoint recover the
input. A prior measurement at slot zero remains available after the
oracle reuses slot one. The borrowed work bit returns to zero.

| Index width / control | Emitted T/T-dagger | Emitted Ry / measurements | Source logical model | Source peak |
| --- | --- | --- | --- | --- |
| 1 / ordinary | 8 | 6 / 2 | 2 And, 14 Clifford, 6 rotation, 2 measurement | 5 |
| 1 / positive | 22 | 6 / 2 | 2 And, 12 Clifford, 2 CSwap, 6 rotation, 2 measurement | 6 |
| 2 / ordinary | 8 | 6 / 2 | 2 And, 21 Clifford, 6 rotation, 2 measurement | 7 |
| 2 / positive | 29 | 6 / 2 | 2 And, 18 Clifford, 3 CSwap, 6 rotation, 2 measurement | 8 |

This closes the missing native callback conformance evidence. It does not
automatically translate an arbitrary external library object. The existing
Cirq importer rejects the original And erasure circuit: it requires line
qubits and terminal measurements, and rejects its conditional feedback
and reset operations. That conversion gap belongs to Ecosystem, not to a
new oracle protocol or an Algorithms dependency on the reference package.
Q-0299 stays open for that concrete foreign conversion requirement.

```bash
python -m examples.algorithms.custom_sparse_oracles
pytest -q tests/unit/test_hermitian_custom_oracles.py
```

## Symbolic declarations and original validation

For the original positive integer symbol `n`, the direct Hadamard leaf
count is `6*n`. Concrete power-of-two preparation dispatch gives four
H at two bits. The symbolic model also reproduces every QEC category and
the peak widths `2*n + 2` and `2*n + 3` for the ordinary and controlled
variants.

The symbolic fallback accepts independent forward and adjoint formulas
for the column and square-root entry oracles. It composes them with
the shared sparse diffusion and both exchanges. The source's Always
wrappers leave the two preparation directions unchanged under the
positive control. Only the exchanges change from Clifford SWAPs to
modeled Fredkins. The peak takes the maximum of the full signature
and the eight child peaks, including the high-order diffusion prefix
inside AutoPartition. Tests compare both resource profiles for full
support, independent symbolic sparsity, a concrete index width with
symbolic sparsity, and fixed and symbolic band sizes. Structured control
fixtures also match the two QEC profiles and symbolic peak. Symbolic
decomposition remains unavailable in the locked source and native binding.

The original concrete and symbolic fixtures, parameter assertions, Hadamard
leaf counts, both tensor fixtures, and direct declarations map to the
native tests and tutorial. The unchanged original non-notebook run has
nine passes, four mode skips and two expected serialization failures.
The skips include unavailable symbolic decomposition and the original
strict `QAny(2)` to `BQUInt(2, 4)` check. Native IR round-trip evidence
applies to concrete circuits. The unchanged 11-cell original and the
13-cell native tutorial passed in independent kernels. The native kernel
rejects reference imports and validates both SVG graphs.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_hermitian_sparse_matrix.ipynb
```
