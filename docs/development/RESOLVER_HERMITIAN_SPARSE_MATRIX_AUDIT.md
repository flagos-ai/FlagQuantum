# Hermitian sparse constructor and direct dependency audit

Reference: unchanged Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389`, Apache-2.0, Google LLC.
Source/test blobs are `95b077d8eb6ac7deb814c910276b4a1d9d603bee` and
`367de6a3ae0ac011a16a5cc13b63dbb77cd7ba53`.

## Ownership and complete call graph

Compiler preparation owns `hermitian_sparse_matrix_component.py`. Its inputs
are complete native column and entry Program pairs, including exact inverses.
System and the MSB a/location/b ancilla are caller-owned through ports; each
child owns its private work. Partition and inverse Partition remain explicit.
Six Always wrappers surround diffusion, column, entry and their exact inverses.
Only the location/system and a/b swaps are controlled. Positive control uses
the existing exact CSwap owner. Negative and grouped controls reuse
ControlledViaAnd, including source predicate order, polarity and erasure.

The source `adjoint()` override returns self. Explicit generic Adjoint instead
reverses the calls through supplied exact pairs. Both are verified separately;
neither reverses measurement instructions. Shared sparse contexts retain exact
helper closures and source bounded-to-unsigned index ports. Algorithms' existing
signature helpers delegate to Compiler; callback execution stays at its existing
boundary. SDK objects stop at Ecosystem, which alone computes
`2 * np.arccos(np.sqrt(entry))`. Existing nominal types and public entrances stay
unchanged. User execution remains `circuit.append(op)`.

## Source fidelity and execution evidence

`test_hermitian_sparse_matrix_component.py` checks both original examples,
default return-self and explicit generic adjoints, default/banded index roots,
recursive comparable calls and operands, and every column of small complex
operators. Positive, negative and two-bit mixed-polarity constructors retain
source control calls. Both raw histories of the two-bit case preserve the full
complex amplitude, branch weight and clean work. Existing padded/structured
Hermitian tests supply the larger unchanged primitive and erasure witnesses.
Source-free native resolution, binding and ordinary append are exercised after
blocking Qualtran and Cirq imports.

UniformSqrtEntryOracle checks zero/one-bit indices, values 0, 77/256 and 1, and
nondefault epsilon. Source epsilon is metadata: the original Ry construction
does not forward the oracle epsilon. The native constructor preserves that
omission. Symbolic widths retain declarations and refuse physical binding.
The original depth-zero examples serialize; deeper AutoPartition tuple codecs
fail in the unchanged source and are recorded without patching it.

## Measured primitives versus models

`g2-hermitian-sparse-constructor-resources-20261125.json` records twelve
directional rows. Counts include T and T-dagger in the T-family total.

| Case | Actual primitives and work | Original model |
| --- | --- | --- |
| Default | 4H, 2Ry, 3SWAP; zero private work; six live wires | Seven Clifford, two rotations, peak six |
| Positive control | 10H, 2Ry, 24CX, 21 T-family; zero private work; seven live wires | Three CSwap, four Clifford, two rotations, peak seven |
| Negative control | Positive case plus two X; seven live wires | Three CSwap, six Clifford, two rotations, peak seven |
| Two-bit 10 control | 13H, 2Ry, 30CX, 25 T-family, four X, one S, one measurement/conditional CZ/reset; one private wire; nine live wires | Three CSwap, one And, five Clifford, two rotations, one measurement, peak nine |
| Square-root entry | One Ry; zero private work; five live wires | One rotation, peak five |

All leases return empty. Raw-history and operator checks separately establish
zero amplitudes on private work; lease bookkeeping alone does not prove this.
Symbolic actual counts are null, with the independent source formulas retained.
Pool capacity is not the actual live peak.

## Remaining boundaries

UniformSqrtEntryOracle closes within its real finite amplitude domain [0,1].
Neither Hermitian constructor is present in the frozen 303-entry resolver
inventory. Both are additional supported identifiers, so this delivery adds
zero to that inventory's complete count; it remains 114/303, with 189 remaining.
Arbitrary source column/entry constructor closures remain partial and require
explicit native width, sparsity and exact Program owners. The original banded
resolver omission, private sparsity roundtrip loss and deeper tuple failures
remain separate. TopLeft genuinely symbolic-bound physical storage awaits
Proposal 121. Scalar-phase Proposal 119 and other repository/GPU/performance
gates remain open. This batch does not close G2 or start G3.
