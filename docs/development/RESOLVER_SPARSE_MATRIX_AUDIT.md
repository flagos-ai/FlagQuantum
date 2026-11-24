# Sparse matrix constructor and direct dependency closure

Reference: unchanged Qualtran v0.7.0,
`bcb17bb25360ed68ce3bc084645db3e929799389`, Apache-2.0, Google LLC.
Source/test blobs are `cacbdab733f0be4f0f28bf2a749e1bad79987bfd`
and `afcb0771b405ed26bfce7827d70d81c3d7f7a0d8`.

## Ownership and replacement

Compiler arithmetic owns sided TopLeft and banded index graphs in
`sparse_index_component.py`. Lookup owns uniform and QROM-driven entry graphs
in `matrix_entry_component.py`. Preparation owns the complete SparseMatrix
root and packed diffusion in `sparse_matrix_component.py`. Existing Algorithms
sparse signature consumers delegate to that owner. Algorithms retains matrix
metadata, signal preparation and independent cost formulas.

Formats owns native constructor recipes, oracle width/sparsity roles and exact
inverse pairs. Ecosystem converts numerical entry payloads through the existing
Algorithms ExplicitEntryOracle table owner. Compiler imports no Algorithms,
source SDK or numerical executor and selects no backend. Normal execution
continues through bound native operations and `circuit.append(op)`.

The layout owner now supplies one exact AutoPartition pair implementation for
packed oracles, Product constituents and sparse diffusion. Existing callers
delegate, removing duplicated inverse-partition logic. Symbolic partition views
retain their existing bodyless declarations; they are not forced through
concrete Split/Join. Existing symbolic packed PREPARE regression checks this
boundary as well as concrete gates, lifetimes and complex action.

## Interface and literal source order

The root exposes opaque system(n) and ancilla(n+1), with no resource port.
Ancilla is MSB flag q followed by location l. Every exposed word is caller-owned;
children own private clean work and exact cleanup. Root order is Split ancilla,
Join location, packed uniform diffusion, column(l,i=system), entry(q,i=l,j=system),
Swap, exact row inverse, exact diffusion inverse, Split location, Join ancilla.
The leading Unused diffusion slice is retained even when its width is zero.

Row/column ports preserve LEFT BQUInt(n,s), RIGHT QUInt(n) and THRU QUInt(n).
TopLeft is the original identity map. Banded calls unsigned Add(a=i,b=l), then
signed AddK(k=-bandsize), and exact adjoint reverses those calls. Entry indices
are QAny, not QUInt. Uniform entry applies source Ry(2*np.arccos(entry)).

Explicit entry allocates its angle word, calls original QROM, splits it, applies
the ordered positive controlled Ry angles 2*pi*2**-(k+1), rejoins, calls QROM
again and frees the angle word. QROM returns itself as its source adjoint and
retains its own measured predicate cleanup. Original controlled Ry is an atomic
Controlled node whose decomposition refuses atomic Ry. Native cry leaves are
verified independently as full complex operators in both directions; no
nonexistent source decomposition is claimed below that boundary.

## Payload and source wire defects

Explicit angle conversion retains the input NumPy scalar precision until after
arccos and scaling. Converting a float32 matrix through `.tolist()` first would
promote it to double and can change integer angle words. A 28-bit table witness
compares every original quantized word and proves the promoted result differs.
The original 10-bit example and common 8-bit table also pass complete complex
operator checks. Large 28-bit exhaustive state simulation is not claimed.

The original producer omits TopLeft's private nondefault `_num_nonzero` field.
Its LEFT ordinal descriptor still explicitly stores the bound. Native binding
recovers this field from that type metadata, validates the full declaration and
uses the recovered sparsity for diffusion. This is explicit native recovery,
not a modification of the reference: original roundtrip of TopLeft(2,3) produces
sparsity four; native resolution matches the unchanged original object's
sparsity-three call graph and full complex operator in both directions.

All four root examples serialize at depth zero. Original default, symbolic and
explicit roots roundtrip in both directions. The banded root does not roundtrip:
the original resolver lacks SymmetricBandedRowColumnOracle. Deeper original
AutoPartition tuple serialization fails unchanged. These outcomes are recorded
independently from successful native construction.

## Verification and resources

`tests/unit/test_sparse_matrix_component.py` covers all original root examples,
directions and oracle families; exact sided signatures and recursive named
calls; full small complex operators; original 10-bit and common 8-bit tables;
every combined raw history for a small measured sparse forward/inverse pair;
coherent flag/index inputs and zero private work. Every 8-bit controlled-Ry leaf
has its complete operator checked against the unchanged source atomic node.
Source SDK imports are blocked during resolution, binding and append.
Symbolic root and entry-precision declarations retain explicit physical refusal.
Zero-width entry indices retain the source signature's omitted i/j ports.
Uniform entry still executes its one-qubit Ry. Original explicit entry then
fails because its decomposition requires the omitted i/j arguments; native
construction retains the valid declaration and explicit physical refusal.

Receipts are `ftqc-reference-env/g2-sparse-matrix-constructor-*` beside the
worktree. Resources verify eleven frozen primary/dependency blobs, Qualtran
0.7.0 and an actual Kickmix 0.1 Simulator. Sixteen directional rows keep actual
primitive counts, private lease peaks and live wires separate from original
QEC and qubit models:

| Original root | Emitted primitives | Independent source model |
| --- | --- | --- |
| Uniform | 4H, 1Ry, 2SWAP; no private work or measurements; live peak 5 | 6 Clifford, 1 rotation; peak 5 |
| Explicit, 10 bits | 16 T-family, 58CX, 10cry, 4 measurements; private peak 11; live peak 14 | 4 And, 45 Clifford, 10 rotations, 4 measurements; peak 14 |
| Banded | 72 T-family, 168CX, 4Rz, 1Ry, 3SWAP, 18 measurements; private peak 5; live peak 12 | 18 And, 129 Clifford, 5 rotations, 18 measurements; peak 12 |

The resource JSON lists every emitted gate name, including remaining Clifford
and reset gates. Atomic model And and rotation estimates are not relabeled as
emitted T or synthesized gates. Supplied pool capacity is not live peak. Lease
release alone is not zero-amplitude evidence; complete operators and branch
checks provide that evidence.

## Catalogue disposition and remaining gaps

UniformEntryOracle and ExplicitEntryOracle have complete native constructor
bindings in the declared numeric/declaration domains. SparseMatrix's arbitrary
oracle contexts remain explicit provider boundaries. TopLeft retains one
physical gap: fixed quantum width with a genuinely symbolic ordinal bound.
The unchanged source executes that identity, while current Core storage rejects
the symbolic parameter. Proposal 121 is pending approval; no protected
admission is changed in this batch. Its refusal is tested and remains visible.
Symbolic numeric width literals that decode to finite widths are separately
probed and do not stand in for this genuinely symbolic bound.

SymmetricBandedRowColumnOracle is additionally supported but is absent from
the frozen 303-entry resolver inventory; it must not inflate that catalogue.
The catalogue receipt verifies both newly closed identifiers against that frozen
inventory: complete coverage moves from 112/303 to 114/303, with 189 remaining.
TopLeft and SparseMatrix remain partial; the additional banded identifier adds
zero to that total.
Source symbolic decomposition, singleton uniform target-body failure, deeper
codec failures, non-real/nonfinite payload refusal and larger exhaustive
operators remain explicit. Pending 116/119 and whole-repository/GPU/performance
gates are independent. Existing Algorithms source-file acceptance does not
close the generic constructor parent.

In the locked environment, preimport torch and run from the repository root:

```bash
python -m pytest tests/unit/test_sparse_matrix_component.py -q
```
