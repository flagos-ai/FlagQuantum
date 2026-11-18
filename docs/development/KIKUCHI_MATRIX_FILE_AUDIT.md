# Complete Kikuchi matrix-entry source mapping

The complete Qualtran v0.7.0
`optimization/k_xor_sat/kikuchi_adjacency_matrix.py` at
`bcb17bb25360ed68ce3bc084645db3e929799389` maps to the native Algorithms owner
`flagquantum.algorithms.kikuchi_matrix`. The source and original test blobs are
checked before comparison. Q-0605 maps the entire class, both example factories,
all width/dtype properties, signature and every direct child of its call graph.

## Interface and dependency closure

`KikuchiMatrixEntry` retains its instance, ell and entry precision. The two THRU
opaque words S and T each contain ell unsigned variable indices, with each index
packed most-significant first. The final THRU bit q is the RHS-rotation target.
The native index dtype and unsigned fractional rotation type match the original.
The concrete factory uses the exact ten-variable instance, ell eight and three
angle bits. The symbolic factory keeps ell=c*k and the same fixed precision.

The five direct calls are the forward and inverse symmetric difference, the
forward and inverse unique-scope index, and the RHS rotation. Both compute pairs
retain the original Always control bypass. Existing Compiler symmetric-difference
and Algorithms index/RHS owners expand the full cost chain. A controlled model
changes only the RHS amplitude QROM control count. Caller-provided graph keys keep
forward and inverse roles explicit and allow equal keys to aggregate correctly.

All seven QEC categories match for concrete/symbolic factories and both outer
control variants. The concrete uncontrolled declaration matches 512 CSwaps,
1,301 Ands, 12,518 Cliffords and 1,301 measurements. These are model categories.
The root emits no physical instruction stream in the frozen source.

## Retained acceptance gaps

The untouched original non-notebook suite reports five passes, six strict skips
and two expected failures. Its source has no physical decomposition for either
factory, so decomposition, count/decomposition comparison and dtype checks retain
their original skips. Both serialization checks retain their KXorInstance xfails.
The source's full expected costs and controlled graph differences run unchanged.

Q-0605 remains active for inherited executable index/RHS control-adder admission
and the physical root boundary. Q-0606 retains the original skips and serialization
limitations. Q-0604 tracks the complete notebook separately. Existing executable
QROM and register components retain their `circuit.append(op)` bindings. This
root's model acceptance carries no complex-amplitude, work-zero, measurement
history or measured-primitive claim. All applicable child evidence is reused from
`KXOR_LOADING_FILE_AUDIT.md` and the Compiler symmetric-difference audits.

## Validation

The complete source suite ran independently with its original five passes, six
skips and two xfails. The concentrated native regression passed 511 tests. Black,
Ruff, strict mypy for both changed Python files, Algorithms scope, architecture,
multi-level IR and protected API checks passed.

## Exact adjoint model dependency

The block-encoding closure exercises inverse RHS/matrix entry costs. The RHS
model now routes the inverse phase-gradient rotation to the existing exact
controlled-add/subtract polarity and reverses the controlled S declaration.
All seven categories match untouched concrete/symbolic source queries, including
the specialized controlled RHS. Inverse RHS and uncontrolled matrix costs have
two fewer Cliffords than their forward models. Forward callers retain their
existing results. The generic controlled-adjoint matrix source query reaches an
atomic controlled And and fails. Native model admission reports that boundary
explicitly. This dependency adds resource evidence and carries no new physical
primitive, amplitude or restoration claim. Concentrated regression passed 618
tests. Strict typing, formatting, lint, scope, architecture, IR and API checks
passed for the three changed Python files.
