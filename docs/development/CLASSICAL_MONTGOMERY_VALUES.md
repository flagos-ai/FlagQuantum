# Classical modular multiplication values

`simulation/classical_modular_multiplication.py` ports ModDbl and CModMulK's
numerical methods from locked Qualtran v0.7.0. Doubling uses the original `x+x`
when `x < p`, with no lower-bound check. Constant multiplication first tests
control truthiness, then `x < p`. It does not normalize a nonzero control to one,
validate a bit, cast operands, or repair unsigned overflow. Inactive and exterior
objects retain identity. Register and constant-constructor validation is separate
from these direct value methods.

The existing basis/coherent doubling and controlled-multiplication builders now
consume these numerical functions and retain independent mathematical asserts.
Source primitive/feedback/resource assertions, output port aliases, all-history
clean-work certificates and forced complex histories remain in the original
quantum suites. This does not broaden their quantum residue domains.

G2-VAL10 acceptance adds 271 source value tests, including Python/NumPy types,
arrays, symbolic relations, exterior values and control truthiness. Compiler gate
implementations are unchanged; their previously recorded source discrepancies
outside the valid domain remain open.

## Integer Montgomery windows and root value action

`simulation/classical_montgomery_multiplication.py` ports the original
DirtyOutOfPlaceMontgomeryModMul constructor guards, `_inversion_data`, integer
`_classical_action_window` and complete `on_classical_vals` call chain. It does
not import compiler or foreign graph types. The table retains original NumPy
`arange`, multiplication, modulo and in-place scaling instead of substituting
Python-integer arithmetic. The root creates one lazy table only when needed and
shares it across windows. An internal callable boundary allows replacement of
that table computation without changing the consumer.

Each window adds selected shifted y values in low-bit-first x order, records the
pre-correction low bits, applies the table correction, shifts the accumulator,
then appends the index to the history in big-endian chunk order. No casts change
the intermediate scalar types. The root performs one final subtract-p if needed,
retaining the source's Fermat-based equality assertion. Constructor and symbolic
rejections retain their original ordering.

Forward auxiliary values must be absent. Exterior x/y return three Python zeros.
Reverse evaluation checks the original target formula and that history/reduced
are present, but does not inspect their contents. Those weak source diagnostics
are preserved; they do not prove a dirty quantum history can be erased. Actual
quantum adjoints consume the matching history and reduction flag.

The source has separate value and circuit algorithms. The unchanged circuit
reduction conditionally adds p, whereas the root value method subtracts p.
For n=5,w=5,p=13,x=5,y=9, the source graph yields target27/history13/reduced1 while
the root value yields target1. For n=5,w=1,p=9,x=y=3, graph target18 differs from
root target0. Some composite-modulus values, including p=9,x=y=1, fail the root's
Fermat assertion. These failures and disagreements remain open. The existing
quantum test still uses the original decomposed graph for its complete expected
state; it separately compares root values or failures to the new numeric method.
It does not replace the graph with a corrected modular product.

G2-VAL11 adds 76 cases, including complete small pair domains, raw window values,
NumPy types, history ordering, array diagnostics, constructor/symbolic failures,
auxiliary assertions, lazy table reuse and native table replacement. Original
source code is not modified. SingleWindowModMul's separate bit-array classical
interface still requires an integer-codec audit; this port covers the root's
integer-window method, not a new typed-register graph interpreter.

Run the user example without reference packages:

```sh
python -m examples.algorithms.classical_montgomery_values
```

It evaluates and independently composes the integer windows, then binds forward
and exact adjoint operations with `circuit.append(op)`. For its fixed n=3,w=1,p=3
case, emitted T/measurement/declared wires are 128/32/21; the pinned source model
sample is printed separately as 128/32/21. The example constructs IR and does not
execute the 21-wire circuit. Full complex sparse trajectories, coherent spectators,
retained outputs, clean workspace, source primitives/feedback and resource-model
comparisons remain in the quantum acceptance tests. Unblocked lookup layouts
have all-history channel certificates; blocked layouts use selected full paths.

Batch acceptance: 271+76 = 347 new value tests, 404 concentrated passes, no skips
or warnings. Strict typing, Black/Ruff, team scope, architecture and the runnable
example pass. Source mod_multiplication.py blob remains
`c0242f9735bf269564c9480c14c008d5f9621fd7`. Whole-file, mathematical repair and
bit-array/source-graph interface work remain open.

## Bit-array window action (G2-VAL13)

`simulation/classical_montgomery_window.py` now ports the separate
SingleWindowModMul bit-array value method. It reuses the shared integer scalar
codecs and exact NumPy inversion-data helper. Target bits decode big endian,
while left[0] is the least-significant control. The source iterates those controls
in increasing index order, records low target bits before correction, applies
one table correction and shift, then encodes the target back to a Python list.
Input x/y preserve their objects. The incoming qrom_index is overwritten rather
than XORed; the clean quantum decomposition separately assumes it starts at zero.
No normalization of nonbinary diagnostic controls is introduced.

Constructor dimension assertions, scalar codec failures, raw table arithmetic
and required lookup metadata failures retain their original stage. In particular,
with concrete dimensions and a symbolic modulus, the source explicitly passes
log_block_sizes=None into QROAMClean's tuple converter and raises TypeError before
accessing data. Native evaluation preserves this failure rather than choosing a
block or claiming symbolic support. Negative/zero diagnostic widths, NumPy size
metadata, even/zero moduli and malformed bit sequences are tested separately from
the valid physical gate domain. Numerical evaluation does not allocate a bank or
run resource block selection; Compiler retains ownership of those decisions.

There are 24 new window tests. Five configurations enumerate all window controls
with selected operands/accumulators and both clean and dirty incoming indices.
The actual decode -> table -> encode dependency sequence is observed and its
table implementation replaced without changing the consumer. The source is not
patched. The n=2,w=1,p=3,x=1,y=2,target5 witness still gives high-level target5
and original graph target1; both interfaces retain that discrepancy.

The existing quantum test now compares the new value method to original
SingleWindowModMul independently of the original source graph. Its primitive
order/feedback, output aliases, full complex states, clean workspace, raw small
histories and all-history certificates remain unchanged. Larger blocked layouts
retain selected full trajectories and their independent model-versus-emitted
resource accounting.

`python -m examples.algorithms.classical_montgomery_values` now also composes the
bit-array windows with shared codecs and compares target/history to the integer
root chain, then constructs forward/adjoint IR through circuit.append(op). It
still reports the 21-wire example as construction only, with no execution claim.

Batch G2-VAL12/13: 401 codec/width cases plus 24 window cases = 425 new tests;
795 concentrated passes, no skips, nine original/native NumPy warnings retained.
The regression includes existing value binding/output collection, typed negation,
Montgomery root methods and complete single/multiple-window quantum suites.
Strict typing, Black/Ruff, scope, architecture and the runnable example pass.
The data_types.py, mod_multiplication.py, qroam_clean.py and qrom_base.py source
blobs are verified unchanged. Other integer encodings and whole source-file
acceptance remain open.
