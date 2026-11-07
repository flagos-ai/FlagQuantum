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
