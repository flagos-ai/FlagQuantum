# Classical comparison values

This slice ports numerical `on_classical_vals` methods from the unchanged
Qualtran v0.7.0 `arithmetic/comparison.py`, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

## Interfaces, bit order and ownership

Simulation owns three numerical groups: XOR-target constant/word/equality
comparisons, the linear and controlled greater-than value actions, and
half-comparison compute/erase values. Inputs are existing integer value objects;
unchanged operands retain identity. Widths and dtypes unused by a source numerical
method are not added as artificial validation arguments. Half comparisons take
width because it controls the stored difference modulus.

The numerical functions do not reinterpret raw words as signed values, allocate
quantum work or emit gates. Existing integer codecs own big-endian representation.
Compiler continues to own all comparison gate sequences, logical output bindings,
work requirements and measurement feedback; Runtime owns execution.

The interfaces preserve different source update policies. XOR comparators do not
validate the target as binary. `LinearDepthGreaterThan` asserts a binary target
and updates it with `(target + 1) % 2`. `CLinearDepthGreaterThan` tests `ctrl == cv`
and preserves inactive objects, without the unconditional comparator's target
assertion. Half comparisons cast operands at the original positions, use shared
integer addition for the difference, and assert both retained outputs before
logical erasure. These policies must not be collapsed into one normalized XOR.

Known source quantum/classical differences remain explicit: the signed linear
gate implementation can copy an overflowing difference sign rather than the
mathematical comparison, and reversed half-relation adjoints expose swapped
logical ports. Numerical methods retain the original classical behavior, while
quantum consumers retain the original decompositions. No source is repaired or
changed to claim consistency between those different interfaces.

## Delivered numerical methods

| Numerical owner | Source methods | Update policy |
| --- | --- | --- |
| `classical_comparison.py` | LessThanConstant, LessThanEqual, Equals | Raw predicate XOR; no width or binary-target check |
| `classical_greater_comparison.py` | LinearDepthGreaterThan | Binary-target assertion, scalar branch, add one modulo two |
| `classical_greater_comparison.py` | CLinearDepthGreaterThan | Exact activation equality, raw XOR, inactive identity |
| `classical_half_comparison.py` | Four LinearDepthHalf relations | Compute retained difference/flag, or assert both before erasure |

Half `>` and `<=` retain `b-a`; `<` and `>=` retain `a-b`, modulo
`2**(width+1)`. Only operands are converted with `int`, at the source positions.
The width is not converted: `typing.cast` in the call to the shared add helper
only expresses its existing static integer-width contract. NumPy and symbolic
width arithmetic remains visible at runtime. Difference and flag are logical
outputs until erased, not clean temporary work.

Unchanged operands retain identity, including arrays. Source array broadcasting,
result dtypes, NumPy boolean-target promotion, errors and scalar-conversion
warnings remain observable. The unconditional comparator's NumPy boolean target
can become a NumPy integer after add/modulo, whereas the controlled comparator's
XOR retains a boolean result. Neither route is normalized to hide this difference.
Diagnostic control values outside zero/one and unused symbolic/invalid width
metadata do not become valid executable compiler inputs by passing a numerical
method.

## Running and validating the user path

```sh
python -m examples.algorithms.classical_comparison_values
pytest -q tests/unit/test_classical_comparison.py tests/unit/test_classical_greater_comparison.py tests/unit/test_classical_half_comparison.py
```

The example reuses the existing `Equality` operation and `circuit.append(op)`.
All three-bit input pairs and both initial target values appear in a coherent
complex state. Numerical values determine the expected target, measured execution
agrees, work is zero, and a second comparison restores the complete input state.
Emitted equality counts are eight T/T-dagger instructions and two measurements,
with two work wires. The pinned source model's eight T equivalents and two
measurements are printed as a separate budget, not inferred from the emitted IR.

Tests connect constant, less-equal, equality, both controlled activation polarities
and four half relations to the existing compiler consumers. Source instruction
dependencies, operands, feedback and resource models remain checked by unchanged
reference traversal. Full complex states, spectators and all small forced
measurement branches verify work cleanup. Half forward outputs are compared
numerically and then erased in every small history; source reversed adjoint port
aliases remain authoritative and separate from classical return names.

## Boundaries retained

- The existing signed linear discrepancy is recorded in
  [the signed difference-sign guide](REGISTER_SIGNED_DIFFERENCE_SIGN.md). For raw
  two-bit `a=3,b=0,target=0`, the original numerical method and native value method
  produce one while the original/native signed gate path produces zero. The port
  does not reinterpret the numeric operands or replace the source gate path.
- For reversed half relations, classical erasure returns the original named
  operands while the decomposed source returns swapped logical port names.
  `half_compare_outputs` preserves that physical contract. No SWAP is inserted.
- The source private half-base relation validator does not reject an unexpected
  string. Its numerical else branch is reproduced in a diagnostic test. Supported
  public relations remain the four declared operations; Compiler still rejects
  unsupported relation names.
- A correct classical erasure check is not proof that a quantum circuit can erase
  arbitrary dirty outputs. Measured cleanup is tested only on correct retained
  outputs, with all amplitudes and probabilities checked.

Acceptance: 479 XOR, 417 greater-than and 216 half-comparison cases are new.
The concentrated batch passes 1,612 cases without skips, retaining 582
reference/native numerical warnings. Four production/example files pass
changed-surface strict typing, formatting, ownership and architecture checks.
Five pinned source blobs covering comparisons and their arithmetic/value
dependencies match the immutable inventory. The full source comparison file and
its known source-interface conflicts remain open in the integration ledger.
