# Montgomery value representation

`simulation/montgomery_values.py` owns direct numerical representation methods
from `_MontgomeryUInt` in unmodified Qualtran v0.7.0, commit
bcb17bb25360ed68ce3bc084645db3e929799389. It introduces no foreign dtype/graph
object, imports no compiler and emits no quantum gate. Compiler keeps the
Montgomery multiplication and inverse circuit algorithms.

| Native function | Original arithmetic and conversion points |
| --- | --- |
| integer_to_montgomery_value | `(x * pow(2, int(width), int(mod))) % mod` |
| montgomery_to_integer_value | `(x * pow(2, -int(width), int(mod))) % mod` |
| montgomery_product_values | `(x * y * pow(2, -int(width), int(mod))) % mod` |
| montgomery_inverse_value | `(pow(x, -1, mod) * pow(2, 2*width, int(mod))) % mod` |
| montgomery_value_domain | Concrete modulus: `range(1, int(mod))`; absent/symbolic modulus: `range(2**width)` |

Every arithmetic method first retains the original non-None, non-symbolic
modulus assertion. No additional positivity, primality, residue bound or operand
conversion is imposed. The inverse's first pow receives the original operand
and modulus, and its second exponent does not cast width. In particular, a
Sympy integer width can work for conversion/product but fail inverse. NumPy
operand types, multiplication overflow, broadcast arrays, noninvertibility and
final remainder types retain the source behavior. The declared int/expression
interfaces do not turn malformed numeric diagnostics into physical registers.

The source domain metadata is distinct from its storage bounds and numerical
methods: a concrete modulus lists nonzero residues, while stored-word encoding
still accepts any width-fitting unsigned word. Symbolic literal moduli select
the full-word domain but fail the arithmetic methods' symbolic assertion. Those
policies are not merged into a new validator.

The general stored-Montgomery array codecs are the original BitEncoding default
vectorization, now shared in `integer_encoding.py`. They preserve scalar dtype
checks and do not use _UInt's mask/transpose fast path. Shape, empty-input errors,
unchecked scalar decode and first-result dtype inference remain upstream. Signed
arrays and wide unsigned arrays also use these shared helpers with their own
existing preconditions. See [integer value encoding](INTEGER_VALUE_ENCODING.md).

## Relation to quantum algorithms

These representation formulas do not replace DirtyOutOfPlaceMontgomeryModMul's
Fermat-based root assertion or its original decomposed graph. For composite
p=9, width5 and x=y=1, the representation product is 2 while that root value
method raises AssertionError. The graph's known conditional-add-p discrepancy
also stays open. Tests compare all three source interfaces independently.

The complete quantum graph test now also compares the representation product to
the unchanged source dtype method for each input pair, and to the successful
prime-modulus root values. Its expected complex quantum state continues to use
the original graph, with its retained history/reduction fields. Original
primitive order, feedback, output aliases, clean work, raw/channel measurement
history evidence and model-versus-emitted resource checks are unchanged.

Run the native example without reference packages:

```sh
python -m examples.algorithms.classical_montgomery_values
```

It verifies representation conversion/product/inverse, generic array round trip,
scalar/bit-array window histories and forward/adjoint `circuit.append(op)` IR.
The fixed 21-wire sample still emits 128 T and 32 measurements; its pinned model
sample is printed separately. This example does not execute the 21-wire circuit;
full quantum execution evidence is in the acceptance tests.

G2-VAL15 adds 355 cases: exact source method values/types/errors and domain range
metadata, complete small residue conversion/product/inverse laws, and deliberate
root-versus-representation differences. This accepts the numerical method slice,
not the entire upstream data_types.py, full dtype hierarchy, or source defects.

Batch G2-VAL14/15 acceptance: 140 generic array + 355 representation cases =
495 new tests; 1,042 concentrated passes, no skips. The 197 visible warnings
retain the original/native NumPy arithmetic and diagnostic reductions, without
filters or casts to hide them. Strict types, Black/Ruff, team scope, architecture
and the executable example pass. The original data_types.py blob is
fe570ea7ee74571d7d5202fef52ac1a215bea1a5; mod_multiplication.py is
c0242f9735bf269564c9480c14c008d5f9621fd7. Neither reference is modified.
