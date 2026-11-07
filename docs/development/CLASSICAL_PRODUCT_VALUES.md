# Classical product values

`simulation/classical_product_accumulation.py` ports the arithmetic expression
from pinned Qualtran v0.7.0 `PlusEqualProduct.apply` and `on_classical_vals`.
`accumulate_product_classical_values(a, b, result, width, adjoint=False)` returns
both original operands and the new accumulator. It multiplies the operands,
multiplies by the direction, adds the accumulator, then reduces modulo `2**width`.
No intermediate Python integer casts or extra modulo reductions are introduced.
NumPy promotion/overflow, arrays and symbolic expressions retain source behavior.
Input-domain and signature-width validation belong to register binding.

The tests compare both original source entry points, result types, arithmetic
failures, array broadcasting and exact forward/inverse values. Existing quantum
acceptance consumes the new evaluator when constructing its complete coherent
output permutation, then compares it to original source tensors, certified
clean-work maps and forced measurement histories. This does not turn a value
model into a compiler dependency; numerical execution stays Simulation-owned.

The original PlusEqualProduct has no gate decomposition. Native executable
accumulation continues to use the previously accepted weighted-column product,
shared Add/Add-adjoint and measured erasure. Its emitted instructions and source
approximate `8 * max(a_width, b_width)**2` T model remain separate. A direct value
port cannot establish source gate-sequence equality where no source sequence exists.


## Square production and checked erasure

`simulation/classical_square.py` ports `Square.on_classical_vals`: forward returns
`a**2` and ignores an extra result field. Reverse requires `result == a**2`,
raising the original assertion failure for inconsistent values. Missing fields
keep KeyError and array equality keeps the original ambiguous-truth error. No
width reduction or casts are introduced. This equality is not replaced by an
unchecked delete or a different inverse formula.

Run the user example without reference dependencies:

```sh
python -m examples.algorithms.classical_product_values
```

It checks occupied multiply-accumulate wraparound and inverse, then appends a
square to a coherent state and erases it with the same clean-work decomposition.
For two input bits the native circuit emits 8 T, 2 measurements and 2 work wires.
The pinned source Square model declares 2 Toffoli; its 8 T-equivalent conversion
uses the explicit four-T assumption. The source supplies tensors and classical
values but no gate decomposition. Existing small forced-history tests compare
complete complex amplitudes including the spectator and zero work sector;
three-bit tests use selected histories plus the all-history channel certificate.

Batch acceptance: 85 new value tests;126 concentrated passes,no skips. The 63
visible warnings come from unchanged NumPy overflow in source `apply`, source
`on_classical_vals` and native evaluation. They are not suppressed. Full source
blobs remain pinned; no reference code is modified. Entire multiplication.py and
other fixed-point/product components are not thereby declared complete.
