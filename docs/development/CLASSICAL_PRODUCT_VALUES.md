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
