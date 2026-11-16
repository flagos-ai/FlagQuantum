# Finite-field package entrances

The full `gf_arithmetic/__init__.py` at Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389` matches inventory Q-0508.
Every export has an executable native owner. The direct module paths
are used in FlagQuantum's internal compiler package.

| Frozen entry | Native entry under `flagquantum/compiler/registers/finite_field` |
| --- | --- |
| GF2AddK | `coefficients.gf2_add_constant_ir` |
| GF2Addition | `coefficients.gf2_add_ir` |
| GF2Inverse | `field_inverse.gf2_inverse_decomposition`, with clean XOR via `gf2_inverse_ir` |
| BinaryPolynomialMultiplication | `polynomial.binary_polynomial_product_ir` in plain mode |
| GF2MulK | `field_linear.gf2_constant_multiplication` |
| GF2Multiplication | `field_multiplication.gf2_multiplication` in quadratic mode |
| GF2MulViaKaratsuba | `field_multiplication.gf2_multiplication` in Karatsuba mode |
| GF2ShiftLeft | `field_linear.gf2_monomial_multiplication` with negated power |
| GF2ShiftRight | `field_linear.gf2_monomial_multiplication` with positive power |
| MultiplyPolyByOnePlusXk | `polynomial.binary_polynomial_product_ir` with a shift |
| GF2Square | `field_linear.gf2_frobenius` |

The source package identities and all 11 native bindings were tested.
Each native operation ran through `circuit.append(op)` on a concrete
GF(4) or degree-one polynomial example, with the original bit order.
Returned wire permutations were bound using the existing output
planner. The inverse example uses the documented clean XOR adapter
and verifies that its workspace is zero.

The earlier addition, inversion, multiplication and square audits
remain the evidence for whole source call graphs, gate streams,
complex amplitudes, model counts and open boundaries. This package
audit closes Q-0508. It leaves the addition source count discrepancies,
foreign descriptor admission and generic serialization boundaries
recorded at Q-0510/Q-0511/Q-0513/Q-0514.

The concentrated native suite passed 392 tests. Strict typing,
formatting, team scope, architecture, IR and public API checks passed.
