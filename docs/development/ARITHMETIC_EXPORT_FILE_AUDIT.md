# Arithmetic package export whole-file audit

Q-0171 covers `qualtran/bloqs/arithmetic/__init__.py` at Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389`. The entire file consists of
41 imports. It contains no execution, register allocation, measurements or cost
formula. The original Git blob is checked, together with export order and exact
identity with every defining module. This verifies the initializer responsibility.
Behavioral evidence comes from the existing consumers below.

## Native responsibility and actual consumers

Paths default to `flagquantum/compiler/registers/`. Explicit algorithms and
Simulation paths start at `flagquantum/`. Consumer names refer to
`tests/unit/test_<name>.py`. These are responsibility links to existing native
implementations, without adding a foreign-name registry or changing public
exports. Physical consumers retain ordinary `circuit.append(op)` binding.

| Source export | Native owner and entry | Existing consumer and scope |
| --- | --- | --- |
| `Add` | `arithmetic/addition.add_ir` | register_addition; addition_file_audit |
| `AddK` | `arithmetic/addition.add_constant_ir` | addition_file_audit |
| `OutOfPlaceAdder` | `arithmetic/out_of_place_addition.out_of_place_add_ir` | addition_file_audit |
| `BitwiseNot` | `bits/bitwise.bitwise_not_ir` | register_bitwise |
| `Xor` | `bits/bitwise.xor_ir` | register_bitwise |
| `XorK` | `bits/bitwise.xor_constant_ir` | register_bitwise |
| `BiQubitsMixer` | `comparison/comparison_tree.bi_qubits_mixer_ir` | comparison_tree |
| `CLinearDepthGreaterThan` | `comparison/controlled_comparison.controlled_greater_ir` | register_linear_greater; signed_linear_greater |
| `Equals` | `comparison/equality.equality_ir` | register_equality |
| `EqualsAConstant` | `comparison/equality.equality_constant_ir` | register_equality |
| `GreaterThan` | `comparison/ordering.greater_ir, greater_adjoint_ir` | register_greater; greater_adjoint |
| `GreaterThanConstant` | `algorithms/primitives/thc_uniform_preparation.THCUniformPreparation.source_resource_model` | thc_uniform_preparation. Declaration only, same-threshold LessThanConstant source edge |
| `LessThanConstant` | `comparison/less_than.less_than_constant_ir` | register_comparison |
| `LessThanEqual` | `comparison/less_equal.less_equal_ir` | register_less_equal |
| `LinearDepthHalfGreaterThan` | `comparison/half_ordering.half_compare_ir (gt)` | half_ordering; half_comparison |
| `LinearDepthHalfGreaterThanEqual` | `comparison/half_ordering.half_compare_ir (ge)` | half_ordering |
| `LinearDepthHalfLessThan` | `comparison/half_ordering.half_compare_ir (lt)` | half_ordering |
| `LinearDepthHalfLessThanEqual` | `comparison/half_ordering.half_compare_ir (le)` | half_ordering |
| `SingleQubitCompare` | `comparison/comparison_tree.single_qubit_compare_ir` | comparison_tree |
| `CAdd` | `arithmetic/controlled_addition.controlled_add_ir, controlled_add_adjoint_ir` | register_controlled_addition; controlled_addition_adjoint |
| `SignedIntegerToTwosComplement` | `bits/signed_encoding.ones_complement_to_twos_ir` | signed_encoding. Original one's-complement signature/model only, no source root decomposition |
| `SignExtend` | `bits/sign_extension.sign_extend_ir` | sign_extension |
| `ToContiguousIndex` | `sorting/contiguous_index.contiguous_index_ir` | contiguous_index |
| `HammingWeightCompute` | `bits/hamming_weight.hamming_weight_ir` | hamming_weight |
| `InvertRealNumber` | `fixed_point/reciprocal.reciprocal_ir` | reciprocal; reciprocal_newton; reciprocal_seed. Source declaration and native execution contracts differ |
| `MultiplyTwoReals` | `fixed_point/fixed_point_scaling.fixed_point_product_ir` | fixed_point_product. Source declaration only |
| `PlusEqualProduct` | `arithmetic/product_accumulation.product_accumulation_ir` | product_accumulation; integer_multiplication_file_audit |
| `Product` | `arithmetic/integer_product.integer_product_ir` | integer_product. Source declaration only |
| `ScaleIntByReal` | `fixed_point/fixed_point_scaling.scale_integer_by_real_ir` | fixed_point_scaling. Source declaration only |
| `Square` | `arithmetic/integer_square.integer_square_ir` | integer_multiplication_file_audit |
| `SquareRealNumber` | `fixed_point/fixed_point_square.fixed_point_square_ir` | integer_multiplication_file_audit. Source two-input signature differs from native one-input execution |
| `SumOfSquares` | `arithmetic/integer_square.sum_of_squares_ir` | sum_of_squares. Source declaration only |
| `Negate` | `arithmetic/addition.negate_ir` | negation_file_audit |
| `BitonicSort` | `sorting/bitonic_sort.bitonic_sort_ir` | sorting_file_audit |
| `Comparator` | `comparison/compare_exchange.compare_exchange_ir` | sorting_file_audit |
| `Subtract` | `arithmetic/subtraction.subtract_ir` | subtraction_file_audit |
| `SubtractFrom` | `arithmetic/subtraction.subtract_from_ir` | subtraction_file_audit |
| `ArcSin` | `simulation/classical_arcsine.source_arcsine_classical_values; fixed_point/fixed_point_arcsine.fixed_point_arcsine_ir` | trigonometric_file_audit; signed_arcsine. Scalar and independent Appendix D execution contracts differ |
| `CHalf` | `modular/modular_declarations.modular_arithmetic_signature (controlled_half)` | modular_declaration_file; modular_declarations. Declaration only |
| `Lt` | `modular/modular_declarations.modular_arithmetic_signature (less_than)` | modular_declaration_file; modular_declarations. Declaration only |
| `MultiCToffoli` | `modular/modular_declarations.modular_arithmetic_signature (multi_controlled_x)` | modular_declaration_file; modular_declarations. Declaration only |

## Reused semantics and retained gaps

Existing execution suites retain MSB-first words, bound logical outputs,
caller-owned clean scratch, exact compute/adjoint, complete complex states and
applicable small measurement histories. The batch reruns these scenario suites
without replacing the locked source or altering reference expectations.
ADDITION_FILE_AUDIT.md, INTEGER_MULTIPLICATION_FILE_AUDIT.md,
SORTING_FILE_AUDIT.md, TRIGONOMETRIC_FILE_AUDIT.md and
MODULAR_DECLARATION_FILE_AUDIT.md retain their complete method and factory
accounting. Their implementation and original-test parent IDs remain independent.

GreaterThanConstant has no executable decomposition or classical action. Its
original graph points to LessThanConstant with the same threshold. The existing
THC consumer deliberately preserves that cost declaration, and its tests record
the source decomposition/tensor failures. This initializer does not establish a
quantum greater-than-threshold emitter. Multiplication declarations lacking
source decompositions and the ArcSin scalar/Appendix D distinction remain open
under their original implementation parents. Generic foreign construction,
serialization, rebinding and protected static admission retain their existing
boundaries.

The initializer emits zero primitives and owns zero physical work or measurement
branches. Existing dependency primitive streams/counts are measured separately
from declared QEC/Toffoli/category/peak estimates. For example the original
ten-bit ArcSin declares 13425 Toffoli, whereas the independent Appendix D circuit
has its own recorded T/CX/CCX/measurement inventory. No package-level total or
primitive equality is inferred from those models.

Q-0171 closes only the complete package-export responsibility. Its child files
keep their separately recorded source defects and incomplete interfaces.
Batch logs are `ftqc-reference-env/g2-arithmetic-export-file-*.log`.

The concentrated regression passed 1167 cases. Two existing warnings came from
an unwritable IPython directory using its temporary-directory fallback and a
non-interactive notebook figure. There were no skips. Formatting, lint, strict typing of the
new export test, Compiler scope, architecture, multilevel IR and protected API
checks passed. This reuses accepted behavioral evidence without asserting that
source declaration-only roots have executable primitive parity.
