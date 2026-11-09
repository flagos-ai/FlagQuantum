# Register package organization

## Scope

This is a mechanical relocation of 191 internal register modules on integration
PR #1084. The base revision is `0bc67ecceba53ef0728e0c05cba98c90899a3c32`.
The execution path is classified as `single_device_fast_path`.

The owning team is Compiler. Cross-domain consumers receive import updates only.
Integration owns the mechanical import update in the pinned-source parity tool.
Core contracts, protected public exports, numerical algorithms, instruction
ordering and resource formulas retain their existing definitions.

The [package guide](../../flagquantum/compiler/registers/README.md) describes
responsibilities, dependencies and executable examples. Test filenames retain
stable scenario identities. Source headers preserve upstream copyright and
pinned-source provenance. Internal callable names are outside this filename
relocation.

## Module map

Paths below are relative to `flagquantum/compiler/registers/`.

| Previous module | Owning module |
| --- | --- |
| `_kickmix_comparison_gates.py` | [primitives/_mixed_gates.py](../../flagquantum/compiler/registers/primitives/_mixed_gates.py) |
| `always.py` | [primitives/always.py](../../flagquantum/compiler/registers/primitives/always.py) |
| `and_compute.py` | [primitives/and_compute.py](../../flagquantum/compiler/registers/primitives/and_compute.py) |
| `arcsine_coefficients.py` | [fixed_point/arcsine_coefficients.py](../../flagquantum/compiler/registers/fixed_point/arcsine_coefficients.py) |
| `arcsine_reduction.py` | [fixed_point/arcsine_reduction.py](../../flagquantum/compiler/registers/fixed_point/arcsine_reduction.py) |
| `arctan_arithmetic.py` | [fixed_point/arctan_arithmetic.py](../../flagquantum/compiler/registers/fixed_point/arctan_arithmetic.py) |
| `arctan_lookup.py` | [lookup/arctan_lookup.py](../../flagquantum/compiler/registers/lookup/arctan_lookup.py) |
| `arithmetic.py` | [arithmetic/addition.py](../../flagquantum/compiler/registers/arithmetic/addition.py) |
| `arithmetic_declarations.py` | [arithmetic/arithmetic_declarations.py](../../flagquantum/compiler/registers/arithmetic/arithmetic_declarations.py) |
| `arithmetic_resource_model.py` | [arithmetic/arithmetic_resource_model.py](../../flagquantum/compiler/registers/arithmetic/arithmetic_resource_model.py) |
| `auto_partition.py` | [layout/auto_partition.py](../../flagquantum/compiler/registers/layout/auto_partition.py) |
| `basis_state.py` | [primitives/basis_state.py](../../flagquantum/compiler/registers/primitives/basis_state.py) |
| `binary_unary_conversion.py` | [bits/binary_unary_conversion.py](../../flagquantum/compiler/registers/bits/binary_unary_conversion.py) |
| `bit_compression.py` | [bits/bit_compression.py](../../flagquantum/compiler/registers/bits/bit_compression.py) |
| `bitonic_merge.py` | [sorting/bitonic_merge.py](../../flagquantum/compiler/registers/sorting/bitonic_merge.py) |
| `bitonic_sort.py` | [sorting/bitonic_sort.py](../../flagquantum/compiler/registers/sorting/bitonic_sort.py) |
| `bitwise.py` | [bits/bitwise.py](../../flagquantum/compiler/registers/bits/bitwise.py) |
| `bounded_increment.py` | [arithmetic/bounded_increment.py](../../flagquantum/compiler/registers/arithmetic/bounded_increment.py) |
| `bounded_majorana_select.py` | [lookup/bounded_majorana_select.py](../../flagquantum/compiler/registers/lookup/bounded_majorana_select.py) |
| `bounded_pauli_select.py` | [lookup/bounded_pauli_select.py](../../flagquantum/compiler/registers/lookup/bounded_pauli_select.py) |
| `classical_controlled.py` | [primitives/classical_controlled.py](../../flagquantum/compiler/registers/primitives/classical_controlled.py) |
| `compare_exchange.py` | [comparison/compare_exchange.py](../../flagquantum/compiler/registers/comparison/compare_exchange.py) |
| `comparison.py` | [comparison/less_than.py](../../flagquantum/compiler/registers/comparison/less_than.py) |
| `comparison_layout.py` | [comparison/comparison_layout.py](../../flagquantum/compiler/registers/comparison/comparison_layout.py) |
| `comparison_resource_model.py` | [comparison/comparison_resource_model.py](../../flagquantum/compiler/registers/comparison/comparison_resource_model.py) |
| `comparison_tree.py` | [comparison/comparison_tree.py](../../flagquantum/compiler/registers/comparison/comparison_tree.py) |
| `constant_add_resource_model.py` | [arithmetic/constant_add_resource_model.py](../../flagquantum/compiler/registers/arithmetic/constant_add_resource_model.py) |
| `contiguous_index.py` | [sorting/contiguous_index.py](../../flagquantum/compiler/registers/sorting/contiguous_index.py) |
| `control_predicate.py` | [control/control_predicate.py](../../flagquantum/compiler/registers/control/control_predicate.py) |
| `control_predicate_resource_model.py` | [control/control_predicate_resource_model.py](../../flagquantum/compiler/registers/control/control_predicate_resource_model.py) |
| `controlled_add_resource_model.py` | [arithmetic/controlled_add_resource_model.py](../../flagquantum/compiler/registers/arithmetic/controlled_add_resource_model.py) |
| `controlled_addition.py` | [arithmetic/controlled_addition.py](../../flagquantum/compiler/registers/arithmetic/controlled_addition.py) |
| `controlled_comparison.py` | [comparison/controlled_comparison.py](../../flagquantum/compiler/registers/comparison/controlled_comparison.py) |
| `controlled_component.py` | [control/controlled_component.py](../../flagquantum/compiler/registers/control/controlled_component.py) |
| `controlled_component_resource_model.py` | [control/controlled_component_resource_model.py](../../flagquantum/compiler/registers/control/controlled_component_resource_model.py) |
| `controlled_modular_addition.py` | [modular/controlled_modular_addition.py](../../flagquantum/compiler/registers/modular/controlled_modular_addition.py) |
| `controlled_modular_addition_adjoint.py` | [modular/controlled_modular_addition_adjoint.py](../../flagquantum/compiler/registers/modular/controlled_modular_addition_adjoint.py) |
| `controlled_modular_negation.py` | [modular/controlled_modular_negation.py](../../flagquantum/compiler/registers/modular/controlled_modular_negation.py) |
| `controlled_modular_subtraction.py` | [modular/controlled_modular_subtraction.py](../../flagquantum/compiler/registers/modular/controlled_modular_subtraction.py) |
| `controlled_not.py` | [primitives/controlled_not.py](../../flagquantum/compiler/registers/primitives/controlled_not.py) |
| `controlled_register_xor.py` | [bits/controlled_register_xor.py](../../flagquantum/compiler/registers/bits/controlled_register_xor.py) |
| `controlled_rotation.py` | [primitives/controlled_rotation.py](../../flagquantum/compiler/registers/primitives/controlled_rotation.py) |
| `controlled_swap.py` | [primitives/controlled_swap.py](../../flagquantum/compiler/registers/primitives/controlled_swap.py) |
| `cost_function_phasing.py` | [phase/cost_function_phasing.py](../../flagquantum/compiler/registers/phase/cost_function_phasing.py) |
| `equality.py` | [comparison/equality.py](../../flagquantum/compiler/registers/comparison/equality.py) |
| `field_erasure.py` | [finite_field/field_erasure.py](../../flagquantum/compiler/registers/finite_field/field_erasure.py) |
| `field_inverse.py` | [finite_field/field_inverse.py](../../flagquantum/compiler/registers/finite_field/field_inverse.py) |
| `field_linear.py` | [finite_field/field_linear.py](../../flagquantum/compiler/registers/finite_field/field_linear.py) |
| `field_multiplication.py` | [finite_field/field_multiplication.py](../../flagquantum/compiler/registers/finite_field/field_multiplication.py) |
| `field_polynomial_addition.py` | [finite_field/field_polynomial_addition.py](../../flagquantum/compiler/registers/finite_field/field_polynomial_addition.py) |
| `field_polynomial_layout.py` | [finite_field/field_polynomial_layout.py](../../flagquantum/compiler/registers/finite_field/field_polynomial_layout.py) |
| `finite_field.py` | [finite_field/coefficients.py](../../flagquantum/compiler/registers/finite_field/coefficients.py) |
| `fixed_point_arcsine.py` | [fixed_point/fixed_point_arcsine.py](../../flagquantum/compiler/registers/fixed_point/fixed_point_arcsine.py) |
| `fixed_point_horner.py` | [fixed_point/fixed_point_horner.py](../../flagquantum/compiler/registers/fixed_point/fixed_point_horner.py) |
| `fixed_point_polynomial.py` | [fixed_point/fixed_point_polynomial.py](../../flagquantum/compiler/registers/fixed_point/fixed_point_polynomial.py) |
| `fixed_point_scaling.py` | [fixed_point/fixed_point_scaling.py](../../flagquantum/compiler/registers/fixed_point/fixed_point_scaling.py) |
| `fixed_point_sqrt.py` | [fixed_point/fixed_point_sqrt.py](../../flagquantum/compiler/registers/fixed_point/fixed_point_sqrt.py) |
| `fixed_point_square.py` | [fixed_point/fixed_point_square.py](../../flagquantum/compiler/registers/fixed_point/fixed_point_square.py) |
| `global_phase.py` | [primitives/global_phase.py](../../flagquantum/compiler/registers/primitives/global_phase.py) |
| `grouped_constant_add_resource_model.py` | [arithmetic/grouped_constant_add_resource_model.py](../../flagquantum/compiler/registers/arithmetic/grouped_constant_add_resource_model.py) |
| `grouped_constant_xor.py` | [bits/grouped_constant_xor.py](../../flagquantum/compiler/registers/bits/grouped_constant_xor.py) |
| `half_comparison.py` | [comparison/half_comparison.py](../../flagquantum/compiler/registers/comparison/half_comparison.py) |
| `half_ordering.py` | [comparison/half_ordering.py](../../flagquantum/compiler/registers/comparison/half_ordering.py) |
| `hamming_weight.py` | [bits/hamming_weight.py](../../flagquantum/compiler/registers/bits/hamming_weight.py) |
| `hamming_weight_phasing.py` | [phase/hamming_weight_phasing.py](../../flagquantum/compiler/registers/phase/hamming_weight_phasing.py) |
| `hamming_weight_resource_model.py` | [bits/hamming_weight_resource_model.py](../../flagquantum/compiler/registers/bits/hamming_weight_resource_model.py) |
| `identity.py` | [primitives/identity.py](../../flagquantum/compiler/registers/primitives/identity.py) |
| `in_place_sort.py` | [sorting/in_place_sort.py](../../flagquantum/compiler/registers/sorting/in_place_sort.py) |
| `indexed_component.py` | [control/indexed_component.py](../../flagquantum/compiler/registers/control/indexed_component.py) |
| `indexed_component_control.py` | [control/indexed_component_control.py](../../flagquantum/compiler/registers/control/indexed_component_control.py) |
| `indexed_component_control_resource_model.py` | [control/indexed_component_control_resource_model.py](../../flagquantum/compiler/registers/control/indexed_component_control_resource_model.py) |
| `indexed_component_resource_model.py` | [control/indexed_component_resource_model.py](../../flagquantum/compiler/registers/control/indexed_component_resource_model.py) |
| `int_vector.py` | [layout/int_vector.py](../../flagquantum/compiler/registers/layout/int_vector.py) |
| `integer_product.py` | [arithmetic/integer_product.py](../../flagquantum/compiler/registers/arithmetic/integer_product.py) |
| `integer_square.py` | [arithmetic/integer_square.py](../../flagquantum/compiler/registers/arithmetic/integer_square.py) |
| `inverse_algorithm.py` | [modular/inversion/inverse_algorithm.py](../../flagquantum/compiler/registers/modular/inversion/inverse_algorithm.py) |
| `inverse_arithmetic.py` | [modular/inversion/inverse_arithmetic.py](../../flagquantum/compiler/registers/modular/inversion/inverse_arithmetic.py) |
| `inverse_exchange.py` | [modular/inversion/inverse_exchange.py](../../flagquantum/compiler/registers/modular/inversion/inverse_exchange.py) |
| `inverse_iteration.py` | [modular/inversion/inverse_iteration.py](../../flagquantum/compiler/registers/modular/inversion/inverse_iteration.py) |
| `inverse_ordering.py` | [modular/inversion/inverse_ordering.py](../../flagquantum/compiler/registers/modular/inversion/inverse_ordering.py) |
| `inverse_parity.py` | [modular/inversion/inverse_parity.py](../../flagquantum/compiler/registers/modular/inversion/inverse_parity.py) |
| `inverse_scaling.py` | [modular/inversion/inverse_scaling.py](../../flagquantum/compiler/registers/modular/inversion/inverse_scaling.py) |
| `inverse_sqrt.py` | [fixed_point/inverse_sqrt.py](../../flagquantum/compiler/registers/fixed_point/inverse_sqrt.py) |
| `inverse_sqrt_newton.py` | [fixed_point/inverse_sqrt_newton.py](../../flagquantum/compiler/registers/fixed_point/inverse_sqrt_newton.py) |
| `inverse_sqrt_seed.py` | [fixed_point/inverse_sqrt_seed.py](../../flagquantum/compiler/registers/fixed_point/inverse_sqrt_seed.py) |
| `inverse_termination.py` | [modular/inversion/inverse_termination.py](../../flagquantum/compiler/registers/modular/inversion/inverse_termination.py) |
| `kickmix_addition.py` | [arithmetic/inplace_addition.py](../../flagquantum/compiler/registers/arithmetic/inplace_addition.py) |
| `kickmix_bit_rotation.py` | [bits/bit_rotation.py](../../flagquantum/compiler/registers/bits/bit_rotation.py) |
| `kickmix_carry_xor.py` | [arithmetic/carry_parity.py](../../flagquantum/compiler/registers/arithmetic/carry_parity.py) |
| `kickmix_comparison.py` | [comparison/mixed_comparison.py](../../flagquantum/compiler/registers/comparison/mixed_comparison.py) |
| `kickmix_controlled_addition.py` | [arithmetic/carry_controlled_addition.py](../../flagquantum/compiler/registers/arithmetic/carry_controlled_addition.py) |
| `kickmix_field_division.py` | [finite_field/division_chain.py](../../flagquantum/compiler/registers/finite_field/division_chain.py) |
| `kickmix_field_erasure.py` | [finite_field/product_erasure.py](../../flagquantum/compiler/registers/finite_field/product_erasure.py) |
| `kickmix_field_inverse.py` | [finite_field/inverse_chain.py](../../flagquantum/compiler/registers/finite_field/inverse_chain.py) |
| `kickmix_field_linear.py` | [finite_field/binary_linear_map.py](../../flagquantum/compiler/registers/finite_field/binary_linear_map.py) |
| `kickmix_field_product.py` | [finite_field/polynomial_product.py](../../flagquantum/compiler/registers/finite_field/polynomial_product.py) |
| `kickmix_increment.py` | [arithmetic/increment.py](../../flagquantum/compiler/registers/arithmetic/increment.py) |
| `kickmix_low_space_comparison.py` | [comparison/low_space_comparison.py](../../flagquantum/compiler/registers/comparison/low_space_comparison.py) |
| `kickmix_mixed_addition.py` | [arithmetic/mixed_addition.py](../../flagquantum/compiler/registers/arithmetic/mixed_addition.py) |
| `kickmix_modular_addition.py` | [modular/mixed_modular_addition.py](../../flagquantum/compiler/registers/modular/mixed_modular_addition.py) |
| `kickmix_modular_doubling.py` | [modular/exact_modular_doubling.py](../../flagquantum/compiler/registers/modular/exact_modular_doubling.py) |
| `kickmix_modular_doubling_dispatch.py` | [modular/modular_doubling_dispatch.py](../../flagquantum/compiler/registers/modular/modular_doubling_dispatch.py) |
| `kickmix_modular_negation.py` | [modular/mixed_modular_negation.py](../../flagquantum/compiler/registers/modular/mixed_modular_negation.py) |
| `kickmix_near_power_doubling.py` | [modular/near_power_doubling.py](../../flagquantum/compiler/registers/modular/near_power_doubling.py) |
| `kickmix_prefix_phase.py` | [phase/masked_prefix_phase.py](../../flagquantum/compiler/registers/phase/masked_prefix_phase.py) |
| `kickmix_qft_measurement.py` | [phase/qft_measurement.py](../../flagquantum/compiler/registers/phase/qft_measurement.py) |
| `kickmix_simple_addition.py` | [arithmetic/measured_carry_addition.py](../../flagquantum/compiler/registers/arithmetic/measured_carry_addition.py) |
| `kickmix_unary_iteration.py` | [lookup/recursive_unary_iteration.py](../../flagquantum/compiler/registers/lookup/recursive_unary_iteration.py) |
| `kickmix_vented_addition.py` | [arithmetic/vented_addition.py](../../flagquantum/compiler/registers/arithmetic/vented_addition.py) |
| `less_equal.py` | [comparison/less_equal.py](../../flagquantum/compiler/registers/comparison/less_equal.py) |
| `linear_comparison.py` | [comparison/linear_comparison.py](../../flagquantum/compiler/registers/comparison/linear_comparison.py) |
| `measured_unlookup.py` | [lookup/measured_unlookup.py](../../flagquantum/compiler/registers/lookup/measured_unlookup.py) |
| `measurement.py` | [primitives/measurement.py](../../flagquantum/compiler/registers/primitives/measurement.py) |
| `modular_addition.py` | [modular/modular_addition.py](../../flagquantum/compiler/registers/modular/modular_addition.py) |
| `modular_constant_addition.py` | [modular/modular_constant_addition.py](../../flagquantum/compiler/registers/modular/modular_constant_addition.py) |
| `modular_declarations.py` | [modular/modular_declarations.py](../../flagquantum/compiler/registers/modular/modular_declarations.py) |
| `modular_doubling.py` | [modular/modular_doubling.py](../../flagquantum/compiler/registers/modular/modular_doubling.py) |
| `modular_halving.py` | [modular/modular_halving.py](../../flagquantum/compiler/registers/modular/modular_halving.py) |
| `modular_inverse.py` | [modular/modular_inverse.py](../../flagquantum/compiler/registers/modular/modular_inverse.py) |
| `modular_multiplication.py` | [modular/modular_multiplication.py](../../flagquantum/compiler/registers/modular/modular_multiplication.py) |
| `modular_negation.py` | [modular/modular_negation.py](../../flagquantum/compiler/registers/modular/modular_negation.py) |
| `modular_scaled_addition.py` | [modular/modular_scaled_addition.py](../../flagquantum/compiler/registers/modular/modular_scaled_addition.py) |
| `modular_subtraction.py` | [modular/modular_subtraction.py](../../flagquantum/compiler/registers/modular/modular_subtraction.py) |
| `multi_control.py` | [control/multi_control.py](../../flagquantum/compiler/registers/control/multi_control.py) |
| `multi_controlled_constant_addition.py` | [arithmetic/multi_controlled_constant_addition.py](../../flagquantum/compiler/registers/arithmetic/multi_controlled_constant_addition.py) |
| `multiplexed_swap.py` | [lookup/multiplexed_swap.py](../../flagquantum/compiler/registers/lookup/multiplexed_swap.py) |
| `ordering.py` | [comparison/ordering.py](../../flagquantum/compiler/registers/comparison/ordering.py) |
| `out_of_place_add_resource_model.py` | [arithmetic/out_of_place_add_resource_model.py](../../flagquantum/compiler/registers/arithmetic/out_of_place_add_resource_model.py) |
| `out_of_place_addition.py` | [arithmetic/out_of_place_addition.py](../../flagquantum/compiler/registers/arithmetic/out_of_place_addition.py) |
| `parallel_comparison.py` | [comparison/parallel_comparison.py](../../flagquantum/compiler/registers/comparison/parallel_comparison.py) |
| `partition.py` | [layout/partition.py](../../flagquantum/compiler/registers/layout/partition.py) |
| `pauli_gate.py` | [primitives/pauli_gate.py](../../flagquantum/compiler/registers/primitives/pauli_gate.py) |
| `pauli_rotation.py` | [primitives/pauli_rotation.py](../../flagquantum/compiler/registers/primitives/pauli_rotation.py) |
| `permutation.py` | [sorting/permutation.py](../../flagquantum/compiler/registers/sorting/permutation.py) |
| `permutation_index.py` | [sorting/permutation_index.py](../../flagquantum/compiler/registers/sorting/permutation_index.py) |
| `permutation_resource_model.py` | [sorting/permutation_resource_model.py](../../flagquantum/compiler/registers/sorting/permutation_resource_model.py) |
| `phase_addition.py` | [phase/phase_addition.py](../../flagquantum/compiler/registers/phase/phase_addition.py) |
| `phase_family.py` | [phase/phase_family.py](../../flagquantum/compiler/registers/phase/phase_family.py) |
| `phase_gradient.py` | [phase/phase_gradient.py](../../flagquantum/compiler/registers/phase/phase_gradient.py) |
| `phase_gradient_lookup.py` | [phase/phase_gradient_lookup.py](../../flagquantum/compiler/registers/phase/phase_gradient_lookup.py) |
| `polynomial.py` | [finite_field/polynomial.py](../../flagquantum/compiler/registers/finite_field/polynomial.py) |
| `power.py` | [primitives/power.py](../../flagquantum/compiler/registers/primitives/power.py) |
| `prepare_reflection.py` | [preparation/prepare_reflection.py](../../flagquantum/compiler/registers/preparation/prepare_reflection.py) |
| `product_accumulation.py` | [arithmetic/product_accumulation.py](../../flagquantum/compiler/registers/arithmetic/product_accumulation.py) |
| `programmable_rotation_array.py` | [phase/programmable_rotation_array.py](../../flagquantum/compiler/registers/phase/programmable_rotation_array.py) |
| `programmed_givens.py` | [phase/programmed_givens.py](../../flagquantum/compiler/registers/phase/programmed_givens.py) |
| `programmed_phase.py` | [phase/programmed_phase.py](../../flagquantum/compiler/registers/phase/programmed_phase.py) |
| `qroam.py` | [lookup/qroam.py](../../flagquantum/compiler/registers/lookup/qroam.py) |
| `qroam_bank_binding.py` | [lookup/qroam_bank_binding.py](../../flagquantum/compiler/registers/lookup/qroam_bank_binding.py) |
| `qroam_clean_erasure.py` | [lookup/qroam_clean_erasure.py](../../flagquantum/compiler/registers/lookup/qroam_clean_erasure.py) |
| `qroam_resource_model.py` | [lookup/qroam_resource_model.py](../../flagquantum/compiler/registers/lookup/qroam_resource_model.py) |
| `qrom.py` | [lookup/qrom.py](../../flagquantum/compiler/registers/lookup/qrom.py) |
| `qrom_resource_model.py` | [lookup/qrom_resource_model.py](../../flagquantum/compiler/registers/lookup/qrom_resource_model.py) |
| `qrom_table.py` | [lookup/qrom_table.py](../../flagquantum/compiler/registers/lookup/qrom_table.py) |
| `qvr_phase_gradient.py` | [phase/qvr_phase_gradient.py](../../flagquantum/compiler/registers/phase/qvr_phase_gradient.py) |
| `qvr_zpow.py` | [phase/qvr_zpow.py](../../flagquantum/compiler/registers/phase/qvr_zpow.py) |
| `rank_uniform_preparation.py` | [preparation/rank_uniform_preparation.py](../../flagquantum/compiler/registers/preparation/rank_uniform_preparation.py) |
| `reciprocal.py` | [fixed_point/reciprocal.py](../../flagquantum/compiler/registers/fixed_point/reciprocal.py) |
| `reciprocal_newton.py` | [fixed_point/reciprocal_newton.py](../../flagquantum/compiler/registers/fixed_point/reciprocal_newton.py) |
| `reciprocal_seed.py` | [fixed_point/reciprocal_seed.py](../../flagquantum/compiler/registers/fixed_point/reciprocal_seed.py) |
| `recursive_lookup.py` | [lookup/recursive_lookup.py](../../flagquantum/compiler/registers/lookup/recursive_lookup.py) |
| `register_bits.py` | [layout/register_bits.py](../../flagquantum/compiler/registers/layout/register_bits.py) |
| `register_cast.py` | [layout/register_cast.py](../../flagquantum/compiler/registers/layout/register_cast.py) |
| `relative_controlled_swap.py` | [primitives/relative_controlled_swap.py](../../flagquantum/compiler/registers/primitives/relative_controlled_swap.py) |
| `rz_phase_gradient.py` | [phase/rz_phase_gradient.py](../../flagquantum/compiler/registers/phase/rz_phase_gradient.py) |
| `scaled_phase_addition.py` | [phase/scaled_phase_addition.py](../../flagquantum/compiler/registers/phase/scaled_phase_addition.py) |
| `select_swap_qrom.py` | [lookup/select_swap_qrom.py](../../flagquantum/compiler/registers/lookup/select_swap_qrom.py) |
| `sign_extension.py` | [bits/sign_extension.py](../../flagquantum/compiler/registers/bits/sign_extension.py) |
| `signed_arcsine.py` | [fixed_point/signed_arcsine.py](../../flagquantum/compiler/registers/fixed_point/signed_arcsine.py) |
| `signed_encoding.py` | [bits/signed_encoding.py](../../flagquantum/compiler/registers/bits/signed_encoding.py) |
| `single_window_mod_mul.py` | [modular/single_window_mod_mul.py](../../flagquantum/compiler/registers/modular/single_window_mod_mul.py) |
| `sorted_duplicates.py` | [sorting/sorted_duplicates.py](../../flagquantum/compiler/registers/sorting/sorted_duplicates.py) |
| `source_qroam_clean.py` | [lookup/source_qroam_clean.py](../../flagquantum/compiler/registers/lookup/source_qroam_clean.py) |
| `source_qrom.py` | [lookup/source_qrom.py](../../flagquantum/compiler/registers/lookup/source_qrom.py) |
| `square_cost_phasing.py` | [phase/square_cost_phasing.py](../../flagquantum/compiler/registers/phase/square_cost_phasing.py) |
| `squared_distance.py` | [fixed_point/squared_distance.py](../../flagquantum/compiler/registers/fixed_point/squared_distance.py) |
| `su2_rotation.py` | [phase/su2_rotation.py](../../flagquantum/compiler/registers/phase/su2_rotation.py) |
| `subtraction.py` | [arithmetic/subtraction.py](../../flagquantum/compiler/registers/arithmetic/subtraction.py) |
| `swap.py` | [primitives/swap.py](../../flagquantum/compiler/registers/primitives/swap.py) |
| `swap_with_zero.py` | [lookup/swap_with_zero.py](../../flagquantum/compiler/registers/lookup/swap_with_zero.py) |
| `symbolic_qroam_resource_model.py` | [lookup/symbolic_qroam_resource_model.py](../../flagquantum/compiler/registers/lookup/symbolic_qroam_resource_model.py) |
| `symbolic_resource_model.py` | [resources/symbolic_resource_model.py](../../flagquantum/compiler/registers/resources/symbolic_resource_model.py) |
| `symmetric_difference.py` | [sorting/symmetric_difference.py](../../flagquantum/compiler/registers/sorting/symmetric_difference.py) |
| `triangular_uniform_preparation.py` | [preparation/triangular_uniform_preparation.py](../../flagquantum/compiler/registers/preparation/triangular_uniform_preparation.py) |
| `unary_iteration.py` | [lookup/unary_iteration.py](../../flagquantum/compiler/registers/lookup/unary_iteration.py) |
| `uniform_preparation.py` | [preparation/uniform_preparation.py](../../flagquantum/compiler/registers/preparation/uniform_preparation.py) |
| `uniform_preparation_resource_model.py` | [preparation/uniform_preparation_resource_model.py](../../flagquantum/compiler/registers/preparation/uniform_preparation_resource_model.py) |
| `windowed_mod_mul.py` | [modular/windowed_mod_mul.py](../../flagquantum/compiler/registers/modular/windowed_mod_mul.py) |
| `workspace.py` | [layout/workspace.py](../../flagquantum/compiler/registers/layout/workspace.py) |
| `xor_resource_model.py` | [bits/xor_resource_model.py](../../flagquantum/compiler/registers/bits/xor_resource_model.py) |
| `zpow_phase_gradient.py` | [phase/zpow_phase_gradient.py](../../flagquantum/compiler/registers/phase/zpow_phase_gradient.py) |

## Local acceptance

The locked reference environment loads Qualtran 0.7.0 and a Kickmix extension
with `Simulator`, with Torch imported before pytest. Nine previously recorded
reference source blobs were rechecked against their Git blob hashes.

| Check | Observed result |
| --- | --- |
| Component relocation | 191 modules in 14 functional packages, plus `modular/inversion/` |
| Functional filenames | 24 upstream-prefixed implementation filenames renamed |
| Existing Python consumers | 676 modules have identical non-import ASTs and canonical import dependencies against the base revision |
| Fresh package import | All 206 component modules and subpackages import successfully |
| Affected regression | 17,522 passed across 315 test files, no skips, 1,078 warnings, 903.70 seconds |
| Static typing | Strict mypy targeting Python 3.12 passes for all 191 moved modules and three checked cross-domain consumers |
| Formatting and imports | Ruff and Black pass for 692 Python files |
| Boundaries | Architecture, public API baseline and internal multi-level IR contract checks pass |
| Documentation | All local links resolve in 17 package guides and mapping documents |
| Executable consumers | Concrete QROM, symbolic QROM and resource-model comparison examples pass |

The three-bit Add example reports two logical AND components, two measurements
and eight peak logical qubits. Its emitted stream contains eight T/Tdag gates.
The reports retain the separate model and emitted-count fields.

The affected regression emits NumPy conversion, overflow and matrix-operation
warnings. This record reports the warnings with the test result.

### Broader checks and baseline limitations

The default smoke/unit tier was also attempted with four workers. It stopped
with 12,801 passed, one failed, 51 skipped and 398 warnings. The failed test is
`tests/team/simulation/test_statevector_ops_numerics.py::test_native_fused_rotation_layer_only_promotes_terminal_regions`.
It expects one six-region native rotation layer and observes an empty tuple.
An isolated source snapshot of base revision `0bc67ecc` reproduces the identical
failure in the same environment. Optional-backend dependency skips are included
in the broader result. The full default tier therefore has an open baseline
failure.

The strict Python 3.10 target reports an unused `no-redef` suppression in the
existing `simulation/numerics/conformance.py` fallback import. The accepted
strict type-check scope above targets Python 3.12.

Persistent local logs are named `g2-register-organization-tests.log`,
`g2-register-organization-default.log`,
`g2-register-organization-baseline-failure.log` and
`g2-register-organization-examples.log` in the existing reference environment.

A fresh post-format run of ten focused component, format and execution test
files passes all 343 cases without warnings or skips. An additional single-rank
Torch/Gloo execution test passes with loopback access enabled. Its initial
sandbox run failed to resolve `127.0.0.1`. These supplementary outcomes are
recorded in `g2-register-organization-postformat-components.log` and
`g2-register-organization-loopback.log`.
