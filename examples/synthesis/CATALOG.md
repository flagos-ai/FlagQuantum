# Classified Classiq Qmod cases

All **216** pinned original paths are classified here. **27** have a byte-matched corresponding native example; the other native entries are explicitly marked missing.

**211** cases link existing Circuit validation sources. A link locates a validation implementation; its own bindings and scope determine what it proves.

The approved numerical revision passes 216/216 execution smoke checks. The original snapshot has 210 passes and six source-semantic refusals.

```sh
python -m examples.synthesis.run_qmod --verify
python -m examples.synthesis.run_qmod --all --revision approved --output /tmp/qmod-examples.json
```

Use each case README for exact bindings, resource records, sources and validation links. Teaching examples in the other indexes may use smaller illustrative programs.

## Tutorials (102)

### advanced

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [tutorials/advanced_tutorials/discrete_quantum_walk/discrete_quantum_walk.qmod](tutorials/advanced/classiq/discrete_quantum_walk/discrete_quantum_walk/README.md) | Not provided | 2 | execution_smoke_passed |
| [tutorials/advanced_tutorials/discrete_quantum_walk/quantum_walk_circle.qmod](tutorials/advanced/classiq/discrete_quantum_walk/quantum_walk_circle/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/advanced_tutorials/discrete_quantum_walk/quantum_walk_circle_balanced_coin.qmod](tutorials/advanced/classiq/discrete_quantum_walk/quantum_walk_circle_balanced_coin/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/advanced_tutorials/discrete_quantum_walk/quantum_walk_hypercube.qmod](tutorials/advanced/classiq/discrete_quantum_walk/quantum_walk_hypercube/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/advanced_tutorials/discrete_quantum_walk_2d/discrete_time_quantum_walk.qmod](tutorials/advanced/classiq/discrete_quantum_walk_2d/discrete_time_quantum_walk/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/advanced_tutorials/linear_approximation_of_tanh/tanh.qmod](tutorials/advanced/classiq/linear_approximation_of_tanh/tanh/README.md) | Not provided | 1 | execution_smoke_passed |

### arithmetic

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [functions/function_usage_examples/arithmetic/arithmetic_expression/arithmetic_expression.qmod](tutorials/arithmetic/classiq/function_usage_examples/arithmetic/arithmetic_expression/arithmetic_expression/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/function_usage_examples/arithmetic/arithmetic_expression/arithmetic_expression_example.qmod](tutorials/arithmetic/classiq/function_usage_examples/arithmetic/arithmetic_expression/arithmetic_expression_example/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/function_usage_examples/arithmetic/bitwise_and/bitwise_and_2vars_example.qmod](tutorials/arithmetic/classiq/function_usage_examples/arithmetic/bitwise_and/bitwise_and_2vars_example/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/function_usage_examples/arithmetic/bitwise_and/bitwise_and_integer_example.qmod](tutorials/arithmetic/classiq/function_usage_examples/arithmetic/bitwise_and/bitwise_and_integer_example/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/function_usage_examples/arithmetic/bitwise_invert/bitwise_invert_example.qmod](tutorials/arithmetic/classiq/function_usage_examples/arithmetic/bitwise_invert/bitwise_invert_example/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/function_usage_examples/arithmetic/bitwise_or/bitwise_or_2vars_example.qmod](tutorials/arithmetic/classiq/function_usage_examples/arithmetic/bitwise_or/bitwise_or_2vars_example/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/function_usage_examples/arithmetic/bitwise_or/bitwise_or_integer_example.qmod](tutorials/arithmetic/classiq/function_usage_examples/arithmetic/bitwise_or/bitwise_or_integer_example/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/function_usage_examples/arithmetic/bitwise_xor/bitwise_xor_2vars_example.qmod](tutorials/arithmetic/classiq/function_usage_examples/arithmetic/bitwise_xor/bitwise_xor_2vars_example/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/function_usage_examples/arithmetic/bitwise_xor/bitwise_xor_integer_example.qmod](tutorials/arithmetic/classiq/function_usage_examples/arithmetic/bitwise_xor/bitwise_xor_integer_example/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/function_usage_examples/arithmetic/comparator/comparator_2vars_example.qmod](tutorials/arithmetic/classiq/function_usage_examples/arithmetic/comparator/comparator_2vars_example/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/function_usage_examples/arithmetic/comparator/comparator_integer_example.qmod](tutorials/arithmetic/classiq/function_usage_examples/arithmetic/comparator/comparator_integer_example/README.md) | Not provided | 0 | execution_smoke_passed |
| [functions/function_usage_examples/arithmetic/extremum/maximum_float_example.qmod](tutorials/arithmetic/classiq/function_usage_examples/arithmetic/extremum/maximum_float_example/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/function_usage_examples/arithmetic/extremum/maximum_integer_example.qmod](tutorials/arithmetic/classiq/function_usage_examples/arithmetic/extremum/maximum_integer_example/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/function_usage_examples/arithmetic/extremum/minimum_2vars_example.qmod](tutorials/arithmetic/classiq/function_usage_examples/arithmetic/extremum/minimum_2vars_example/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/function_usage_examples/arithmetic/modular_exp/modular_exp_example.qmod](tutorials/arithmetic/classiq/function_usage_examples/arithmetic/modular_exp/modular_exp_example/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/function_usage_examples/arithmetic/modulo/modulo_example.qmod](tutorials/arithmetic/classiq/function_usage_examples/arithmetic/modulo/modulo_example/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/function_usage_examples/arithmetic/multiplication/multiplication_2vars_example.qmod](tutorials/arithmetic/classiq/function_usage_examples/arithmetic/multiplication/multiplication_2vars_example/README.md) | Available | 1 | execution_smoke_passed |
| [functions/function_usage_examples/arithmetic/multiplication/multiplication_float_example.qmod](tutorials/arithmetic/classiq/function_usage_examples/arithmetic/multiplication/multiplication_float_example/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/function_usage_examples/arithmetic/negation/negation_example.qmod](tutorials/arithmetic/classiq/function_usage_examples/arithmetic/negation/negation_example/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/function_usage_examples/arithmetic/subtraction/subtraction_2vars_example.qmod](tutorials/arithmetic/classiq/function_usage_examples/arithmetic/subtraction/subtraction_2vars_example/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/function_usage_examples/arithmetic/subtraction/subtraction_float_example.qmod](tutorials/arithmetic/classiq/function_usage_examples/arithmetic/subtraction/subtraction_float_example/README.md) | Not provided | 1 | execution_smoke_passed |

### basics

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [tutorials/basic_tutorials/add_bell_states/add_bell_states.qmod](tutorials/basics/classiq/add_bell_states/add_bell_states/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/basic_tutorials/entanglement/entanglement.qmod](tutorials/basics/classiq/entanglement/entanglement/README.md) | Available | 2 | execution_smoke_passed |
| [tutorials/basic_tutorials/exponentiation/example_exponentiation.qmod](tutorials/basics/classiq/exponentiation/example_exponentiation/README.md) | Not provided | 2 | execution_smoke_passed |
| [tutorials/basic_tutorials/grover_graph_coloring/grover_graph_coloring.qmod](tutorials/basics/classiq/grover_graph_coloring/grover_graph_coloring/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/basic_tutorials/mcx/mcx.qmod](tutorials/basics/classiq/mcx/mcx/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/basic_tutorials/optimization/learning_optimization.qmod](tutorials/basics/classiq/optimization/learning_optimization/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/basic_tutorials/prepare_state/prepare_state.qmod](tutorials/basics/classiq/prepare_state/prepare_state/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/basic_tutorials/qmci_pi_estimation/qmci_pi_estimation.qmod](tutorials/basics/classiq/qmci_pi_estimation/qmci_pi_estimation/README.md) | Not provided | 2 | execution_smoke_passed |
| [tutorials/basic_tutorials/qml_with_classiq_guide/qml_with_classiq_guide.qmod](tutorials/basics/classiq/qml_with_classiq_guide/qml_with_classiq_guide/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/basic_tutorials/quantum_primitives/hadamard_test_tutorial/hadamard_test_tutorial.qmod](tutorials/basics/classiq/quantum_primitives/hadamard_test_tutorial/hadamard_test_tutorial/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/basic_tutorials/quantum_primitives/linear_combination_of_unitaries/linear_combination_of_unitaries.qmod](tutorials/basics/classiq/quantum_primitives/linear_combination_of_unitaries/linear_combination_of_unitaries/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/basic_tutorials/quantumwalk_complex_network/quantumwalk_complex_network.qmod](tutorials/basics/classiq/quantumwalk_complex_network/quantumwalk_complex_network/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/basic_tutorials/the_classiq_tutorial/Qmod_tutorial_part1.qmod](tutorials/basics/classiq/the_classiq_tutorial/Qmod_tutorial_part1/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/basic_tutorials/the_classiq_tutorial/Qmod_tutorial_part2.qmod](tutorials/basics/classiq/the_classiq_tutorial/Qmod_tutorial_part2/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/basic_tutorials/the_classiq_tutorial/classiq_overview_tutorial.qmod](tutorials/basics/classiq/the_classiq_tutorial/classiq_overview_tutorial/README.md) | Available | 1 | execution_smoke_passed |
| [tutorials/basic_tutorials/the_classiq_tutorial/execution_tutorial.qmod](tutorials/basics/classiq/the_classiq_tutorial/execution_tutorial/README.md) | Available | 1 | execution_smoke_passed |
| [tutorials/basic_tutorials/the_classiq_tutorial/execution_tutorial_part2.qmod](tutorials/basics/classiq/the_classiq_tutorial/execution_tutorial_part2/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/basic_tutorials/the_classiq_tutorial/synthesis_tutorial.qmod](tutorials/basics/classiq/the_classiq_tutorial/synthesis_tutorial/README.md) | Available | 1 | execution_smoke_passed |

### evolution

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [functions/qmod_library_reference/qmod_core_library/hamiltonian_evolution/qdrift/qdrift.qmod](tutorials/evolution/classiq/qmod_library_reference/qmod_core_library/hamiltonian_evolution/qdrift/qdrift/README.md) | Not provided | 0 | execution_smoke_passed |
| [functions/qmod_library_reference/qmod_core_library/hamiltonian_evolution/suzuki_trotter/suzuki_trotter.qmod](tutorials/evolution/classiq/qmod_library_reference/qmod_core_library/hamiltonian_evolution/suzuki_trotter/suzuki_trotter/README.md) | Not provided | 1 | execution_smoke_passed |

### gates

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [functions/function_usage_examples/mcx/mcx_example.qmod](tutorials/gates/classiq/function_usage_examples/mcx/mcx_example/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/qmod_library_reference/qmod_core_library/standard_gates/CRX.qmod](tutorials/gates/classiq/qmod_library_reference/qmod_core_library/standard_gates/CRX/README.md) | Available | 2 | execution_smoke_passed |
| [functions/qmod_library_reference/qmod_core_library/standard_gates/CX.qmod](tutorials/gates/classiq/qmod_library_reference/qmod_core_library/standard_gates/CX/README.md) | Available | 2 | execution_smoke_passed |
| [functions/qmod_library_reference/qmod_core_library/standard_gates/PHASE.qmod](tutorials/gates/classiq/qmod_library_reference/qmod_core_library/standard_gates/PHASE/README.md) | Available | 2 | execution_smoke_passed |
| [functions/qmod_library_reference/qmod_core_library/standard_gates/R.qmod](tutorials/gates/classiq/qmod_library_reference/qmod_core_library/standard_gates/R/README.md) | Available | 2 | execution_smoke_passed |
| [functions/qmod_library_reference/qmod_core_library/standard_gates/RZ.qmod](tutorials/gates/classiq/qmod_library_reference/qmod_core_library/standard_gates/RZ/README.md) | Available | 2 | execution_smoke_passed |
| [functions/qmod_library_reference/qmod_core_library/standard_gates/RZZ.qmod](tutorials/gates/classiq/qmod_library_reference/qmod_core_library/standard_gates/RZZ/README.md) | Available | 1 | execution_smoke_passed |
| [functions/qmod_library_reference/qmod_core_library/standard_gates/SWAP.qmod](tutorials/gates/classiq/qmod_library_reference/qmod_core_library/standard_gates/SWAP/README.md) | Available | 2 | execution_smoke_passed |
| [functions/qmod_library_reference/qmod_core_library/standard_gates/U.qmod](tutorials/gates/classiq/qmod_library_reference/qmod_core_library/standard_gates/U/README.md) | Available | 2 | execution_smoke_passed |
| [functions/qmod_library_reference/qmod_core_library/standard_gates/X.qmod](tutorials/gates/classiq/qmod_library_reference/qmod_core_library/standard_gates/X/README.md) | Available | 1 | execution_smoke_passed |

### library

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [functions/qmod_library_reference/classiq_open_library/amplitude_amplification/exact_amplitude_amplification.qmod](tutorials/library/classiq/qmod_library_reference/classiq_open_library/amplitude_amplification/exact_amplitude_amplification/README.md) | Available | 1 | execution_smoke_passed |
| [functions/qmod_library_reference/classiq_open_library/grover_operator/grover_operator.qmod](tutorials/library/classiq/qmod_library_reference/classiq_open_library/grover_operator/grover_operator/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/qmod_library_reference/classiq_open_library/hadamard_transform/hadamard_transform.qmod](tutorials/library/classiq/qmod_library_reference/classiq_open_library/hadamard_transform/hadamard_transform/README.md) | Available | 2 | execution_smoke_passed |
| [functions/qmod_library_reference/classiq_open_library/linear_pauli_rotations/linear_pauli_rotations.qmod](tutorials/library/classiq/qmod_library_reference/classiq_open_library/linear_pauli_rotations/linear_pauli_rotations/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/qmod_library_reference/classiq_open_library/qct_qst/qct_qst_type1.qmod](tutorials/library/classiq/qmod_library_reference/classiq_open_library/qct_qst/qct_qst_type1/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/qmod_library_reference/classiq_open_library/qct_qst/qct_qst_type2.qmod](tutorials/library/classiq/qmod_library_reference/classiq_open_library/qct_qst/qct_qst_type2/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/qmod_library_reference/classiq_open_library/qct_qst/qct_type2.qmod](tutorials/library/classiq/qmod_library_reference/classiq_open_library/qct_qst/qct_type2/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/qmod_library_reference/classiq_open_library/qct_qst/qst_type2.qmod](tutorials/library/classiq/qmod_library_reference/classiq_open_library/qct_qst/qst_type2/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/qmod_library_reference/classiq_open_library/qft/qft.qmod](tutorials/library/classiq/qmod_library_reference/classiq_open_library/qft/qft/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/qmod_library_reference/classiq_open_library/qpe/qpe.qmod](tutorials/library/classiq/qmod_library_reference/classiq_open_library/qpe/qpe/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/qmod_library_reference/classiq_open_library/qpe/qpe_flexible.qmod](tutorials/library/classiq/qmod_library_reference/classiq_open_library/qpe/qpe_flexible/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/qmod_library_reference/classiq_open_library/qsvt/qsvt.qmod](tutorials/library/classiq/qmod_library_reference/classiq_open_library/qsvt/qsvt/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/qmod_library_reference/qmod_core_library/unitary/unitary.qmod](tutorials/library/classiq/qmod_library_reference/qmod_core_library/unitary/unitary/README.md) | Not provided | 1 | execution_smoke_passed |

### state preparation

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [functions/qmod_library_reference/classiq_open_library/special_state_preparations/prepare_bell_state.qmod](tutorials/state_preparation/classiq/qmod_library_reference/classiq_open_library/special_state_preparations/prepare_bell_state/README.md) | Available | 2 | execution_smoke_passed |
| [functions/qmod_library_reference/classiq_open_library/special_state_preparations/prepare_exponential_state.qmod](tutorials/state_preparation/classiq/qmod_library_reference/classiq_open_library/special_state_preparations/prepare_exponential_state/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/qmod_library_reference/classiq_open_library/special_state_preparations/prepare_ghz_state.qmod](tutorials/state_preparation/classiq/qmod_library_reference/classiq_open_library/special_state_preparations/prepare_ghz_state/README.md) | Available | 1 | execution_smoke_passed |
| [functions/qmod_library_reference/classiq_open_library/special_state_preparations/prepare_uniform_interval_state.qmod](tutorials/state_preparation/classiq/qmod_library_reference/classiq_open_library/special_state_preparations/prepare_uniform_interval_state/README.md) | Available | 1 | execution_smoke_passed |
| [functions/qmod_library_reference/classiq_open_library/special_state_preparations/prepare_uniform_trimmed_state.qmod](tutorials/state_preparation/classiq/qmod_library_reference/classiq_open_library/special_state_preparations/prepare_uniform_trimmed_state/README.md) | Available | 1 | execution_smoke_passed |
| [functions/qmod_library_reference/classiq_open_library/variational_data_encoding/encode_in_angle.qmod](tutorials/state_preparation/classiq/qmod_library_reference/classiq_open_library/variational_data_encoding/encode_in_angle/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/qmod_library_reference/classiq_open_library/variational_data_encoding/encode_on_bloch.qmod](tutorials/state_preparation/classiq/qmod_library_reference/classiq_open_library/variational_data_encoding/encode_on_bloch/README.md) | Not provided | 1 | execution_smoke_passed |
| [functions/qmod_library_reference/qmod_core_library/prepare_state_and_amplitudes/example_prepare_amplitudes.qmod](tutorials/state_preparation/classiq/qmod_library_reference/qmod_core_library/prepare_state_and_amplitudes/example_prepare_amplitudes/README.md) | Available | 2 | execution_smoke_passed |
| [functions/qmod_library_reference/qmod_core_library/prepare_state_and_amplitudes/example_prepare_state.qmod](tutorials/state_preparation/classiq/qmod_library_reference/qmod_core_library/prepare_state_and_amplitudes/example_prepare_state/README.md) | Available | 2 | execution_smoke_passed |

### techniques

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [tutorials/technology_demonstrations/QPE/qpe_for_grover_operator/qpe_for_grover_operator.qmod](tutorials/techniques/classiq/QPE/qpe_for_grover_operator/qpe_for_grover_operator/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/technology_demonstrations/QPE/qpe_for_unitary_matrix/qpe_for_unitary_matrix.qmod](tutorials/techniques/classiq/QPE/qpe_for_unitary_matrix/qpe_for_unitary_matrix/README.md) | Not provided | 2 | execution_smoke_passed |
| [tutorials/technology_demonstrations/approximated_state_preparation/approximated_state_preparation.qmod](tutorials/techniques/classiq/approximated_state_preparation/approximated_state_preparation/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/technology_demonstrations/arithmetic_expressions/arithmetic_expressions.qmod](tutorials/techniques/classiq/arithmetic_expressions/arithmetic_expressions/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/technology_demonstrations/auxiliary_managment/auxiliary_management.qmod](tutorials/techniques/classiq/auxiliary_managment/auxiliary_management/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/technology_demonstrations/classiq_paper/qsvt/classiq_qsvt.qmod](tutorials/techniques/classiq/classiq_paper/qsvt/classiq_qsvt/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/technology_demonstrations/classiq_paper/quantum_walk/classiq_discrete_quantum_walk.qmod](tutorials/techniques/classiq/classiq_paper/quantum_walk/classiq_discrete_quantum_walk/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/technology_demonstrations/discrete_quantum_walk_circle/discrete_quantum_walk_circle.qmod](tutorials/techniques/classiq/discrete_quantum_walk_circle/discrete_quantum_walk_circle/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/technology_demonstrations/hamiltonian_evolution/hamiltonian_evolution.qmod](tutorials/techniques/classiq/hamiltonian_evolution/hamiltonian_evolution/README.md) | Not provided | 2 | execution_smoke_passed |
| [tutorials/technology_demonstrations/hardware_aware_mcx/hardware_aware_mcx.qmod](tutorials/techniques/classiq/hardware_aware_mcx/hardware_aware_mcx/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/technology_demonstrations/hhl/hhl_example.qmod](tutorials/techniques/classiq/hhl/hhl_example/README.md) | Not provided | 1 | frontend_refused |
| [tutorials/technology_demonstrations/oracle_generation/3sat_oracles.qmod](tutorials/techniques/classiq/oracle_generation/3sat_oracles/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/technology_demonstrations/qaoa/qaoa_demonstration.qmod](tutorials/techniques/classiq/qaoa/qaoa_demonstration/README.md) | Not provided | 1 | execution_smoke_passed |

### workshops

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [tutorials/workshops/algo_design_QCE_tutorial/algo_design_QCE_tutorial_part_I.qmod](tutorials/workshops/classiq/algo_design_QCE_tutorial/algo_design_QCE_tutorial_part_I/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/workshops/algo_design_QCE_tutorial/algo_design_QCE_tutorial_part_II.qmod](tutorials/workshops/classiq/algo_design_QCE_tutorial/algo_design_QCE_tutorial_part_II/README.md) | Not provided | 2 | execution_smoke_passed |
| [tutorials/workshops/combinatorial_workshop/combinatorial_qmod_workshop_for_maxcut.qmod](tutorials/workshops/classiq/combinatorial_workshop/combinatorial_qmod_workshop_for_maxcut/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/workshops/finance_workshops/Option_Pricing_Workshop.qmod](tutorials/workshops/classiq/finance_workshops/Option_Pricing_Workshop/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/workshops/finance_workshops/combi_workshop_Inequality_constriants_PO.qmod](tutorials/workshops/classiq/finance_workshops/combi_workshop_Inequality_constriants_PO/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/workshops/finance_workshops/combi_workshop_equality_constriants_PO.qmod](tutorials/workshops/classiq/finance_workshops/combi_workshop_equality_constriants_PO/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/workshops/finance_workshops/rainbow_options_workshop_bruteforce.qmod](tutorials/workshops/classiq/finance_workshops/rainbow_options_workshop_bruteforce/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/workshops/grover_workshop/grover_workshop.qmod](tutorials/workshops/classiq/grover_workshop/grover_workshop/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/workshops/hhl_workshop/hhl_workshop.qmod](tutorials/workshops/classiq/hhl_workshop/hhl_workshop/README.md) | Not provided | 1 | execution_smoke_passed |
| [tutorials/workshops/oracle_workshop/oracles_workshop.qmod](tutorials/workshops/classiq/oracle_workshop/oracles_workshop/README.md) | Not provided | 1 | execution_smoke_passed |

## Algorithms (53)

### amplitude estimation

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [algorithms/amplitude_amplification_and_estimation/oblivious_amplitude_amplification/oblivious_amplitude_amplification.qmod](algorithms/amplitude_estimation/classiq/oblivious_amplitude_amplification/oblivious_amplitude_amplification/README.md) | Not provided | 1 | execution_smoke_passed |
| [algorithms/amplitude_amplification_and_estimation/qmc_user_defined/qmc_user_defined.qmod](algorithms/amplitude_estimation/classiq/qmc_user_defined/qmc_user_defined/README.md) | Not provided | 1 | execution_smoke_passed |
| [algorithms/amplitude_amplification_and_estimation/qsvt_fixed_point_amplitude_amplification/qsvt_fixed_point_amplitude_amplification.qmod](algorithms/amplitude_estimation/classiq/qsvt_fixed_point_amplitude_amplification/qsvt_fixed_point_amplitude_amplification/README.md) | Not provided | 1 | execution_smoke_passed |
| [algorithms/amplitude_amplification_and_estimation/quantum_counting/quantum_counting.qmod](algorithms/amplitude_estimation/classiq/quantum_counting/quantum_counting/README.md) | Not provided | 2 | execution_smoke_passed |
| [algorithms/amplitude_amplification_and_estimation/quantum_counting/quantum_counting_iqae.qmod](algorithms/amplitude_estimation/classiq/quantum_counting/quantum_counting_iqae/README.md) | Not provided | 2 | execution_smoke_passed |
| [algorithms/amplitude_amplification_and_estimation/quantum_counting/quantum_counting_qpe.qmod](algorithms/amplitude_estimation/classiq/quantum_counting/quantum_counting_qpe/README.md) | Not provided | 2 | execution_smoke_passed |

### block encoding

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [community/paper_implementation_project/Block_encoding-ND_Laplacian/1D_Periodic_Laplacian_BE.qmod](algorithms/block_encoding/classiq/paper_implementation_project/Block_encoding-ND_Laplacian/1D_Periodic_Laplacian_BE/README.md) | Not provided | 1 | execution_smoke_passed |
| [community/paper_implementation_project/Block_encoding-ND_Laplacian/ND_Periodic_Laplacian_BE.qmod](algorithms/block_encoding/classiq/paper_implementation_project/Block_encoding-ND_Laplacian/ND_Periodic_Laplacian_BE/README.md) | Not provided | 1 | execution_smoke_passed |
| [community/paper_implementation_project/block_encoding/checkerboard_BE_N8.qmod](algorithms/block_encoding/classiq/paper_implementation_project/block_encoding/checkerboard_BE_N8/README.md) | Not provided | 1 | execution_smoke_passed |
| [community/paper_implementation_project/block_encoding/laplacian_BE_Nx4_Ny4.qmod](algorithms/block_encoding/classiq/paper_implementation_project/block_encoding/laplacian_BE_Nx4_Ny4/README.md) | Not provided | 1 | execution_smoke_passed |
| [community/paper_implementation_project/block_encoding/toeplitz_BE_N16.qmod](algorithms/block_encoding/classiq/paper_implementation_project/block_encoding/toeplitz_BE_N16/README.md) | Not provided | 1 | execution_smoke_passed |
| [community/paper_implementation_project/block_encoding/tridiagonal_BE_size16.qmod](algorithms/block_encoding/classiq/paper_implementation_project/block_encoding/tridiagonal_BE_size16/README.md) | Not provided | 1 | frontend_refused |

### cryptography

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [algorithms/number_theory_and_cryptography/discrete_log/discrete_log.qmod](algorithms/cryptography/classiq/discrete_log/discrete_log/README.md) | Not provided | 1 | execution_smoke_passed |
| [algorithms/number_theory_and_cryptography/elliptic_curves/elliptic_curve_discrete_log.qmod](algorithms/cryptography/classiq/elliptic_curves/elliptic_curve_discrete_log/README.md) | Not provided | 1 | execution_smoke_passed |
| [algorithms/number_theory_and_cryptography/hidden_shift/hidden_shift.qmod](algorithms/cryptography/classiq/hidden_shift/hidden_shift/README.md) | Available | 2 | execution_smoke_passed |
| [algorithms/number_theory_and_cryptography/shor/shor.qmod](algorithms/cryptography/classiq/shor/shor/README.md) | Not provided | 1 | execution_smoke_passed |
| [community/paper_implementation_project/bb84/bb84_implementation.qmod](algorithms/cryptography/classiq/paper_implementation_project/bb84/bb84_implementation/README.md) | Not provided | 2 | execution_smoke_passed |

### foundational

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [algorithms/foundational/bernstein_vazirani/bernstein_vazirani.qmod](algorithms/foundational/classiq/bernstein_vazirani/bernstein_vazirani/README.md) | Not provided | 1 | execution_smoke_passed |
| [algorithms/foundational/deutsch_jozsa/deutsch_jozsa.qmod](algorithms/foundational/classiq/deutsch_jozsa/deutsch_jozsa/README.md) | Not provided | 2 | execution_smoke_passed |
| [algorithms/foundational/quantum_teleportation/quantum_teleportation.qmod](algorithms/foundational/classiq/quantum_teleportation/quantum_teleportation/README.md) | Not provided | 0 | execution_smoke_passed |
| [algorithms/foundational/simon/simon.qmod](algorithms/foundational/classiq/simon/simon/README.md) | Not provided | 1 | execution_smoke_passed |

### grover

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [algorithms/search_and_optimization/grover/3_sat_grover.qmod](algorithms/grover/classiq/grover/3_sat_grover/README.md) | Not provided | 2 | execution_smoke_passed |
| [algorithms/search_and_optimization/grover/grover.qmod](algorithms/grover/classiq/grover/grover/README.md) | Not provided | 1 | execution_smoke_passed |
| [algorithms/search_and_optimization/grover/grover_max_cut.qmod](algorithms/grover/classiq/grover/grover_max_cut/README.md) | Not provided | 2 | execution_smoke_passed |
| [algorithms/search_and_optimization/grover_mixer_qaoa/gm_qaoa.qmod](algorithms/grover/classiq/grover_mixer_qaoa/gm_qaoa/README.md) | Not provided | 1 | execution_smoke_passed |

### hamiltonian simulation

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [algorithms/hamiltonian_simulation/hamiltonian_simulation_guide/hamiltonian_simulation_guide.qmod](algorithms/hamiltonian_simulation/classiq/hamiltonian_simulation_guide/hamiltonian_simulation_guide/README.md) | Not provided | 1 | execution_smoke_passed |
| [algorithms/hamiltonian_simulation/hamiltonian_simulation_guide/hamiltonian_simulation_guide_qdrift.qmod](algorithms/hamiltonian_simulation/classiq/hamiltonian_simulation_guide/hamiltonian_simulation_guide_qdrift/README.md) | Not provided | 1 | execution_smoke_passed |
| [algorithms/hamiltonian_simulation/hamiltonian_simulation_guide/hamiltonian_simulation_guide_trotter.qmod](algorithms/hamiltonian_simulation/classiq/hamiltonian_simulation_guide/hamiltonian_simulation_guide_trotter/README.md) | Not provided | 1 | execution_smoke_passed |

### machine learning

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [algorithms/QML/hybrid_qnn/hybrid_qnn_for_subset_majority.qmod](algorithms/machine_learning/classiq/hybrid_qnn/hybrid_qnn_for_subset_majority/README.md) | Not provided | 1 | execution_smoke_passed |
| [algorithms/QML/qgan/qgan_bars_and_strips.qmod](algorithms/machine_learning/classiq/qgan/qgan_bars_and_strips/README.md) | Not provided | 1 | execution_smoke_passed |
| [algorithms/QML/qsvm/qsvm.qmod](algorithms/machine_learning/classiq/qsvm/qsvm/README.md) | Not provided | 1 | execution_smoke_passed |
| [algorithms/QML/quantum_autoencoder/quantum_autoencoder.qmod](algorithms/machine_learning/classiq/quantum_autoencoder/quantum_autoencoder/README.md) | Available | 2 | execution_smoke_passed |

### metrology

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [algorithms/metrology/classical_shadow_tomography.qmod](algorithms/metrology/classiq/classical_shadow_tomography/README.md) | Available | 2 | execution_smoke_passed |
| [community/paper_implementation_project/classical_shadows/classical_shadows.qmod](algorithms/metrology/classiq/paper_implementation_project/classical_shadows/classical_shadows/README.md) | Available | 2 | execution_smoke_passed |

### phase estimation

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [algorithms/quantum_phase_estimation/qpe_for_matrix/qpe_for_matrix.qmod](algorithms/phase_estimation/classiq/qpe_for_matrix/qpe_for_matrix/README.md) | Not provided | 2 | execution_smoke_passed |
| [algorithms/quantum_phase_estimation/qpe_with_qubitization/qpe_for_molecule_with_qubitization.qmod](algorithms/phase_estimation/classiq/qpe_with_qubitization/qpe_for_molecule_with_qubitization/README.md) | Not provided | 1 | execution_smoke_passed |

### primitives

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [algorithms/quantum_primitives/gqsp/gqsp.qmod](algorithms/primitives/classiq/gqsp/gqsp/README.md) | Not provided | 1 | execution_smoke_passed |
| [algorithms/quantum_primitives/hadamard_test/hadamard_test.qmod](algorithms/primitives/classiq/hadamard_test/hadamard_test/README.md) | Not provided | 1 | execution_smoke_passed |
| [algorithms/quantum_primitives/swap_test/swap_test.qmod](algorithms/primitives/classiq/swap_test/swap_test/README.md) | Not provided | 2 | frontend_refused |

### qaoa

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [algorithms/search_and_optimization/QAOA/qaoa.qmod](algorithms/qaoa/classiq/QAOA/qaoa/README.md) | Not provided | 1 | execution_smoke_passed |
| [algorithms/search_and_optimization/QAOA/qaoa_knapsack.qmod](algorithms/qaoa/classiq/QAOA/qaoa_knapsack/README.md) | Not provided | 2 | execution_smoke_passed |
| [algorithms/search_and_optimization/QAOA/qaoa_max_cut.qmod](algorithms/qaoa/classiq/QAOA/qaoa_max_cut/README.md) | Not provided | 1 | execution_smoke_passed |

### quantum differential equations solvers

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [algorithms/quantum_differential_equations_solvers/discrete_poisson_solver/discrete_poisson_solver.qmod](algorithms/quantum_differential_equations_solvers/classiq/discrete_poisson_solver/discrete_poisson_solver/README.md) | Not provided | 1 | execution_smoke_passed |
| [algorithms/quantum_differential_equations_solvers/time_marching/time_marching.qmod](algorithms/quantum_differential_equations_solvers/classiq/time_marching/time_marching/README.md) | Not provided | 1 | execution_smoke_passed |

### quantum linear solvers

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [algorithms/quantum_linear_solvers/adiabatic_linear_solvers/solving_qlsp_with_aqc.qmod](algorithms/quantum_linear_solvers/classiq/adiabatic_linear_solvers/solving_qlsp_with_aqc/README.md) | Not provided | 1 | execution_smoke_passed |
| [algorithms/quantum_linear_solvers/hhl/hhl.qmod](algorithms/quantum_linear_solvers/classiq/hhl/hhl/README.md) | Not provided | 1 | execution_smoke_passed |
| [algorithms/quantum_linear_solvers/qsvt_matrix_inversion/qsvt_matrix_inversion.qmod](algorithms/quantum_linear_solvers/classiq/qsvt_matrix_inversion/qsvt_matrix_inversion/README.md) | Not provided | 1 | execution_smoke_passed |
| [algorithms/quantum_linear_solvers/vqls/vqls_with_lcu.qmod](algorithms/quantum_linear_solvers/classiq/vqls/vqls_with_lcu/README.md) | Not provided | 1 | execution_smoke_passed |

### search and optimization

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [algorithms/search_and_optimization/dqi/dqi_max_xorsat.qmod](algorithms/search_and_optimization/classiq/dqi/dqi_max_xorsat/README.md) | Not provided | 1 | execution_smoke_passed |

### state preparation

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [algorithms/quantum_state_preparation/adapt_vqe/adapt_vqe.qmod](algorithms/state_preparation/classiq/adapt_vqe/adapt_vqe/README.md) | Not provided | 1 | execution_smoke_passed |
| [algorithms/quantum_state_preparation/gibbs/quantum_thermal_state_preparation.qmod](algorithms/state_preparation/classiq/gibbs/quantum_thermal_state_preparation/README.md) | Not provided | 1 | execution_smoke_passed |
| [community/paper_implementation_project/quantum_compression_algorithm_for_symmetric_states/quantum_compression_algorithm_for_symmetric_states.qmod](algorithms/state_preparation/classiq/paper_implementation_project/quantum_compression_algorithm_for_symmetric_states/quantum_compression_algorithm_for_symmetric_states/README.md) | Not provided | 1 | execution_smoke_passed |
| [community/paper_implementation_project/quantum_state_preparation_without_coherent_arithmetic/stateprep_guassian_using_qsvt.qmod](algorithms/state_preparation/classiq/paper_implementation_project/quantum_state_preparation_without_coherent_arithmetic/stateprep_guassian_using_qsvt/README.md) | Not provided | 0 | frontend_refused |

## Applications (61)

### benchmarking

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [applications/benchmarking/quantum_volume/quantum_volume.qmod](applications/benchmarking/classiq/quantum_volume/quantum_volume/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/benchmarking/randomized_benchmarking/randomized_benchmarking.qmod](applications/benchmarking/classiq/randomized_benchmarking/randomized_benchmarking/README.md) | Not provided | 1 | execution_smoke_passed |

### chemistry

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [applications/chemistry/molecular_energy_curve/molecular_energy_curve.qmod](applications/chemistry/classiq/molecular_energy_curve/molecular_energy_curve/README.md) | Not provided | 2 | execution_smoke_passed |
| [applications/chemistry/molecule_eigensolver/molecule_eigensolver.qmod](applications/chemistry/classiq/molecule_eigensolver/molecule_eigensolver/README.md) | Not provided | 2 | execution_smoke_passed |
| [applications/chemistry/protein_folding/protein_folding_with_qaoa/protein_folding.qmod](applications/chemistry/classiq/protein_folding/protein_folding_with_qaoa/protein_folding/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/chemistry/protein_folding/protein_folding_with_quantum_walk/qfold.qmod](applications/chemistry/classiq/protein_folding/protein_folding_with_quantum_walk/qfold/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/chemistry/qpe_for_molecules/qpe_for_molecules.qmod](applications/chemistry/classiq/qpe_for_molecules/qpe_for_molecules/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/chemistry/quantum_drude_oscillator/quantum_drude_oscillator.qmod](applications/chemistry/classiq/quantum_drude_oscillator/quantum_drude_oscillator/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/chemistry/quantum_walk_fmo/quantum_walk_fmo.qmod](applications/chemistry/classiq/quantum_walk_fmo/quantum_walk_fmo/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/chemistry/second_quantized_hamiltonian/second_quantized_hamiltonian.qmod](applications/chemistry/classiq/second_quantized_hamiltonian/second_quantized_hamiltonian/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/chemistry/tensor_hypercontraction/tensor_hypercontraction.qmod](applications/chemistry/classiq/tensor_hypercontraction/tensor_hypercontraction/README.md) | Not provided | 1 | execution_smoke_passed |

### cybersecurity

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [applications/cybersecurity/link_monitoring/link_monitoring.qmod](applications/cybersecurity/classiq/link_monitoring/link_monitoring/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/cybersecurity/patching_management/patch_min_vertex_cover.qmod](applications/cybersecurity/classiq/patching_management/patch_min_vertex_cover/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/cybersecurity/whitebox_fuzzing/whitebox_fuzzing.qmod](applications/cybersecurity/classiq/whitebox_fuzzing/whitebox_fuzzing/README.md) | Not provided | 1 | execution_smoke_passed |

### differential equations

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [community/paper_implementation_project/bpde/bpde.qmod](applications/differential_equations/classiq/paper_implementation_project/bpde/bpde/README.md) | Available | 2 | execution_smoke_passed |
| [community/paper_implementation_project/quantum_algo_for_solving_linear_differential_equations/harmonic_oscillator.qmod](applications/differential_equations/classiq/paper_implementation_project/quantum_algo_for_solving_linear_differential_equations/harmonic_oscillator/README.md) | Not provided | 1 | execution_smoke_passed |

### finance

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [applications/finance/autocallable_options/partial_exponential_state_preparation.qmod](applications/finance/classiq/autocallable_options/partial_exponential_state_preparation/README.md) | Not provided | 1 | frontend_refused |
| [applications/finance/autocallable_options/quantum_autocallable_option_pricing.qmod](applications/finance/classiq/autocallable_options/quantum_autocallable_option_pricing/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/finance/brownian_chebyshev_polynomials/brownian_chebyshev_polynomials.qmod](applications/finance/classiq/brownian_chebyshev_polynomials/brownian_chebyshev_polynomials/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/finance/credit_card_fraud/credit_card_fraud.qmod](applications/finance/classiq/credit_card_fraud/credit_card_fraud/README.md) | Not provided | 2 | execution_smoke_passed |
| [applications/finance/hybrid_hhl_for_portfolio_optimization/portfolio_optimization_with_hhl.qmod](applications/finance/classiq/hybrid_hhl_for_portfolio_optimization/portfolio_optimization_with_hhl/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/finance/option_pricing/option_pricing.qmod](applications/finance/classiq/option_pricing/option_pricing/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/finance/portfolio_optimization/portfolio_optimization.qmod](applications/finance/classiq/portfolio_optimization/portfolio_optimization/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/finance/portfolio_optimization_hhl/HHL_portfolio.qmod](applications/finance/classiq/portfolio_optimization_hhl/HHL_portfolio/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/finance/rainbow_options/rainbow_options_bruteforce_method.qmod](applications/finance/classiq/rainbow_options/rainbow_options_bruteforce_method/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/finance/rainbow_options/rainbow_options_direct_method.qmod](applications/finance/classiq/rainbow_options/rainbow_options_direct_method/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/finance/rainbow_options/rainbow_options_integration_method.qmod](applications/finance/classiq/rainbow_options/rainbow_options_integration_method/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/finance/value_at_risk/value_at_risk.qmod](applications/finance/classiq/value_at_risk/value_at_risk/README.md) | Not provided | 1 | execution_smoke_passed |

### fluid dynamics

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [applications/CFD/double_slit_experiment/quantum_double_slit_experiment.qmod](applications/fluid_dynamics/classiq/double_slit_experiment/quantum_double_slit_experiment/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/CFD/heat_eq_qsvt/heat_eq_qsvt.qmod](applications/fluid_dynamics/classiq/heat_eq_qsvt/heat_eq_qsvt/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/CFD/qlbm/qlbm.qmod](applications/fluid_dynamics/classiq/qlbm/qlbm/README.md) | Not provided | 0 | frontend_refused |

### logistics

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [applications/logistics/facility_location/facility_location.qmod](applications/logistics/classiq/facility_location/facility_location/README.md) | Not provided | 2 | execution_smoke_passed |
| [applications/logistics/task_scheduling_problem/task_scheduling_problem.qmod](applications/logistics/classiq/task_scheduling_problem/task_scheduling_problem/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/logistics/traveling_salesman_problem/traveling_salesman_problem.qmod](applications/logistics/classiq/traveling_salesman_problem/traveling_salesman_problem/README.md) | Not provided | 2 | execution_smoke_passed |
| [applications/logistics/vehicle_routing_problem/vehicle_routing_problem.qmod](applications/logistics/classiq/vehicle_routing_problem/vehicle_routing_problem/README.md) | Not provided | 1 | execution_smoke_passed |

### optimization

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [applications/optimization/adapt_qaoa/adapt_qaoa.qmod](applications/optimization/classiq/adapt_qaoa/adapt_qaoa/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/optimization/electric_grid_optimization/electric_grid_optimization.qmod](applications/optimization/classiq/electric_grid_optimization/electric_grid_optimization/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/optimization/integer_linear_programming/integer_linear_programming.qmod](applications/optimization/classiq/integer_linear_programming/integer_linear_programming/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/optimization/kidney_exchange/kidney_exchange_problem.qmod](applications/optimization/classiq/kidney_exchange/kidney_exchange_problem/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/optimization/low_autocorrelation_binary_sequences_problem/evidence_scaling_labs.qmod](applications/optimization/classiq/low_autocorrelation_binary_sequences_problem/evidence_scaling_labs/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/optimization/max_clique/max_clique.qmod](applications/optimization/classiq/max_clique/max_clique/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/optimization/max_independent_set/max_independent_set.qmod](applications/optimization/classiq/max_independent_set/max_independent_set/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/optimization/max_induced_k_color_subgraph/max_induced_k_color_subgraph.qmod](applications/optimization/classiq/max_induced_k_color_subgraph/max_induced_k_color_subgraph/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/optimization/max_k_vertex_cover/max_k_vertex_cover.qmod](applications/optimization/classiq/max_k_vertex_cover/max_k_vertex_cover/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/optimization/min_graph_coloring/min_graph_coloring.qmod](applications/optimization/classiq/min_graph_coloring/min_graph_coloring/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/optimization/minimum_dominating_set/minimum_dominating_set.qmod](applications/optimization/classiq/minimum_dominating_set/minimum_dominating_set/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/optimization/qaoa_in_qaoa/qaoa_in_qaoa.qmod](applications/optimization/classiq/qaoa_in_qaoa/qaoa_in_qaoa/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/optimization/rectangles_packing/rectangles_packing_grid.qmod](applications/optimization/classiq/rectangles_packing/rectangles_packing_grid/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/optimization/robust_posture_optimization/robust_posture_optimization.qmod](applications/optimization/classiq/robust_posture_optimization/robust_posture_optimization/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/optimization/set_cover/set_cover.qmod](applications/optimization/classiq/set_cover/set_cover/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/optimization/set_partition/set_partition.qmod](applications/optimization/classiq/set_partition/set_partition/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/optimization/variational_quantum_imaginary_time_evolution/variational_quantum_imaginary_time_evolution.qmod](applications/optimization/classiq/variational_quantum_imaginary_time_evolution/variational_quantum_imaginary_time_evolution/README.md) | Not provided | 1 | execution_smoke_passed |

### physical systems

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [applications/physical_systems/fermi_hubbard_model_1D/fermi_hubbard_1D.qmod](applications/physical_systems/classiq/fermi_hubbard_model_1D/fermi_hubbard_1D/README.md) | Not provided | 2 | execution_smoke_passed |
| [applications/physical_systems/hhl_lanchester/hhl_lanchester.qmod](applications/physical_systems/classiq/hhl_lanchester/hhl_lanchester/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/physical_systems/ising_model/ising_model.qmod](applications/physical_systems/classiq/ising_model/ising_model/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/physical_systems/quantum_chaos/quantum_sawtooth_map.qmod](applications/physical_systems/classiq/quantum_chaos/quantum_sawtooth_map/README.md) | Not provided | 1 | execution_smoke_passed |

### plasma

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [applications/plasma/vlasov_ampere/vlasov_ampere.qmod](applications/plasma/classiq/vlasov_ampere/vlasov_ampere/README.md) | Not provided | 1 | execution_smoke_passed |

### telecom

| Original case | Native qfunc | Circuit validation links | Original status |
| --- | --- | --- | --- |
| [applications/telecom/network_traffic_optimization/network_traffic_optimization.qmod](applications/telecom/classiq/network_traffic_optimization/network_traffic_optimization/README.md) | Not provided | 2 | execution_smoke_passed |
| [applications/telecom/radio_access_network/radio_access_network_positioning_antennas.qmod](applications/telecom/classiq/radio_access_network/radio_access_network_positioning_antennas/README.md) | Not provided | 1 | execution_smoke_passed |
| [applications/telecom/resiliency_planning/resiliency_planning.qmod](applications/telecom/classiq/resiliency_planning/resiliency_planning/README.md) | Not provided | 2 | execution_smoke_passed |
| [applications/telecom/resiliency_planning/resiliency_planning_AMD.qmod](applications/telecom/classiq/resiliency_planning/resiliency_planning_AMD/README.md) | Not provided | 2 | execution_smoke_passed |
