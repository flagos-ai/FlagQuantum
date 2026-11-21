# Notebook editorial review ledger

This ledger records human editorial inspection and fresh cell execution,
separately from historical source migration acceptance. A row marked reviewed
does not certify every capability demonstrated by the paper or upstream platform.
Unlisted accepted Notebooks remain pending editorial review.

## Functions: 31 reviewed and replayed

Each Notebook now explains its quantum intent, the compiler responsibility,
how to interpret its result, a change to try and material support limits.
Bitwise AND exposes both qfunc definitions. The standard-gate and unitary cases
add coherent inputs with independent state checks. Optional reference circuits
remain clearly separated from program construction.

Full cell execution passed for all 31 in the dedicated local Python 3.12
environment. Raw per-case logs and exit codes are retained in the local delivery
evidence directory `notebook-ux-20261121/functions-replay.json`.

| Notebook | Review and replay |
| --- | --- |
| [functions/function_usage_examples/arithmetic/arithmetic_expression/arithmetic_expression_example](functions/arithmetic/arithmetic_expression/arithmetic_expression_example/walkthrough.ipynb) | Passed |
| [functions/function_usage_examples/arithmetic/bitwise_and/bitwise_and_example](functions/arithmetic/bitwise_and/overview/walkthrough.ipynb) | Passed |
| [functions/function_usage_examples/arithmetic/bitwise_invert/bitwise_invert_example](functions/arithmetic/bitwise_invert/bitwise_invert_example/walkthrough.ipynb) | Passed |
| [functions/function_usage_examples/arithmetic/bitwise_or/bitwise_or_example](functions/arithmetic/bitwise_or/overview/walkthrough.ipynb) | Passed |
| [functions/function_usage_examples/arithmetic/bitwise_xor/bitwise_xor_example](functions/arithmetic/bitwise_xor/overview/walkthrough.ipynb) | Passed |
| [functions/function_usage_examples/arithmetic/comparator/comparator_example](functions/arithmetic/comparator/overview/walkthrough.ipynb) | Passed |
| [functions/function_usage_examples/arithmetic/extremum/extremum_example](functions/arithmetic/extremum/overview/walkthrough.ipynb) | Passed |
| [functions/function_usage_examples/arithmetic/modular_exp/modular_exp_example](functions/arithmetic/modular_exp/modular_exp_example/walkthrough.ipynb) | Passed |
| [functions/function_usage_examples/arithmetic/modulo/modulo_example](functions/arithmetic/modulo/modulo_example/walkthrough.ipynb) | Passed |
| [functions/function_usage_examples/arithmetic/multiplication/multiplication](functions/arithmetic/multiplication/overview/walkthrough.ipynb) | Passed |
| [functions/function_usage_examples/arithmetic/negation/negation_example](functions/arithmetic/negation/overview/walkthrough.ipynb) | Passed |
| [functions/function_usage_examples/arithmetic/subtraction/subtraction_example](functions/arithmetic/subtraction/overview/walkthrough.ipynb) | Passed |
| [functions/function_usage_examples/mcx/mcx_example](functions/mcx/mcx_example/overview/walkthrough.ipynb) | Passed |
| [functions/qmod_library_reference/classiq_open_library/amplitude_amplification/exact_amplitude_amplification](functions/amplitude_amplification/exact_amplitude_amplification/walkthrough.ipynb) | Passed |
| [functions/qmod_library_reference/classiq_open_library/grover_operator/grover_operator](functions/grover_operator/overview/walkthrough.ipynb) | Passed |
| [functions/qmod_library_reference/classiq_open_library/hadamard_transform/hadamard_transform](functions/hadamard_transform/walkthrough.ipynb) | Passed |
| [functions/qmod_library_reference/classiq_open_library/linear_pauli_rotations/linear_pauli_rotations](functions/linear_pauli_rotations/overview/walkthrough.ipynb) | Passed |
| [functions/qmod_library_reference/classiq_open_library/qct_qst/qct_qst](functions/qct_qst/overview/walkthrough.ipynb) | Passed |
| [functions/qmod_library_reference/classiq_open_library/qft/qft](functions/qft/walkthrough.ipynb) | Passed |
| [functions/qmod_library_reference/classiq_open_library/qpe/qpe](functions/qpe/overview/walkthrough.ipynb) | Passed |
| [functions/qmod_library_reference/classiq_open_library/qsvt/qsvt](functions/qsvt/overview/walkthrough.ipynb) | Passed |
| [functions/qmod_library_reference/classiq_open_library/special_state_preparations/prepare_bell_state](functions/state_preparation/prepare_bell_state/walkthrough.ipynb) | Passed |
| [functions/qmod_library_reference/classiq_open_library/special_state_preparations/prepare_exponential_state](functions/state_preparation/prepare_exponential_state/overview/walkthrough.ipynb) | Passed |
| [functions/qmod_library_reference/classiq_open_library/special_state_preparations/prepare_ghz_state](functions/state_preparation/prepare_ghz_state/walkthrough.ipynb) | Passed |
| [functions/qmod_library_reference/classiq_open_library/special_state_preparations/prepare_partial_uniform_state](functions/state_preparation/prepare_partial_uniform_state/overview/walkthrough.ipynb) | Passed |
| [functions/qmod_library_reference/classiq_open_library/variational_data_encoding/variational_data_encoding](functions/data_encoding/variational_data_encoding/overview/walkthrough.ipynb) | Passed |
| [functions/qmod_library_reference/qmod_core_library/hamiltonian_evolution/qdrift/qdrift](functions/hamiltonian_evolution/qdrift/overview/walkthrough.ipynb) | Passed |
| [functions/qmod_library_reference/qmod_core_library/hamiltonian_evolution/suzuki_trotter/suzuki_trotter](functions/hamiltonian_evolution/suzuki_trotter/walkthrough.ipynb) | Passed |
| [functions/qmod_library_reference/qmod_core_library/prepare_state_and_amplitudes/prepare_state_and_amplitudes](functions/state_preparation/prepare_state_and_amplitudes/walkthrough.ipynb) | Passed |
| [functions/qmod_library_reference/qmod_core_library/standard_gates/standard_gates](functions/standard_gates/overview/walkthrough.ipynb) | Passed |
| [functions/qmod_library_reference/qmod_core_library/unitary/unitary](functions/unitary/walkthrough.ipynb) | Passed |

## Cross-category review

| Notebook | Review outcome | Replay |
| --- | --- | --- |
| [community/basic_examples/logical_qubits/logical_qubits_by_alice_and_bob/overview](community/basic_examples/logical_qubits/logical_qubits_by_alice_and_bob/overview/walkthrough.ipynb) | Explains gate approximation versus native SWAP/noise study; retains all 105 configurations | Passed |
| [community/basic_examples/superposition](community/basic_examples/superposition/walkthrough.ipynb) | Explains coherent uniform state, named outcomes and optional reference | Passed |
| [applications/image_processing/quantum_hadamard_edge_detection/overview](applications/image_processing/quantum_hadamard_edge_detection/overview/walkthrough.ipynb) | Exposes detector qfunc; explicitly identifies 16384-amplitude Circuit loader versus 256-entry qfunc limit | Passed |
| [tutorials/technology_demonstrations/approximated_state_preparation](tutorials/technology_demonstrations/approximated_state_preparation/walkthrough.ipynb) | Explains exact implementation under allowed approximation bounds | Passed |
| [tutorials/technology_demonstrations/auxiliary_managment/auxiliary_management](tutorials/technology_demonstrations/auxiliary_managment/auxiliary_management/walkthrough.ipynb) | Separates workspace constraints from claims of optimal scheduling | Passed |
| [tutorials/technology_demonstrations/oracle_generation/three_sat_oracles/overview](tutorials/technology_demonstrations/oracle_generation/three_sat_oracles/overview/walkthrough.ipynb) | Explains predicate phase and the independent truth-table Circuit | Passed |
| [tutorials/basics/qml/overview](tutorials/basics/qml/overview/walkthrough.ipynb) | Exposes ansatz and execution; explains static binding and explicit shift gradients | Passed |
| [tutorials/basics/quantumwalk_complex_network/overview](tutorials/basics/quantumwalk_complex_network/overview/walkthrough.ipynb) | Exposes round structure; distinguishes entangled coin drop from workspace cleanup | Passed |
| [tutorials/basics/quantum_programming/execution_tutorial/overview](tutorials/basics/quantum_programming/execution_tutorial/overview/walkthrough.ipynb) | Exposes arithmetic and exact named outputs; identifies finite-count adapter work | Passed |
| [tutorials/basics/mcx/overview](tutorials/basics/mcx/overview/walkthrough.ipynb) | Exposes all-ones control; explains zero-input and active/coherent witnesses | Passed |
| [tutorials/basics/quantum_programming/execution_tutorial_part2/overview](tutorials/basics/quantum_programming/execution_tutorial_part2/overview/walkthrough.ipynb) | Exposes Bell qfunc; distinguishes Circuit runtime binding from synth specialization | Passed |
| [tutorials/basics/exponentiation/example_exponentiation/overview](tutorials/basics/exponentiation/example_exponentiation/overview/walkthrough.ipynb) | Exposes full-Hamiltonian Trotter call and actual named output | Passed |

| [community/basics/vqe/overview](community/basics/vqe/overview/walkthrough.ipynb) | The editable qfunc prepares the state for a bound angle. | Passed |

| [community/basic_examples/hw_aware_synthesis/hw_aware_synthesis/overview](community/basic_examples/hw_aware_synthesis/hw_aware_synthesis/overview/walkthrough.ipynb) | The editable qfunc below specifies an all-ones control over a logical array. | Passed |
| [community/hackathons/iquhack_2025/challenge_template/overview](community/hackathons/iquhack_2025/challenge_template/overview/walkthrough.ipynb) | The editable starter allocates a logical array and requests uniform superposition. | Passed |
| [community/hackathons/iquhack_2025/gaussian_reference/overview](community/hackathons/iquhack_2025/gaussian_reference/overview/walkthrough.ipynb) | The editable direct qfunc supplies normalized Gaussian probabilities to native state preparation. | Passed |
| [community/hackathons/iquhack_2025/winning_gaussian/overview](community/hackathons/iquhack_2025/winning_gaussian/overview/walkthrough.ipynb) | The tuned product rotations, controlled adjustment and truncated Fourier schedule are built by the explicit [Circuit builder](workflow.py). | Passed |
| [community/hackathons/iquhack_2025/workshop/overview](community/hackathons/iquhack_2025/workshop/overview/walkthrough.ipynb) | The editable arithmetic qfunc below requests coherent output computation from two three-bit inputs. | Passed |
| [community/paper_implementation_project/bb84/bb84_implementation/overview](community/paper_implementation_project/bb84/bb84_implementation/overview/walkthrough.ipynb) | The editable root below prepares eight signals with classically bound message and basis arrays. | Passed |

| [community/hackathons/qinnovision_2025/advection_equation/second_place/overview](community/hackathons/qinnovision_2025/advection_equation/second_place/overview/walkthrough.ipynb) | The linked quantum builder initializes the seven-bit profile and repeats a controlled-rotation ladder for the declared time grid. | Passed |
| [community/hackathons/qinnovision_2025/advection_equation/winning/overview](community/hackathons/qinnovision_2025/advection_equation/winning/overview/walkthrough.ipynb) | The native qfunc loads the original seven-bit initial vector. | Passed |
| [community/paper_implementation_project/block_encoding/select_structures_be/overview](community/paper_implementation_project/block_encoding/select_structures_be/overview/walkthrough.ipynb) | The current builder uses native synth to load coefficient amplitudes and a bounded Circuit SELECT to implement each structured address map. | Passed |
| [community/paper_implementation_project/block_encoding_nd_laplacian/overview](community/paper_implementation_project/block_encoding_nd_laplacian/overview/walkthrough.ipynb) | The editable one-dimensional root composes the shared Fourier-shift Laplacian block. | Passed |
| [community/paper_implementation_project/bpde/overview](community/paper_implementation_project/bpde/overview/walkthrough.ipynb) | The quantum builder forms a two-branch evolution matrix from the regenerated hydrogen models, then adds a reusable symbolic phase and interferometric readout. | Passed |
| [community/paper_implementation_project/explicit_quantum_circuits_for_block_encoding/quantum_walks_via_efficient_blockencoding/overview](community/paper_implementation_project/explicit_quantum_circuits_for_block_encoding/quantum_walks_via_efficient_blockencoding/overview/walkthrough.ipynb) | The current implementation constructs tree, circulant and discriminant walk blocks with explicit Circuit gates and uses native synth for the Chebyshev input preparation. | Passed |
| [community/paper_implementation_project/quantum_algo_for_solving_linear_differential_equations/harmonic_oscillator/overview](community/paper_implementation_project/quantum_algo_for_solving_linear_differential_equations/harmonic_oscillator/overview/walkthrough.ipynb) | The editable factory binds the classical coefficient-encoding matrix and composes native preparation, selected generator powers and inverse encoding. | Passed |
| [community/paper_implementation_project/quantum_compression_algorithm_for_symmetric_states/overview](community/paper_implementation_project/quantum_compression_algorithm_for_symmetric_states/overview/walkthrough.ipynb) | The editable uniform root below prepares five logical qubits and calls the shared reversible compressor. | Passed |
| [community/paper_implementation_project/quantum_state_preparation_without_coherent_arithmetic/stateprep_guassian_using_qsvt/overview](community/paper_implementation_project/quantum_state_preparation_without_coherent_arithmetic/stateprep_guassian_using_qsvt/overview/walkthrough.ipynb) | The native preparation composes even/odd fitted responses and the shared QSVT blocks, then measures the actual eight-bit success amplitude used by exact amplification. | Passed |

## Algorithms: 45 reviewed and replayed

All original experiment sizes and existing numerical checks are retained.
The first seventeen rows were verified in the initial Algorithms checkpoint;
subsequent rows use the solver, ML, phase, dynamics and remaining-Algorithms
replay manifests. Every listed workflow executed all code cells. The original
20-bit Glued Trees experiment completed in 874.711 seconds without shrinking
its time grid, term set or samples.

| Notebook | Review outcome | Replay |
| --- | --- | --- |
| [algorithms/foundational/quantum_teleportation/overview](algorithms/foundational/quantum_teleportation/overview/walkthrough.ipynb) | Exposes send/teleport qfuncs and parameter binding; explains deferred measurement and retained environment | Passed |
| [algorithms/quantum_primitives/swap_test/overview](algorithms/quantum_primitives/swap_test/overview/walkthrough.ipynb) | Exposes loading and overlap composition; distinguishes overlap magnitude from phase and finite sampling | Passed |
| [algorithms/number_theory_and_cryptography/shor/overview](algorithms/number_theory_and_cryptography/shor/overview/walkthrough.ipynb) | Exposes period-finding qfunc and named fractional readout; preserves quantum modulus 21 and classical postprocessing | Passed |
| [algorithms/number_theory_and_cryptography/hidden_shift/overview](algorithms/number_theory_and_cryptography/hidden_shift/overview/walkthrough.ipynb) | Exposes four-bit qfunc while preserving both sixteen-bit routes and GF(2) recovery | Passed |
| [algorithms/search_and_optimization/grover/grover/overview](algorithms/search_and_optimization/grover/grover/overview/walkthrough.ipynb) | Exposes SAT marker and diffuser qfunc; preserves original adaptive searches and stop condition | Passed |
| [algorithms/foundational/bernstein_vazirani](algorithms/foundational/bernstein_vazirani/walkthrough.ipynb) | Explains classical oracle specialization, quantum phase interference and binary-secret promise | Passed |
| [algorithms/foundational/deutsch_jozsa](algorithms/foundational/deutsch_jozsa/walkthrough.ipynb) | Explains explicit algebraic reductions, constant phase and exact promised decision | Passed |
| [algorithms/foundational/simon/overview](algorithms/foundational/simon/overview/walkthrough.ipynb) | Explains borrowed arithmetic input, retained outputs and independent GF(2) evidence requirements | Passed |
| [algorithms/amplitude_amplification_and_estimation/oblivious_amplitude_amplification/overview](algorithms/amplitude_amplification_and_estimation/oblivious_amplitude_amplification/overview/walkthrough.ipynb) | Explains reversible block reflection, nonmonotone amplification and concrete one-round scope | Passed |
| [algorithms/amplitude_amplification_and_estimation/qmc_user_defined/overview](algorithms/amplitude_amplification_and_estimation/qmc_user_defined/overview/walkthrough.ipynb) | Explains integrand binding, signed QPE and validation-only exact expectation | Passed |
| [algorithms/amplitude_amplification_and_estimation/quantum_counting/overview](algorithms/amplitude_amplification_and_estimation/quantum_counting/overview/walkthrough.ipynb) | Distinguishes qfunc counting from classical interval updates and probabilistic precision | Passed |
| [algorithms/hamiltonian_simulation/hamiltonian_simulation_guide/overview](algorithms/hamiltonian_simulation/hamiltonian_simulation_guide/overview/walkthrough.ipynb) | Explains Pauli lowering, static time specialization and observable versus complete-state error | Passed |
| [algorithms/hamiltonian_simulation/hamiltonian_simulation_with_block_encoding/jacobi_anger_expansion/overview](algorithms/hamiltonian_simulation/hamiltonian_simulation_with_block_encoding/jacobi_anger_expansion/overview/walkthrough.ipynb) | Classical supporting derivation; explicitly not a quantum program; links actual QSVT execution | Passed |
| [algorithms/hamiltonian_simulation/hamiltonian_simulation_with_block_encoding/hamiltonian_simulation_gqsp/overview](algorithms/hamiltonian_simulation/hamiltonian_simulation_with_block_encoding/hamiltonian_simulation_gqsp/overview/walkthrough.ipynb) | Mixed synth/Circuit construction; explains walk composition, phase residuals and postselection | Passed |
| [algorithms/hamiltonian_simulation/hamiltonian_simulation_with_block_encoding/hamiltonian_simulation_qsvt/overview](algorithms/hamiltonian_simulation/hamiltonian_simulation_with_block_encoding/hamiltonian_simulation_qsvt/overview/walkthrough.ipynb) | Circuit and dense response construction; opaque operations explicitly not basis-gate costs | Passed |
| [algorithms/hamiltonian_simulation/hamiltonian_simulation_with_block_encoding/hamiltonian_simulation_qubitization/overview](algorithms/hamiltonian_simulation/hamiltonian_simulation_with_block_encoding/hamiltonian_simulation_qubitization/overview/walkthrough.ipynb) | Mixed synth/Circuit construction; identifies integrated qfunc gap and retained flags | Passed |
| [algorithms/amplitude_amplification/qsvt_fixed_point/overview](algorithms/amplitude_amplification/qsvt_fixed_point/overview/walkthrough.ipynb) | Explains bounded truth-table specialization, fresh phase fitting and checked qfunc cleanup | Passed |

| [algorithms/machine_learning/hybrid_qnn/hybrid_qnn_for_subset_majority/overview](algorithms/machine_learning/hybrid_qnn/hybrid_qnn_for_subset_majority/overview/walkthrough.ipynb) | The visible qfuncs expose feature encoding, variational rotations and their composition. | Passed |
| [algorithms/machine_learning/qgan/qgan_bars_and_strips/overview](algorithms/machine_learning/qgan/qgan_bars_and_strips/overview/walkthrough.ipynb) | The visible qfunc preview prepares the same four-qubit encoding and eight-weight ansatz used by the training layer. | Passed |
| [algorithms/machine_learning/qsvm/overview](algorithms/machine_learning/qsvm/overview/walkthrough.ipynb) | The editable Pauli qfunc below composes two feature layers and the adjoint of the second input's encoding. | Passed |
| [algorithms/machine_learning/quantum_autoencoder/overview](algorithms/machine_learning/quantum_autoencoder/overview/walkthrough.ipynb) | The editable encoder below defines data rotations, both entangling layers and the final trash rotations. | Passed |
| [algorithms/metrology/classical_shadow_tomography/overview](algorithms/metrology/classical_shadow_tomography/overview/walkthrough.ipynb) | The editable Pauli probe below prepares a Bell state and composes the selected basis rotations. | Passed |
| [algorithms/number_theory_and_cryptography/discrete_log/overview](algorithms/number_theory_and_cryptography/discrete_log/overview/walkthrough.ipynb) | The editable modulo-five qfunc below initializes the function register to one, prepares two coherent exponents, requests two modular exponentiations and applies inverse QFTs. | Passed |
| [algorithms/number_theory_and_cryptography/elliptic_curves/elliptic_curve_discrete_log/overview](algorithms/number_theory_and_cryptography/elliptic_curves/elliptic_curve_discrete_log/overview/walkthrough.ipynb) | The sampled native qfunc route uses bounded table-driven point translation. | Passed |
| [algorithms/quantum_differential_equations_solvers/discrete_poisson_solver/overview](algorithms/quantum_differential_equations_solvers/discrete_poisson_solver/overview/walkthrough.ipynb) | The editable `poisson` qfunc expresses state loading, spatial transforms, controlled spectral evolution, reciprocal rotation and automatic inverse regions. | Passed |
| [algorithms/quantum_differential_equations_solvers/lchs/overview](algorithms/quantum_differential_equations_solvers/lchs/overview/walkthrough.ipynb) | The 128-dimensional comparison is classical numerical analysis. | Passed |
| [algorithms/quantum_differential_equations_solvers/time_marching/overview](algorithms/quantum_differential_equations_solvers/time_marching/overview/walkthrough.ipynb) | The reusable `magnus` qfunc in [quantum.py](quantum.py) requests the static phase-stream slice. | Passed |
| [algorithms/quantum_linear_solvers/adiabatic_linear_solvers/solving_qlsp_with_aqc/overview](algorithms/quantum_linear_solvers/adiabatic_linear_solvers/solving_qlsp_with_aqc/overview/walkthrough.ipynb) | The visible factory below converts the model's Pauli description to observables, then defines a qfunc that requests probability preparation and two product formulas per time slice. | Passed |
| [algorithms/quantum_linear_solvers/hhl/overview](algorithms/quantum_linear_solvers/hhl/overview/walkthrough.ipynb) | The visible roots specify state loading, eigenphase precision, reciprocal rotations and the compute/use/uncompute region. | Passed |
| [algorithms/quantum_linear_solvers/qsvt_matrix_inversion/overview](algorithms/quantum_linear_solvers/qsvt_matrix_inversion/overview/walkthrough.ipynb) | The original-size Notebook constructs all sixteen Pauli terms in a native Circuit block, fits fresh phases, and applies a projected response using the shared Circuit QSVT machinery. | Passed |
| [algorithms/quantum_linear_solvers/vqls/vqls_with_lcu/overview](algorithms/quantum_linear_solvers/vqls/vqls_with_lcu/overview/walkthrough.ipynb) | The editable objective below composes the nine-angle ansatz, LCU preparation, selector-controlled phases, inverse preparation and the target Hadamards. | Passed |
| [algorithms/quantum_phase_estimation/qpe_for_matrix/overview](algorithms/quantum_phase_estimation/qpe_for_matrix/overview/walkthrough.ipynb) | The visible roots prepare the data and compose QPE around exact unitary or power-aware product-formula callbacks. | Passed |
| [algorithms/quantum_phase_estimation/qpe_with_qubitization/qpe_for_molecule_with_qubitization/overview](algorithms/quantum_phase_estimation/qpe_with_qubitization/qpe_for_molecule_with_qubitization/overview/walkthrough.ipynb) | The optimized root shown below prepares Hartree-Fock data, allocates selector and phase outputs, and composes the checked `estimate` helper. | Passed |
| [algorithms/quantum_primitives/gqsp/overview](algorithms/quantum_primitives/gqsp/overview/walkthrough.ipynb) | The polynomial coefficients and freshly fitted rotation rows are classical. | Passed |
| [algorithms/quantum_primitives/gradient_estimation/overview](algorithms/quantum_primitives/gradient_estimation/overview/walkthrough.ipynb) | The editable qfuncs request coherent polynomial phases and inverse Fourier transforms. | Passed |
| [algorithms/quantum_primitives/hadamard_test/overview](algorithms/quantum_primitives/hadamard_test/overview/walkthrough.ipynb) | The helper composition is entirely visible: within/apply supplies the two Hadamards around controlled QFT, and `drop(psi)` keeps the data as unobserved quantum environment. | Passed |
| [algorithms/quantum_primitives/quantum_oracle_sketching_boolean/overview](algorithms/quantum_primitives/quantum_oracle_sketching_boolean/overview/walkthrough.ipynb) | The editable three-bit preview below composes equality-controlled constant phases over a uniform register. | Passed |
| [algorithms/quantum_state_preparation/adapt_vqe/overview](algorithms/quantum_state_preparation/adapt_vqe/overview/walkthrough.ipynb) | The editable one-pool preview below requests a nonzero Pauli evolution rather than showing only allocation. | Passed |
| [algorithms/quantum_state_preparation/fermionic_gaussian/overview](algorithms/quantum_state_preparation/fermionic_gaussian/overview/walkthrough.ipynb) | The visible `prepare_pair` qfunc composes a rotation, relative phase, entangler and parity choice. | Passed |
| [algorithms/quantum_state_preparation/gibbs/quantum_thermal_state_preparation/overview](algorithms/quantum_state_preparation/gibbs/quantum_thermal_state_preparation/overview/walkthrough.ipynb) | The editable Fourier probe below requests coherent energy-dependent phases and inverse QFT. | Passed |

## Tutorials: 49 reviewed and replayed

The nine initial Tutorial rows remain in the cross-category table. The following
additional workflows retain all original code stages and numerical scope.

| Notebook | Review outcome | Replay |
| --- | --- | --- |

| [tutorials/advanced_tutorials/discrete_quantum_walk/overview](tutorials/advanced_tutorials/discrete_quantum_walk/overview/walkthrough.ipynb) | The editable circle root below expresses coin rotation, Fourier-domain signed motion and its inverse transform. | Passed |
| [tutorials/advanced_tutorials/discrete_quantum_walk_2d/discrete_time_quantum_walk/overview](tutorials/advanced_tutorials/discrete_quantum_walk_2d/discrete_time_quantum_walk/overview/walkthrough.ipynb) | The visible roots compose the coin and reversible movement, with six position bits on the ring or two five-bit coordinates on the torus. | Passed |
| [tutorials/advanced_tutorials/high_level_modeling_flexible_qpe/overview](tutorials/advanced_tutorials/high_level_modeling_flexible_qpe/overview/walkthrough.ipynb) | The editable root below passes a static power callback to QPE. | Passed |
| [tutorials/basics/entanglement](tutorials/basics/entanglement/walkthrough.ipynb) | `Output` declares newly allocated logical qubits; `allocate` initializes them in zero. | Passed |
| [tutorials/basics/grover_graph_coloring/overview](tutorials/basics/grover_graph_coloring/overview/walkthrough.ipynb) | The editable root below prepares all five two-bit color variables and composes the shared [predicate and diffuser](../main.py). | Passed |
| [tutorials/basics/optimization/learning_optimization/overview](tutorials/basics/optimization/learning_optimization/overview/walkthrough.ipynb) | The cost is a real phase expression over coherent two-bit integers. | Passed |
| [tutorials/basics/prepare_state](tutorials/basics/prepare_state/walkthrough.ipynb) | The 32 nonnegative entries are target probabilities, not amplitudes. | Passed |
| [tutorials/basics/qmci_pi_estimation/overview](tutorials/basics/qmci_pi_estimation/overview/walkthrough.ipynb) | The editable preparation root below allocates coordinates at a checked static bit width and composes the shared [prepare and reflect helpers](operations.py). | Passed |
| [tutorials/basics/quantum_primitives/hadamard_test_tutorial/overview](tutorials/basics/quantum_primitives/hadamard_test_tutorial/overview/walkthrough.ipynb) | The control Hadamards surround a coherent controlled four-qubit QFT. | Passed |
| [tutorials/basics/quantum_primitives/linear_combination_of_unitaries/overview](tutorials/basics/quantum_primitives/linear_combination_of_unitaries/overview/walkthrough.ipynb) | The two-bit controller coherently chooses identity, QFT or inverse QFT. | Passed |
| [tutorials/basics/quantum_programming/classiq_overview_tutorial](tutorials/basics/quantum_programming/classiq_overview_tutorial/walkthrough.ipynb) | The classical expression `x**2 + 1` describes the function; the three-bit Hadamard input requests all eight arguments coherently. | Passed |
| [tutorials/basics/quantum_programming/qmod_tutorial_part1/overview](tutorials/basics/quantum_programming/qmod_tutorial_part1/overview/walkthrough.ipynb) | These are FlagQuantum Python programs; the historical directory name records the source lesson, not a requirement to learn another language. | Passed |
| [tutorials/basics/quantum_programming/qmod_tutorial_part2/overview](tutorials/basics/quantum_programming/qmod_tutorial_part2/overview/walkthrough.ipynb) | The first pass runs each supplied solution with independent numerical checks. | Passed |
| [tutorials/basics/quantum_programming/synthesis_tutorial](tutorials/basics/quantum_programming/synthesis_tutorial/walkthrough.ipynb) | Both runs compute the same coherent `x**2 + 1` function. | Passed |
| [tutorials/technology_demonstrations/QPE/qpe_for_unitary_matrix/overview](tutorials/technology_demonstrations/QPE/qpe_for_unitary_matrix/overview/walkthrough.ipynb) | The visible qfunc roots allocate three data bits and one of eight fractional phase widths, then invoke coherent QPE on the supplied 8-by-8 unitary. | Passed |
| [tutorials/technology_demonstrations/arithmetic_expressions/overview](tutorials/technology_demonstrations/arithmetic_expressions/overview/walkthrough.ipynb) | The three input bits describe all eight `(x,y)` pairs. | Passed |
| [tutorials/technology_demonstrations/classiq_paper/qsvt/catalyst_qsvt/overview](tutorials/technology_demonstrations/classiq_paper/qsvt/catalyst_qsvt/overview/walkthrough.ipynb) | The visible native root retains the original ten logical bits and four phases, while [program.py](program.py) defines the source-specific block, projector phases and inverse sequence. | Passed |
| [tutorials/technology_demonstrations/classiq_paper/qsvt/classiq_qsvt/overview](tutorials/technology_demonstrations/classiq_paper/qsvt/classiq_qsvt/overview/walkthrough.ipynb) | The root allocates six data bits, three block bits and a QSVT auxiliary, then composes the shared [QSVT sequence](program.py). | Passed |
| [tutorials/technology_demonstrations/classiq_paper/qsvt/qiskit_qsvt/overview](tutorials/technology_demonstrations/classiq_paper/qsvt/qiskit_qsvt/overview/walkthrough.ipynb) | The commented source is expressed entirely in native FlagQuantum. | Passed |
| [tutorials/technology_demonstrations/classiq_paper/qsvt/tket_qsvt/overview](tutorials/technology_demonstrations/classiq_paper/qsvt/tket_qsvt/overview/walkthrough.ipynb) | The native helper reconstructs the literal commented boxes, including their register wiring and half-turn rotations. | Passed |
| [tutorials/technology_demonstrations/classiq_paper/quantum_walk/classiq_discrete_quantum_walk/overview](tutorials/technology_demonstrations/classiq_paper/quantum_walk/classiq_discrete_quantum_walk/overview/walkthrough.ipynb) | The qfunc requests a coherent conditional increment/decrement on six position bits. | Passed |
| [tutorials/technology_demonstrations/classiq_paper/quantum_walk/pennylane_catalyst_discrete_quantum_walk/overview](tutorials/technology_demonstrations/classiq_paper/quantum_walk/pennylane_catalyst_discrete_quantum_walk/overview/walkthrough.ipynb) | This reconstruction deliberately retains the source cascade, including X on bit zero inside every cascade step. | Passed |
| [tutorials/technology_demonstrations/classiq_paper/quantum_walk/qiskit_discrete_quantum_walk/overview](tutorials/technology_demonstrations/classiq_paper/quantum_walk/qiskit_discrete_quantum_walk/overview/walkthrough.ipynb) | This is a FlagQuantum qfunc reconstruction of the commented source circuit; no external SDK is executed. | Passed |
| [tutorials/technology_demonstrations/classiq_paper/quantum_walk/tket_discrete_quantum_walk/overview](tutorials/technology_demonstrations/classiq_paper/quantum_walk/tket_discrete_quantum_walk/overview/walkthrough.ipynb) | The historical source is commented code, now reconstructed with FlagQuantum allocation, coherent control and modular arithmetic. | Passed |
| [tutorials/technology_demonstrations/discrete_quantum_walk_circle/overview](tutorials/technology_demonstrations/discrete_quantum_walk_circle/overview/walkthrough.ipynb) | The visible root below requests one coin-controlled modular shift using shared [reversible helpers](quantum.py). | Passed |
| [tutorials/technology_demonstrations/hamiltonian_evolution/overview](tutorials/technology_demonstrations/hamiltonian_evolution/overview/walkthrough.ipynb) | The native builders in [benchmark.py](benchmark.py) bind a fresh 551-term, twelve-qubit water Hamiltonian into qfunc evolution requests. | Passed |
| [tutorials/workshops/algo_design_qce_tutorial/algo_design_qce_tutorial_part_i/overview](tutorials/workshops/algo_design_qce_tutorial/algo_design_qce_tutorial_part_i/overview/walkthrough.ipynb) | The same familiar arithmetic builds different reversible programs for unsigned, signed and fractional registers. | Passed |
| [tutorials/workshops/algo_design_qce_tutorial/algo_design_qce_tutorial_part_ii/overview](tutorials/workshops/algo_design_qce_tutorial/algo_design_qce_tutorial_part_ii/overview/walkthrough.ipynb) | The visible qfuncs combine arithmetic phases, coherent feasibility conditions and static repeated layers. | Passed |
| [tutorials/workshops/combinatorial_workshop/combinatorial_qmod_workshop_for_maxcut/overview](tutorials/workshops/combinatorial_workshop/combinatorial_qmod_workshop_for_maxcut/overview/walkthrough.ipynb) | The editable qfunc below expresses the negative cut count and the shared RX mixer across all four layers. | Passed |
| [tutorials/workshops/finance_workshops/combi_workshop_equality_constriants_po/overview](tutorials/workshops/finance_workshops/combi_workshop_equality_constriants_po/overview/walkthrough.ipynb) | The editable quantum program below applies the portfolio cost as a phase on three two-bit holdings, then mixes each bit. | Passed |
| [tutorials/workshops/finance_workshops/combi_workshop_inequality_constriants_po/overview](tutorials/workshops/finance_workshops/combi_workshop_inequality_constriants_po/overview/walkthrough.ipynb) | The editable program uses three unsigned two-bit holdings and a three-bit numeric slack. | Passed |
| [tutorials/workshops/finance_workshops/option_pricing_workshop/overview](tutorials/workshops/finance_workshops/option_pricing_workshop/overview/walkthrough.ipynb) | The editable root below composes native probability loading, payoff rotations and bounded amplification. | Passed |
| [tutorials/workshops/finance_workshops/rainbow_options_workshop_bruteforce/overview](tutorials/workshops/finance_workshops/rainbow_options_workshop_bruteforce/overview/walkthrough.ipynb) | The editable preparation root below composes the fresh two-asset model and its controlled payoff rotations. | Passed |
| [tutorials/workshops/grover_workshop/grover_workshop/overview](tutorials/workshops/grover_workshop/grover_workshop/overview/walkthrough.ipynb) | The two drafts intentionally allocate seven qubits without an oracle or reflection. | Passed |
| [tutorials/workshops/hhl_workshop/overview](tutorials/workshops/hhl_workshop/overview/walkthrough.ipynb) | The qfunc loads the RHS, performs finite-resolution QPE, rotates the indicator using a classical reciprocal table and applies the inverse QPE. | Passed |
| [tutorials/workshops/oracle_workshop/oracles_workshop/overview](tutorials/workshops/oracle_workshop/oracles_workshop/overview/walkthrough.ipynb) | The bit oracle XORs a predicate into a target, preserving reversibility even for a nonzero target. | Passed |

| [algorithms/quantum_walks/glued_trees/overview](algorithms/quantum_walks/glued_trees/overview/walkthrough.ipynb) | The twenty-qubit qfunc below requests the complete ordered 200-term evolution with ten repetitions. | Passed |
| [algorithms/search_and_optimization/dqi/dqi_max_xorsat/overview](algorithms/search_and_optimization/dqi/dqi_max_xorsat/overview/walkthrough.ipynb) | The editable DQI root below requests symmetric amplitudes, sign phases, GF(2) encoding, the lookup decoder and final Hadamards. | Passed |
| [algorithms/search_and_optimization/grover_mixer_qaoa/gm_qaoa/overview](algorithms/search_and_optimization/grover_mixer_qaoa/gm_qaoa/overview/walkthrough.ipynb) | The visible roots express the ordinary penalized phase/mixer and the one-hot Grover composition. | Passed |
| [algorithms/search_and_optimization/qaoa/overview](algorithms/search_and_optimization/qaoa/overview/walkthrough.ipynb) | The visible programs prepare coherent decision variables and compose cost and mixer helpers. | Passed |
| [algorithms/search_and_optimization/quantum_likelihood_estimation/overview](algorithms/search_and_optimization/quantum_likelihood_estimation/overview/walkthrough.ipynb) | The editable factory below builds a one-qubit experiment from a selected Hamiltonian, preparation, evolution time and analysis rotations. | Passed |

| [tutorials/technology_demonstrations/QPE/qpe_for_grover_operator/overview](tutorials/technology_demonstrations/QPE/qpe_for_grover_operator/overview/walkthrough.ipynb) | The visible predicate encodes the six marked four-bit integers coherently. | Passed |
| [tutorials/technology_demonstrations/hardware_aware_mcx/overview](tutorials/technology_demonstrations/hardware_aware_mcx/overview/walkthrough.ipynb) | The user specifies one coherent all-ones control over fifteen logical inputs. | Passed |
| [tutorials/technology_demonstrations/qaoa/qaoa_demonstration/overview](tutorials/technology_demonstrations/qaoa/qaoa_demonstration/overview/walkthrough.ipynb) | The editable root below uses the regenerated seven-vertex diagonal Hamiltonian and all ten QAOA layers. | Passed |

| [tutorials/technology_demonstrations/hhl/hhl_example/overview](tutorials/technology_demonstrations/hhl/hhl_example/overview/walkthrough.ipynb) | The visible qfunc composes RHS preparation, finite QPE, a reciprocal rotation table and inverse QPE. | Passed |

## Separate FlagQuantum learning path

The [six lessons](tutorials/learning_path/README.md) are new teaching material,
not additional upstream acceptance cases. All cells and independent full-state
checks passed; topology checks apply the final routed layout in the test only.
Named user readout does not require a physical-layout decoder.

## Confirmed follow-up needs

1. **Large amplitude loading:** the image workflow needs 16384 real amplitudes;
   the current qfunc profile allows at most 256. Preserve the full image and
   establish scalable construction, bounds, refusal tests and independent error
   checks before replacing its existing Circuit loader.
2. **Parameter reuse:** Circuit symbolic runtime binding and qfunc classical
   specialization are distinct supported paths. A shared parameterized qfunc
   path requires its own compiler/API contract and conformance evidence.
3. **Named finite-shot readout:** the execution tutorial uses a local low-level
   adapter for counts. Inspect existing runtime outputs before proposing a
   reusable named synth sampling interface.
4. **Block algorithm composition:** GQSP/qubitization manually compose Circuit
   primitives; the QSVT evolution example classically assembles a dense response
   matrix. These are valid executed FlagQuantum workflows, but not integrated
   qfunc block-algorithm synthesis. Before adding an API, assess existing
   algorithms/primitives and specify callback, control, inverse, selection and
   budget semantics. Do not compare opaque matrix counts with basis-gate counts.
5. **Teaching navigation:** generated workflow links now name the original case
   rather than displaying dozens of indistinguishable `overview` entries.

6. **Structured state families:** the community Gaussian tree example currently
   builds an explicit Circuit, numerically verifies eight bits and only compiles
   the 127-bit extension. A future qfunc family must preserve the tuned angle
   and truncation semantics, and distinguish construction from simulation.
7. **Signed arithmetic teaching:** the iQuHack fixed-point exercise uses a
   Circuit sign-extension adder while its unsigned example uses native qfunc.
   Reuse existing signed/fractional arithmetic support before defining another
   interface; require the original all-branch phase and carry checks.

These are scoped observations, not authorization to silently change protected
public APIs or replace original-scale acceptance with smaller demonstrations.

## Integration checkpoint: November 21, 2026

Reviewed and freshly replayed: **60/212** existing Notebooks (28.3%). The separate
six-lesson learning path also passed its initial full-cell numerical acceptance.
The remaining 152 workflows are pending this editorial review, regardless of
historical migration acceptance.

Validation on the local Python 3.12 CPU path:

- All 31 Functions Notebooks and the 12 listed cross-category Notebooks executed
  every cell, preserving their original workload and numerical checks.
- Eight learning/coherent-example tests independently checked states, cleanup,
  interference, topology and objective evaluation.
- A combined run of 122 catalog, focused library, bitwise, QML, walk, image,
  MCX, execution and molecular evolution regression tests passed. The exact commands and raw logs are kept
  in the local delivery evidence directory.
- Ruff, Black, scoped strict mypy and nbformat validation passed for the changed
  Python sources and 49 edited/new Notebooks respectively.
- The raw default suite recorded 21134 passes, 801 skips, nine failures and six
  setup errors in 1022.42 seconds. Its log is preserved, not relabeled as green.
  Two Git vocabulary nodes passed with an explicit worktree, 103 launch/MPS
  tests passed with local socket permissions, and the evidence node passed
  against an isolated ref set containing the fetched public main revision
  `0f7b86d922ffd665ebfd788cdffe38a6a7a20465`. The shared local repository has
  private refs that are absent from a public clone. These are environmental
  rechecks, not a second full-suite passing run.

No remote CI run or release certification is claimed by this checkpoint.

## Algorithms validation checkpoint

Seventeen additional Notebooks passed complete cell replay. The four affected
algorithm regression suites plus the independent SWAP reference passed all
131 tests in 35.82 seconds. Notebook format and generated catalog/audit checks
are rerun before commit. A broader 632-node selection was intentionally stopped
after 641.79 seconds, with 211 passes and no recorded failure, after entering
unrelated large-state references. It is partial evidence, not a full corpus
pass. The historical corpus and migration evidence remain unchanged.

## Current editorial checkpoint

**144/212 (67.9%)** accepted workflows have now been inspected and
freshly replayed; **68** remain. Counts by category: Functions 31,
Algorithms 45, Tutorials 49, Applications 1,
Community 18. The six new lessons are separate from this denominator.

Per-Notebook complete-cell exit status and elapsed time are retained in
`notebook-ux-20261121/*-replay.json`. Only successful replays count here.
The earlier raw default-suite and partial-catalog records above remain unchanged.
A missing scientific dependency on the first QSVM replay was resolved by using
the existing verified scientific runtime, and the retry passed. The first
editable programming-part-II replay exposed a notebook variable shadowing the
`phase` operation; renaming the result variable restored the intended operation,
and its complete-cell retry passed. Original failing logs are retained.

The solver/ML follow-up regression recorded **100 passes in 95.47 seconds**.
No remote CI or release claim is made. All original workloads, random choices,
independent references and approved migration denominators are preserved.
