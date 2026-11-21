# Notebook editorial review ledger

This ledger records case-by-case editorial inspection and fresh cell execution,
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
| [community/hackathons/iquhack_2025/winning_gaussian/overview](community/hackathons/iquhack_2025/winning_gaussian/overview/walkthrough.ipynb) | The tuned product rotations, controlled adjustment and truncated Fourier schedule are built by the explicit [Circuit builder](community/hackathons/iquhack_2025/winning_gaussian/overview/workflow.py). | Passed |
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
| [algorithms/quantum_differential_equations_solvers/time_marching/overview](algorithms/quantum_differential_equations_solvers/time_marching/overview/walkthrough.ipynb) | The reusable `magnus` qfunc in [quantum.py](algorithms/quantum_differential_equations_solvers/time_marching/overview/quantum.py) requests the static phase-stream slice. | Passed |
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
| [tutorials/basics/grover_graph_coloring/overview](tutorials/basics/grover_graph_coloring/overview/walkthrough.ipynb) | The editable root below prepares all five two-bit color variables and composes the shared [predicate and diffuser](tutorials/basics/grover_graph_coloring/main.py). | Passed |
| [tutorials/basics/optimization/learning_optimization/overview](tutorials/basics/optimization/learning_optimization/overview/walkthrough.ipynb) | The cost is a real phase expression over coherent two-bit integers. | Passed |
| [tutorials/basics/prepare_state](tutorials/basics/prepare_state/walkthrough.ipynb) | The 32 nonnegative entries are target probabilities, not amplitudes. | Passed |
| [tutorials/basics/qmci_pi_estimation/overview](tutorials/basics/qmci_pi_estimation/overview/walkthrough.ipynb) | The editable preparation root below allocates coordinates at a checked static bit width and composes the shared [prepare and reflect helpers](tutorials/basics/qmci_pi_estimation/overview/operations.py). | Passed |
| [tutorials/basics/quantum_primitives/hadamard_test_tutorial/overview](tutorials/basics/quantum_primitives/hadamard_test_tutorial/overview/walkthrough.ipynb) | The control Hadamards surround a coherent controlled four-qubit QFT. | Passed |
| [tutorials/basics/quantum_primitives/linear_combination_of_unitaries/overview](tutorials/basics/quantum_primitives/linear_combination_of_unitaries/overview/walkthrough.ipynb) | The two-bit controller coherently chooses identity, QFT or inverse QFT. | Passed |
| [tutorials/basics/quantum_programming/classiq_overview_tutorial](tutorials/basics/quantum_programming/classiq_overview_tutorial/walkthrough.ipynb) | The classical expression `x**2 + 1` describes the function; the three-bit Hadamard input requests all eight arguments coherently. | Passed |
| [tutorials/basics/quantum_programming/qmod_tutorial_part1/overview](tutorials/basics/quantum_programming/qmod_tutorial_part1/overview/walkthrough.ipynb) | These are FlagQuantum Python programs; the historical directory name records the source lesson, not a requirement to learn another language. | Passed |
| [tutorials/basics/quantum_programming/qmod_tutorial_part2/overview](tutorials/basics/quantum_programming/qmod_tutorial_part2/overview/walkthrough.ipynb) | The first pass runs each supplied solution with independent numerical checks. | Passed |
| [tutorials/basics/quantum_programming/synthesis_tutorial](tutorials/basics/quantum_programming/synthesis_tutorial/walkthrough.ipynb) | Both runs compute the same coherent `x**2 + 1` function. | Passed |
| [tutorials/technology_demonstrations/QPE/qpe_for_unitary_matrix/overview](tutorials/technology_demonstrations/QPE/qpe_for_unitary_matrix/overview/walkthrough.ipynb) | The visible qfunc roots allocate three data bits and one of eight fractional phase widths, then invoke coherent QPE on the supplied 8-by-8 unitary. | Passed |
| [tutorials/technology_demonstrations/arithmetic_expressions/overview](tutorials/technology_demonstrations/arithmetic_expressions/overview/walkthrough.ipynb) | The three input bits describe all eight `(x,y)` pairs. | Passed |
| [tutorials/technology_demonstrations/classiq_paper/qsvt/catalyst_qsvt/overview](tutorials/technology_demonstrations/classiq_paper/qsvt/catalyst_qsvt/overview/walkthrough.ipynb) | The visible native root retains the original ten logical bits and four phases, while [program.py](tutorials/technology_demonstrations/classiq_paper/qsvt/catalyst_qsvt/overview/program.py) defines the source-specific block, projector phases and inverse sequence. | Passed |
| [tutorials/technology_demonstrations/classiq_paper/qsvt/classiq_qsvt/overview](tutorials/technology_demonstrations/classiq_paper/qsvt/classiq_qsvt/overview/walkthrough.ipynb) | The root allocates six data bits, three block bits and a QSVT auxiliary, then composes the shared [QSVT sequence](tutorials/technology_demonstrations/classiq_paper/qsvt/classiq_qsvt/overview/program.py). | Passed |
| [tutorials/technology_demonstrations/classiq_paper/qsvt/qiskit_qsvt/overview](tutorials/technology_demonstrations/classiq_paper/qsvt/qiskit_qsvt/overview/walkthrough.ipynb) | The commented source is expressed entirely in native FlagQuantum. | Passed |
| [tutorials/technology_demonstrations/classiq_paper/qsvt/tket_qsvt/overview](tutorials/technology_demonstrations/classiq_paper/qsvt/tket_qsvt/overview/walkthrough.ipynb) | The native helper reconstructs the literal commented boxes, including their register wiring and half-turn rotations. | Passed |
| [tutorials/technology_demonstrations/classiq_paper/quantum_walk/classiq_discrete_quantum_walk/overview](tutorials/technology_demonstrations/classiq_paper/quantum_walk/classiq_discrete_quantum_walk/overview/walkthrough.ipynb) | The qfunc requests a coherent conditional increment/decrement on six position bits. | Passed |
| [tutorials/technology_demonstrations/classiq_paper/quantum_walk/pennylane_catalyst_discrete_quantum_walk/overview](tutorials/technology_demonstrations/classiq_paper/quantum_walk/pennylane_catalyst_discrete_quantum_walk/overview/walkthrough.ipynb) | This reconstruction deliberately retains the source cascade, including X on bit zero inside every cascade step. | Passed |
| [tutorials/technology_demonstrations/classiq_paper/quantum_walk/qiskit_discrete_quantum_walk/overview](tutorials/technology_demonstrations/classiq_paper/quantum_walk/qiskit_discrete_quantum_walk/overview/walkthrough.ipynb) | This is a FlagQuantum qfunc reconstruction of the commented source circuit; no external SDK is executed. | Passed |
| [tutorials/technology_demonstrations/classiq_paper/quantum_walk/tket_discrete_quantum_walk/overview](tutorials/technology_demonstrations/classiq_paper/quantum_walk/tket_discrete_quantum_walk/overview/walkthrough.ipynb) | The historical source is commented code, now reconstructed with FlagQuantum allocation, coherent control and modular arithmetic. | Passed |
| [tutorials/technology_demonstrations/discrete_quantum_walk_circle/overview](tutorials/technology_demonstrations/discrete_quantum_walk_circle/overview/walkthrough.ipynb) | The visible root below requests one coin-controlled modular shift using shared [reversible helpers](tutorials/technology_demonstrations/discrete_quantum_walk_circle/overview/quantum.py). | Passed |
| [tutorials/technology_demonstrations/hamiltonian_evolution/overview](tutorials/technology_demonstrations/hamiltonian_evolution/overview/walkthrough.ipynb) | The native builders in [benchmark.py](tutorials/technology_demonstrations/hamiltonian_evolution/overview/benchmark.py) bind a fresh 551-term, twelve-qubit water Hamiltonian into qfunc evolution requests. | Passed |
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

## Applications: 69 reviewed and replayed

The initial image workflow remains in the cross-category table. These additional
workflows retain the complete application stages and numerical scope.

| Notebook | Review outcome | Replay |
| --- | --- | --- |

| [applications/optimization/max_clique/overview](applications/optimization/max_clique/overview/walkthrough.ipynb) | The visible quantum program prepares seven decision bits and composes three cost/mixer layers. | Passed |
| [applications/optimization/max_independent_set/overview](applications/optimization/max_independent_set/overview/walkthrough.ipynb) | The visible qfunc combines the eight-bit independent-set cost with three mixing layers. | Passed |
| [applications/optimization/max_k_vertex_cover/overview](applications/optimization/max_k_vertex_cover/overview/walkthrough.ipynb) | The visible qfunc composes three layers for ten decision bits. | Passed |
| [applications/optimization/set_partition/overview](applications/optimization/set_partition/overview/walkthrough.ipynb) | The qfunc requests a phase for the squared difference between two subsets of the original ten integer weights, followed by three mixer layers. | Passed |

| [applications/benchmarking/quantum_volume/overview](applications/benchmarking/quantum_volume/overview/walkthrough.ipynb) | The visible builder allocates the chosen three-to-six-bit target and composes fresh static pair-unitary helpers. | Passed |
| [applications/benchmarking/randomized_benchmarking/overview](applications/benchmarking/randomized_benchmarking/overview/walkthrough.ipynb) | The displayed model builder captures a static Clifford helper and applies its checked adjoint before named readout. | Passed |
| [applications/cybersecurity/link_monitoring/overview](applications/cybersecurity/link_monitoring/overview/walkthrough.ipynb) | The visible qfunc prepares twelve resource decisions with node one fixed using X and applies H and mixers only to the remaining nodes. | Passed |
| [applications/cybersecurity/patching_management/patch_min_vertex_cover/overview](applications/cybersecurity/patching_management/patch_min_vertex_cover/overview/walkthrough.ipynb) | The visible preview composes three phase/mixer layers over eight resource bits. | Passed |
| [applications/cybersecurity/whitebox_fuzzing/overview](applications/cybersecurity/whitebox_fuzzing/overview/walkthrough.ipynb) | The visible qfunc allocates two six-bit coordinates, prepares uniform inputs and composes the shared mark/diffuse helpers for the requested bounded number of rounds. | Passed |
| [applications/finance/credit_card_fraud/overview](applications/finance/credit_card_fraud/overview/walkthrough.ipynb) | The visible native overlap qfunc allocates three feature bits, applies the Pauli embedding and composes a checked adjoint for the other row. | Passed |
| [applications/finance/option_pricing/overview](applications/finance/option_pricing/overview/walkthrough.ipynb) | The visible root prepares a five-bit asset distribution, applies the shared payoff rotation helper and composes a bounded Grover family. | Passed |
| [applications/finance/portfolio_optimization/overview](applications/finance/portfolio_optimization/overview/walkthrough.ipynb) | The visible qfunc allocates three unsigned three-bit holdings and three modeled slack bits, then composes three arithmetic cost/mixer layers. | Passed |
| [applications/logistics/facility_location/overview](applications/logistics/facility_location/overview/walkthrough.ipynb) | The visible preview prepares twelve assignment decisions and five cost/mixer layers, using the shared cost helper for supply and empty-facility penalties. | Passed |
| [applications/logistics/task_scheduling_problem/overview](applications/logistics/task_scheduling_problem/overview/walkthrough.ipynb) | The visible schedule qfunc allocates the model's bounded decision width, prepares a superposition and applies eight polynomial phase/mixer layers. | Passed |
| [applications/logistics/traveling_salesman_problem/overview](applications/logistics/traveling_salesman_problem/overview/walkthrough.ipynb) | The visible ansatz prepares sixteen assignment bits and composes all eight cost/mixer layers. | Passed |
| [applications/logistics/vehicle_routing_problem/overview](applications/logistics/vehicle_routing_problem/overview/walkthrough.ipynb) | The visible qfunc prepares sixteen decisions and twelve cost/mixer layers from freshly generated departure, arrival and intercity distances. | Passed |
| [applications/optimization/set_cover/overview](applications/optimization/set_cover/overview/walkthrough.ipynb) | The preview exposes three cost/mixer layers over eight selection bits and fifteen slack bits. | Passed |

| [applications/finance/autocallable_options/partial_exponential_state_preparation/overview](applications/finance/autocallable_options/partial_exponential_state_preparation/overview/walkthrough.ipynb) | The visible native composition prepares the five-bit exponential amplitudes, uses a private attenuation bit and applies bounded interval marking and inverse preparation. | Passed |
| [applications/finance/autocallable_options/quantum_autocallable_option_pricing/overview](applications/finance/autocallable_options/quantum_autocallable_option_pricing/overview/walkthrough.ipynb) | The visible native root loads three years of bounded return fields, encodes the payoff reference and composes the contract's reversible preparation. | Passed |
| [applications/finance/brownian_chebyshev_polynomials/overview](applications/finance/brownian_chebyshev_polynomials/overview/walkthrough.ipynb) | The visible native root composes bounded, explicitly quantized Brownian/ Chebyshev helpers and returns named time and price fields. | Passed |
| [applications/finance/hybrid_hhl_for_portfolio_optimization/portfolio_optimization_with_hhl/overview](applications/finance/hybrid_hhl_for_portfolio_optimization/portfolio_optimization_with_hhl/overview/walkthrough.ipynb) | The visible probe builder prepares the input and composes four-bit QPE for the rebuilt portfolio operator. | Passed |
| [applications/finance/portfolio_optimization_hhl/hhl_portfolio/overview](applications/finance/portfolio_optimization_hhl/hhl_portfolio/overview/walkthrough.ipynb) | The visible builder regenerates the matrix evolution, prepares its encoded forcing vector, composes six-bit QPE and applies all 64 reciprocal rotations inside within/apply. | Passed |
| [applications/finance/rainbow_options/rainbow_options_bruteforce_method/overview](applications/finance/rainbow_options/rainbow_options_bruteforce_method/overview/walkthrough.ipynb) | The visible native root loads two asset registers, composes a joint payoff rotation and the requested Grover iterations. | Passed |
| [applications/finance/rainbow_options/rainbow_options_direct_method/overview](applications/finance/rainbow_options/rainbow_options_direct_method/overview/walkthrough.ipynb) | The visible qfunc allocates two asset fields, a five-bit auxiliary selector and an indicator, then composes exponential loading and reflection helpers. | Passed |
| [applications/finance/rainbow_options/rainbow_options_integration_method/overview](applications/finance/rainbow_options/rainbow_options_integration_method/overview/walkthrough.ipynb) | The visible qfunc composes two-asset loading, a five-bit exponential reference and an indicator integration predicate. | Passed |
| [applications/finance/value_at_risk/overview](applications/finance/value_at_risk/overview/walkthrough.ipynb) | The visible native root loads 128 asset probabilities, XORs a strict threshold predicate into an indicator and composes the bounded reflection family. | Passed |
| [applications/optimization/adapt_qaoa/overview](applications/optimization/adapt_qaoa/overview/walkthrough.ipynb) | The visible adaptive qfunc prepares eight bits and composes cost layers with statically selected mixer kinds. | Passed |
| [applications/optimization/low_autocorrelation_binary_sequences_problem/evidence_scaling_labs/overview](applications/optimization/low_autocorrelation_binary_sequences_problem/evidence_scaling_labs/overview/walkthrough.ipynb) | This workflow constructs three exact 13-bit Circuit candidates for the same 162-term cost and three layers. | Passed |
| [applications/optimization/minimum_dominating_set/overview](applications/optimization/minimum_dominating_set/overview/walkthrough.ipynb) | The preview shows six decision bits, fourteen modeled slack bits and six cost/mixer layers. | Passed |
| [applications/optimization/qaoa_in_qaoa/overview](applications/optimization/qaoa_in_qaoa/overview/walkthrough.ipynb) | The visible community qfunc prepares up to six decision bits and two weighted cut/mixer layers. | Passed |
| [applications/optimization/rectangles_packing/rectangles_packing_grid/overview](applications/optimization/rectangles_packing/rectangles_packing_grid/overview/walkthrough.ipynb) | The visible program builder captures the regenerated 102-term placement Observable, prepares seventeen decisions and composes ten cost/mixer layers. | Passed |
| [applications/optimization/robust_posture_optimization/overview](applications/optimization/robust_posture_optimization/overview/walkthrough.ipynb) | The visible Y-readout qfunc allocates two bits, composes the source rotations and applies S then H exactly as specified. | Passed |
| [applications/optimization/variational_quantum_imaginary_time_evolution/overview](applications/optimization/variational_quantum_imaginary_time_evolution/overview/walkthrough.ipynb) | The visible qfunc composes six ordered Pauli rotations after a four-bit uniform preparation. | Passed |

| [applications/chemistry/lih_chemistry/overview](applications/chemistry/lih_chemistry/overview/walkthrough.ipynb) | The displayed ansatz factory prepares the tapered Hartree-Fock occupation and composes ordered Pauli factors for ten excitation operators. | Passed |
| [applications/chemistry/molecular_energy_curve/overview](applications/chemistry/molecular_energy_curve/overview/walkthrough.ipynb) | The visible qfunc prepares the one-bit tapered reference and applies its variational Pauli evolution. | Passed |
| [applications/chemistry/molecule_eigensolver/overview](applications/chemistry/molecule_eigensolver/overview/walkthrough.ipynb) | The visible four-bit qfunc composes explicit single-qubit rotations and a three-layer entangler on the declared reference occupation. | Passed |
| [applications/chemistry/projection_based_embedding/projected_based_embedding_tutorial/overview](applications/chemistry/projection_based_embedding/projected_based_embedding_tutorial/overview/walkthrough.ipynb) | The displayed Circuit kernel initializes the fourteen-bit tapered Hartree-Fock word and composes thirty ordered excitation rotations. | Passed |
| [applications/physical_systems/fermi_hubbard_model_1d/fermi_hubbard_1d/overview](applications/physical_systems/fermi_hubbard_model_1d/fermi_hubbard_1d/overview/walkthrough.ipynb) | The visible qfunc allocates eight spin-orbital bits, prepares occupied modes with signed adjacent Givens rotations and calls the retained Trotter evolution helper. | Passed |
| [applications/physical_systems/hhl_lanchester/overview](applications/physical_systems/hhl_lanchester/overview/walkthrough.ipynb) | The visible native builder prepares the padded forcing vector, composes six-bit phase estimation and reciprocal rotations, then uncomputes within/apply. | Passed |
| [applications/physical_systems/ising_model/overview](applications/physical_systems/ising_model/overview/walkthrough.ipynb) | The visible qfunc prepares six spin bits and five cost/mixer layers using the shared Ising cost helper. | Passed |

| [applications/cfd/double_slit_experiment/quantum_double_slit_experiment/overview](applications/cfd/double_slit_experiment/quantum_double_slit_experiment/overview/walkthrough.ipynb) | The displayed Circuit factory prepares the source field and composes all 192 projector angles with alternating block/adjoint calls. | Passed |
| [applications/cfd/heat_eq_qsvt/overview](applications/cfd/heat_eq_qsvt/overview/walkthrough.ipynb) | The visible native iterate qfunc loads the padded temperature field, composes the block or its adjoint and uses within/apply for projector phases. | Passed |
| [applications/cfd/qlbm/overview](applications/cfd/qlbm/overview/walkthrough.ipynb) | The visible qfunc allocates named position, direction and magnitude fields, prepares the stated amplitudes and composes thirteen static lattice steps from the generated active-velocity schedule. | Passed |
| [applications/cfd/qls_for_hybrid_solvers/chebyshev_approximation/overview](applications/cfd/qls_for_hybrid_solvers/chebyshev_approximation/overview/walkthrough.ipynb) | The visible loader qfunc prepares seven coefficient bits, loads a reduced amplitude table into the upper slice and applies H on explicitly uniform lower bits. | Passed |
| [applications/cfd/qls_for_hybrid_solvers/qls_chebyshev_lcu/overview](applications/cfd/qls_for_hybrid_solvers/qls_chebyshev_lcu/overview/walkthrough.ipynb) | The displayed Circuit walk composes the matrix encoding, coefficient loader and controlled odd powers. | Passed |
| [applications/cfd/qls_for_hybrid_solvers/qls_qsvt/overview](applications/cfd/qls_for_hybrid_solvers/qls_qsvt/overview/walkthrough.ipynb) | The displayed Circuit builder packs the two projector phase streams into an exact simulator matrix and composes it with input preparation and H gates. | Passed |
| [applications/cfd/qls_for_hybrid_solvers/verify_block_encoding/overview](applications/cfd/qls_for_hybrid_solvers/verify_block_encoding/overview/walkthrough.ipynb) | The visible banded factory constructs explicit FlagQuantum Circuit kernels; the projection wrapper chooses Pauli or banded encoding and executes it. | Passed |
| [applications/chemistry/protein_folding/protein_folding_with_qaoa/protein_folding/overview](applications/chemistry/protein_folding/protein_folding_with_qaoa/protein_folding/overview/walkthrough.ipynb) | The visible qfunc prepares seven assignment bits and five Pauli cost/mixer layers. | Passed |
| [applications/chemistry/protein_folding/protein_folding_with_quantum_walk/qfold/overview](applications/chemistry/protein_folding/protein_folding_with_quantum_walk/qfold/overview/walkthrough.ipynb) | The visible qfunc prepares two logical angle bits, composes the checked coin inside within/apply, shifts the selected move and applies the oracle. | Passed |
| [applications/chemistry/qpe_for_molecules/overview](applications/chemistry/qpe_for_molecules/overview/walkthrough.ipynb) | The visible builder composes the original grouped Strang evolution powers under phase controls, followed by inverse QFT. | Passed |
| [applications/chemistry/quantum_drude_oscillator/overview](applications/chemistry/quantum_drude_oscillator/overview/walkthrough.ipynb) | The visible builder selects the complete four-bit or six-bit native ansatz for two or three oscillators, with twelve or forty finite angle parameters. | Passed |
| [applications/chemistry/quantum_walk_fmo](applications/chemistry/quantum_walk_fmo/walkthrough.ipynb) | The visible qfunc allocates three site bits and applies ten first-order product-formula steps for the supplied time. | Passed |
| [applications/chemistry/second_quantized_hamiltonian/overview](applications/chemistry/second_quantized_hamiltonian/overview/walkthrough.ipynb) | The visible native qfunc allocates four bits and composes three rotation/ entangler layers. | Passed |
| [applications/chemistry/tensor_hypercontraction/overview](applications/chemistry/tensor_hypercontraction/overview/walkthrough.ipynb) | The displayed clock/walk factory composes explicit FlagQuantum PREPARE, SELECT, inverse and clock Circuit blocks. | Passed |

| [applications/automotive/cooling_systems_optimization/overview](applications/automotive/cooling_systems_optimization/overview/walkthrough.ipynb) | The displayed thermal factory composes a seven-bit Circuit from normalized pair blocks and a selector preparation. | Passed |
| [applications/physical_systems/quantum_chaos/quantum_sawtooth_map/overview](applications/physical_systems/quantum_chaos/quantum_sawtooth_map/overview/walkthrough.ipynb) | The visible qfunc starts at signed momentum minus two, composes QFT within/apply and applies both quadratic phases for each static kick. | Passed |
| [applications/plasma/vlasov_ampere/overview](applications/plasma/vlasov_ampere/overview/walkthrough.ipynb) | The displayed Circuit factory composes the original 17-bit encoding on an 18-bit signal register and applies all 252 phase rotations with 251 alternating block/adjoint applications. | Passed |
| [applications/plasma/vlasov_ampere/vlasov_ampere_qiskit/overview](applications/plasma/vlasov_ampere/vlasov_ampere_qiskit/overview/walkthrough.ipynb) | The displayed block factory constructs the 17-bit finite-difference encoding from velocity, force, current and boundary helpers. | Passed |

| [applications/optimization/electric_grid_optimization/overview](applications/optimization/electric_grid_optimization/overview/walkthrough.ipynb) | The qfunc below exposes twelve source/consumer decisions and six modeled slack bits across four layers. | Passed |
| [applications/optimization/integer_linear_programming/overview](applications/optimization/integer_linear_programming/overview/walkthrough.ipynb) | The visible qfunc allocates three unsigned two-bit integers and two modeled slack outputs, prepares a uniform superposition and composes three arithmetic phase/mixer layers. | Passed |
| [applications/optimization/kidney_exchange/kidney_exchange_problem/overview](applications/optimization/kidney_exchange/kidney_exchange_problem/overview/walkthrough.ipynb) | The qfunc below prepares nine donor/patient decision bits and all five cost/mixer layers. | Passed |
| [applications/telecom/network_traffic_optimization/overview](applications/telecom/network_traffic_optimization/overview/walkthrough.ipynb) | The visible qfunc prepares all twenty-four flow decisions and five cost/mixer layers. | Passed |
| [applications/telecom/radio_access_network/radio_access_network_positioning_antennas/overview](applications/telecom/radio_access_network/radio_access_network_positioning_antennas/overview/walkthrough.ipynb) | The visible qfunc prepares thirteen location bits and three binary slack bits, then composes all four source cost/mixer layers. | Passed |
| [applications/telecom/resiliency_planning/overview](applications/telecom/resiliency_planning/overview/walkthrough.ipynb) | The visible qfunc allocates twenty-four decisions and composes twenty cost/mixer layers with the source correlated-failure penalty helper. | Passed |
| [applications/telecom/resiliency_planning/resiliency_planning_amd/overview](applications/telecom/resiliency_planning/resiliency_planning_amd/overview/walkthrough.ipynb) | The visible qfunc retains the same twenty-four decisions and twenty source cost/mixer layers as the shared resiliency program. | Passed |

| [applications/physical_systems/maxwell_equation/maxwell_2d_simulation/overview](applications/physical_systems/maxwell_equation/maxwell_2d_simulation/overview/walkthrough.ipynb) | The displayed block factory composes conductor predicates and periodic finite-difference helpers on the original twelve data bits plus block storage. | Passed |

| [applications/optimization/max_induced_k_color_subgraph/overview](applications/optimization/max_induced_k_color_subgraph/overview/walkthrough.ipynb) | The visible qfunc prepares two six-bit color arrays and composes all eight cost/mixer layers. | Passed |

| [applications/optimization/min_graph_coloring/overview](applications/optimization/min_graph_coloring/overview/walkthrough.ipynb) | The qfunc below allocates fifteen color decisions, prepares their uniform superposition and composes six cost/mixer layers. | Passed |

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
   The original CFD nozzle Pauli block now has a native qfunc comparison using
   existing checked helpers and within/apply: all eight columns and two signed
   inputs match the independent complete state, including imaginary LCU phases
   and clean workspace. It needs 12 rather than eight qubits. Its banded sibling
   and general QSVT composition remain distinct work; this is not blanket
   block-algorithm support or an original-budget replacement claim.
5. **Teaching navigation:** generated workflow links now name the original case
   rather than displaying dozens of indistinguishable `overview` entries.

6. **Structured state families:** the community Gaussian tree example currently
   executes its eight-bit stage through native qfunc and only compiles the
   127-bit extension through Circuit. The native program preserves the fitted
   angles and thresholds, and exposes a controlled helper and pairwise phases.
   All 17 cells replayed; 18 related tests passed, including independent full
   states at widths two, five and eight and the unchanged large-stage schedule.
   The source-sized qfunc state error was below 6e-17 with no phase alignment.
   Larger native compile-only construction remains distinct from execution.
7. **Signed arithmetic teaching:** the iQuHack fixed-point exercise uses a
   Circuit sign-extension adder while its unsigned example uses native qfunc.
   Reuse existing signed/fractional arithmetic support before defining another
   interface; require the original all-branch phase and carry checks.
   A fresh original-domain preflight after `65e1243c8` needs 39 native qubits,
   including 25 work bits, so the 14-qubit Circuit remains the executed route.
   Borrowing immutable input casts reduces retained work bits from 36 to 25
   and instructions from 286 to 264 (recompute: 304 to 264). Exact reversible
   tests cover every input and arbitrary XOR target under both schedules;
   smaller independent full-state tests check phase and cleanup. This partial
   compiler improvement does not close the original-budget native gap.

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

## Applications verification notes

Application teaching notes expose the central native qfunc or explicit Circuit
kernel and explain classical modeling, compiler responsibilities, named readout,
interpretation, a concrete change to try and scope limits. Preview definitions
and the shared optimizer implementation are distinguished explicitly; editing
only a preview must not be described as changing a helper-owned full search.
Mixed Circuit/synth kernels and large compile-only stages keep those labels.

Canonical public imports were normalized without changing quantum operations.
The local binding check verified 368 moved references were the identical objects
already imported by their shared example modules. Custom example aliases such
as the benchmark's `Sequence` type remain in their owning module. Notebook
format and undefined-name checks apply after this import-only normalization.
Complete-cell numerical logs and the binding-check report remain separate.

The initial plain-Python Randomized Benchmarking replay rejected valid Notebook
top-level `await`. Its complete source-cached asynchronous replay passed using
`PyCF_ALLOW_TOP_LEVEL_AWAIT`; the original executor failure is retained. The two
coloring searches and Maxwell retain original sizes, angles, optimizer budgets,
phase rows and numerical assertions. A 1200-second local timeout is not a
semantic pass: one 7200-second complete retry is used where that initial budget
is insufficient. Both raw attempts are retained and only a successful full
replay enters the editorial count. Coloring timing wrappers only record the
canonical builder's calls and return its unchanged result.


## Applications validation checkpoint

All 68 revised Application Notebooks retain their original cell-stage layout.
An ordered-AST comparison confirms that merging the visible program previews
into existing cells preserves every executed statement and its order; original
provenance records and source hashes are unchanged. The static stage/provenance
selection passed **39 tests in 0.96 seconds**. Eight initial cell-count/location
failures are retained in the earlier run record; they were corrected in Notebook
organization without weakening the tests. Canonical public imports were checked
against **368 identical bindings** from the existing example helpers.

The current full default run recorded **21033 passes, 801 skips, 43 failures and
73 errors in 1390.86 seconds**. It inherited `GIT_WORK_TREE`, which also affected
Git repositories created inside test fixtures. A scoped rerun without that
variable passed **103 of 104** exact failed/error nodes in 63.83 seconds. The
remaining node is the existing evidence-history check: the shared checkout has
private historical refs that conflict with declarations for a public clone. No
revision declarations were rewritten. A separate localhost-enabled launch/MPS
rerun passed **103 tests in 0.83 seconds**. The eight relative helper links
introduced in this editorial ledger were rebased to their actual Notebook
directories, and the repository link check passed. These rechecks do not relabel
the original full default run as passing.

The first randomized-benchmarking replay used a synchronous executor that
rejected the Notebook's valid top-level `await`. Its complete asynchronous
replay passed with the same original programs, lengths, shots and seed. The two
coloring searches and the full Maxwell walk exceeded the initial 1200-second
process limit. Their one-time 7200-second replays passed without reducing the
workload: minimum coloring in 2692.162 seconds, maximum induced coloring in
1927.368 seconds and Maxwell in 1840.380 seconds. Every original iteration,
objective and all 1372 Maxwell walk selections were retained. Only successful
complete runs enter the editorial denominator.

Compiler and readout performance follow-ups are separate local changes, each
with its own conformance record. On commit `4346ad249`, all **216 original Qmod
paths and six separately approved alternatives** were freshly replayed in
484.349 seconds of total case time. All 222 matched their recorded stages:
210 original execution smokes passed, six originals retained their expected
frontend refusal, and the six approved alternatives passed. These execution
smokes check normalization/profile contracts; they do not replace independent
full-state numerical acceptance or turn a refused original into a pass. Original
source identities and historical resource baselines remain unchanged.

The final catalog/audit selection passed **23 tests in 4.67 seconds**. Notebook
format validation passed for all 68 changed Application files. The 69 Compiler
preparation tests and 200 Services execution/readout tests passed separately,
including absolute-phase, port-order, mutable-plan, density-matrix, exact-MPS
and numeric-reading checks. The protected API migration baseline passed.

## Current editorial checkpoint

**212/212 (100.0%)** accepted workflows have now been inspected and
freshly replayed; **0** remain. Counts by category: Functions 31,
Algorithms 45, Tutorials 49, Applications 69,
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


The Hamiltonian-QSVT follow-up retains all original 12 stages and visibly executes
both full phase streams through native Pauli blocks. All 26 focused tests and
12 cells pass. Full states for two source seeds and a signed input, all 16 signal
columns, absolute phases, named success probability, four clean work bits and
three budget refusals are covered. The 212/212 editorial denominator is unchanged;
the native route's ten qubits are recorded separately from the six-qubit packed
Circuit. Raw preflight, initial refusal and corrected acceptance logs remain in
the local `notebook-ux-20261121` evidence directory.


Finite-shot teaching now uses the authoritative Runtime API in the Bell and
Hamiltonian-QSVT examples. The 98 related checks cover exact versus empirical
results, seeded counts, joint-shot correlation, the accepted-shot denominator,
requested bit order and the sealed synthesis boundary. All changed cells replay;
no new sampling algorithm or public export was introduced.


The signed workshop's native compilation preflight now uses 25 total qubits,
11 work bits and 53 instructions including basis preparation. Its original
fourteen-qubit Circuit remains the executable original-budget reference.
The generic equal-modulus sum fusion preserves arbitrary XOR targets and shared
intermediates; read-only sign aliases remain separate from injective multiplier
operands. All 16 Notebook stages replay, with no provenance or original-source
changes. This capability follow-up does not change the 212 editorial denominator.

## Frozen integration and replay checkpoint

At implementation commit `e997d5cd8`, all 216 original Qmod paths and six
approved alternatives matched their recorded stages in 460.174 seconds of total
case time: 210 original execution smokes, six expected original frontend refusals
and six successful alternative smokes. These are normalization/profile checks,
not fresh independent full-state numerical acceptance.

A cross-check of the 212 approved workflows found twelve early cases with
integration evidence but without a uniform complete-cell process record. All
twelve now replay at their original size in 195.962 seconds total, including the
128-by-128 image and its 200000 shots, all 105 logical-cat grid points, and the
original walk and oracle stages. Source hashes are unchanged. Every approved
workflow now has an editorial link and a successful nonempty replay log. Records
for the other workflows retain their own historical implementation commits;
this is not a claim that all 212 executed on this Compiler commit.

The frozen default run completed with 21179 passes, 801 skips, nine failures and
six setup errors in 1023.78 seconds. Two vocabulary checks pass when Git receives
the explicit linked-worktree root. Six launch failures and six MPS setup errors
report restricted local socket/Gloo access, consistent with the separately
recorded successful environment rechecks. One evidence-history check still
rejects 99 origin declarations because shared private refs appear fetchable.
Historical refs and origin declarations were not rewritten. The raw full run
remains failed; focused and environment rechecks do not rename it green.

## Immutable root-input shift checkpoint

The terminal XOR reader now borrows immutable root-input bits for a static
power-of-two multiplication when that value is consumed only by the terminal
sum. One owned zero wire supplies all shift and padding positions. No operand
is written. A temporary arithmetic result, nonterminal consumer or unsupported
width retains owned storage; no temporary-lifetime pinning is introduced.
Both schedules preserve arbitrary XOR targets, absolute complex phases and
clean workspace. Existing multiplication kernels keep their injective maps.

The signed workshop now executes its native program, using 20 total qubits,
six work bits and 47 logical instructions including preparation (depth 23).
The arithmetic portion uses 43 instructions, compared with the preceding
25-qubit/11-work-bit/49-arithmetic-instruction implementation. Complete coherent
witnesses retain every original input pair and all 64 possible output words;
no global-phase alignment is used. Named numeric reading returns the signed
half-unit values 5, -7.5 and -2.5. The independent 14-qubit Circuit remains
available, and that smaller native budget still refuses. Qubit, workspace,
operation and execution-memory refusals remain distinct and checked.


The CFD banded source now has a second executable native route as well as its
independent Circuit reference. All eight columns, source seeds 37/91, full flag
branches, absolute phases and three clean work bits pass. Native resources are
nine qubits/454 logical instructions/depth 420 for either seed; the reference
retains six qubits. All 29 CFD tests and all ten original Notebook stages pass;
the editorial 212/212 denominator and original source identities are unchanged.


The subsequent rotation-table reuse uses eight qubits/two work bits/104 logical
instructions (depth 89). Tightening the total-qubit limit to seven selects one
work bit/156 instructions (depth 131). Both complete-state implementations
retain the original matrix and scale; the older 454-instruction record remains
historical. Neither change alters the approved migration denominator.
