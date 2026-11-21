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
4. **Teaching navigation:** generated workflow links now name the original case
   rather than displaying dozens of indistinguishable `overview` entries.

These are scoped observations, not authorization to silently change protected
public APIs or replace original-scale acceptance with smaller demonstrations.

## Integration checkpoint: November 21, 2026

Reviewed and freshly replayed: **43/212** existing Notebooks (20.3%). The separate
six-lesson learning path also passed its initial full-cell numerical acceptance.
The remaining 169 workflows are pending this editorial review, regardless of
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
