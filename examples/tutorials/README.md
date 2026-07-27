# FlagQuantum Tutorials

Tutorials teach the concepts behind the runnable examples. They should be read
in order by new users, but each notebook should also stand alone.

## Current Beginner Series

| Order | Notebook | Learning goal | Cleanup status |
| --- | --- | --- | --- |
| 00 | `00_understanding_states.ipynb` | Understand state tensors, amplitudes, probabilities, and qubit order | Executable modern API contract |
| 01 | `01_basic_operations.ipynb` | Apply single-qubit and two-qubit operations | Executable modern API contract |
| 02 | `02_measurement.ipynb` | Measure states and interpret expectation values | Executable modern API contract |
| 03 | `03_parameterized_gates.ipynb` | Use trainable gates and gradients | Executable modern API contract |
| 04 | `04_quantum_circuit_builder.ipynb` | Build and inspect reusable circuits | Executable modern API contract |
| 05 | `05_quantum_machine_learning.ipynb` | Train a small QML model end to end | Executable modern API contract |
| 06 | `06_vqe_statevector.ipynb` | Train a small VQE model with local statevector simulation | Executable modern API contract |
| 07 | `07_runtime_selection_statevector_mps_tn.ipynb` | Compare statevector, MPS, and tensor-network summaries | Executable modern API contract |
| 08 | `08_pytorch_jax_qml_layer.ipynb` | Train an `fq.Module` backed by a JAX quantum kernel | Executable modern API contract |
| 09 | `09_gradient_precision_speed_benchmark.ipynb` | Compare gradient precision and value+gradient speed across FlagQuantum runtimes | New benchmark tutorial |

## Target Tutorial Shape

Every tutorial should include:

- Title and learning objective.
- Prerequisites and expected runtime.
- A minimal runnable cell before advanced explanation.
- A result interpretation section.
- Runtime notes that distinguish statevector, MPS, tensor-network, and hybrid
  JAX/PyTorch paths.
- Links to the matching script under `examples/`.

## Planned Additions

| Topic | Why it matters | Related example |
| --- | --- | --- |
| First circuit with `fq.Circuit` | Introduces the stable high-level API | `single_machine_quantum_ai/00_local_fast_path_check.py` |
| VQE statevector | Canonical variational algorithm | `06_vqe_statevector.ipynb`, `single_machine_quantum_ai/01_vqe_statevector.py` |
| PyTorch QML layer | Shows how FlagQuantum fits AI training loops | `08_pytorch_jax_qml_layer.ipynb`, `single_machine_quantum_ai/04_jax_kernel_torch_layer.py` |
| Choosing statevector, MPS, or TN | Teaches runtime selection instead of trial and error | `07_runtime_selection_statevector_mps_tn.ipynb`, `single_machine_quantum_ai/03_mps_training.py` |
| Tensor-network contraction basics | Explains contraction paths and memory tradeoffs | Planned |
| Noise and density matrix simulation | Covers noisy local simulation | Planned |
| Train then deploy | Connects training to real-device/cloud packaging | `../train_parameterized_circuit_then_deploy.py` |

## Current Health Notes

All ten notebooks are valid notebook v4 JSON, use the maintained FlagQuantum
API, keep stored outputs empty, and execute under the example integration
contract. Tutorial 08 additionally requires the optional JAX dependency.
