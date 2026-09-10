# FlagQuantum Tutorials

Tutorials teach the concepts behind the runnable examples. They should be read
in order by new users, but each notebook should also stand alone.

## Learning path

| Order | Notebook | Learning goal |
| --- | --- | --- |
| 00 | `00_understanding_states.ipynb` | Understand state tensors, amplitudes, probabilities, and qubit order |
| 01 | `01_basic_operations.ipynb` | Apply single-qubit and two-qubit operations |
| 02 | `02_measurement.ipynb` | Measure states and interpret expectation values |
| 03 | `03_parameterized_gates.ipynb` | Use trainable gates and gradients |
| 04 | `04_quantum_circuit_builder.ipynb` | Build and inspect reusable circuits |
| 05 | `05_quantum_machine_learning.ipynb` | Train a small QML model end to end |
| 06 | `06_vqe_statevector.ipynb` | Train a small VQE model with local statevector simulation |
| 07 | `07_runtime_selection_statevector_mps_tn.ipynb` | Compare statevector, MPS, and tensor-network summaries |
| 08 | `08_pytorch_jax_qml_layer.ipynb` | Train an `fq.Module` backed by a JAX quantum kernel |
| 09 | `09_gradient_precision_speed_benchmark.ipynb` | Compare gradient precision and value+gradient speed across FlagQuantum runtimes |

## Writing a tutorial

Every tutorial should include:

- Title and learning objective.
- Prerequisites and expected runtime.
- A minimal runnable cell before advanced explanation.
- A result interpretation section.
- Runtime notes that distinguish statevector, MPS, tensor-network, and hybrid
  JAX/PyTorch paths.
- Links to the matching script under `examples/`.

## Continue learning

Use the [example catalog](../README.md) for script-based workflows, including
local training and deployment. New tutorials should extend that learning path
with noise, contraction methods, hardware comparison, and error correction;
they should not duplicate an existing introductory notebook.

Run notebooks in a fresh kernel and keep stored outputs empty. Tutorial 08
requires the optional JAX dependency. Support boundaries belong in the
[capability catalog](../../docs/generated/CAPABILITIES.md).
