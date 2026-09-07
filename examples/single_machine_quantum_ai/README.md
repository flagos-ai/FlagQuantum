# Single-Machine Quantum AI Examples

These examples are designed for laptop CPU, workstation CPU, or one GPU. They
do not initialize distributed backends and do not make scalability claims.

Run the local fast-path check first:

```bash
python examples/single_machine_quantum_ai/00_local_fast_path_check.py
```

Then try the end-to-end training examples:

```bash
python examples/single_machine_quantum_ai/01_vqe_statevector.py --steps 100
python examples/single_machine_quantum_ai/02_quantum_classifier.py --steps 100
python examples/single_machine_quantum_ai/03_mps_training.py --steps 100 --n-qubits 8
python examples/single_machine_quantum_ai/03_mps_training.py --steps 100 --n-qubits 60
python examples/single_machine_quantum_ai/04_jax_kernel_torch_layer.py --steps 300
python examples/single_machine_quantum_ai/05_mps_1000q_dimer_training.py --steps 1000 --n-qubits 1000
```

Examples 01 through 03 use the PyTorch-native path so the minimal installation
works without optional accelerators. Examples 04 and 05 contain the dedicated
JAX workflows.

MPS correctness and scale diagnostics:

```bash
# Small systems report an exact dense reference automatically.
python examples/single_machine_quantum_ai/03_mps_training.py --steps 100 --n-qubits 8 --reference small_exact

```

Short smoke runs:

```bash
python examples/single_machine_quantum_ai/01_vqe_statevector.py --steps 2
python examples/single_machine_quantum_ai/02_quantum_classifier.py --steps 2
python examples/single_machine_quantum_ai/03_mps_training.py --steps 2 --n-qubits 4
python examples/single_machine_quantum_ai/04_jax_kernel_torch_layer.py --steps 2
python examples/single_machine_quantum_ai/05_mps_1000q_dimer_training.py --steps 2 --n-qubits 20
```

What they show:

- `01_vqe_statevector.py`: VQE on the PyTorch-native statevector path.
- `02_quantum_classifier.py`: a tiny PyTorch-native quantum classifier.
- `03_mps_training.py`: PyTorch-native MPS training for larger local circuits.
- `04_jax_kernel_torch_layer.py`: PyTorch training interface with a FlagQuantum JAX quantum kernel.
- `05_mps_1000q_dimer_training.py`: a structured 1000-qubit dimerized MPS
  teacher-student task. It is intentionally favorable to FlagQuantum's
  PyTorch-facing JAX/MPS path and intentionally unfavorable to dense
  statevector simulators that need O(2^n) memory.

Each script prints a correctness reference and a backend speed comparison:

- VQE/MPS/JAX-kernel examples report an exact dense ground-state energy for
  the small Hamiltonian, plus the final energy gap.
- The classifier uses a known teacher quantum circuit as the theoretical
  solution and reports final loss/accuracy against that teacher-generated data.
- Speed sections report the selected backend. The dedicated JAX examples report
  their compilation and steady-state timing explicitly.
- The 1000q dimer example is a structure-aware MPS benchmark, not a claim about
  arbitrary 1000-qubit circuits. It reports `single_device_fast_path` and does
  not make distributed scalability claims.
