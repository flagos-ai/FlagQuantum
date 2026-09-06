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

By default, training examples use the PyTorch-native path so the minimal
installation works without optional accelerators. Select the FlagQuantum JAX
quantum kernel explicitly when JAX is installed:

```bash
python examples/single_machine_quantum_ai/03_mps_training.py --backend jax --steps 100 --compare-torch
```

MPS correctness and scale diagnostics:

```bash
# Small systems report an exact dense reference automatically.
python examples/single_machine_quantum_ai/03_mps_training.py --steps 100 --n-qubits 8 --reference small_exact

# Large systems skip dense exact diagonalization and report JAX compile/steady-state timing.
python examples/single_machine_quantum_ai/03_mps_training.py --scale-report 6,12,30,60 --layers 2 --max-bond 4 --scale-iters 3 --scale-warmup 1

# Reuse JAX persistent compilation cache across repeated runs with the same structure.
python examples/single_machine_quantum_ai/03_mps_training.py --scale-report 60,120,300,600 --layers 2 --max-bond 4 --scale-iters 3 --scale-warmup 1 --jax-cache-dir .fq_jax_cache

# ========================================================================
# MPS JAX Scale Report
# ========================================================================
#   layers       : 2
#   max_bond     : 4
#   device       : cpu
#   scale_warmup : 1
#   scale_iters  : 3
#   note         : first_loss_grad_s includes JAX compilation; steady_loss_grad_s is after warmup.
#   n_wires  params  max_bond  compile_setup_s  first_loss_grad_s  steady_loss_grad_s  loss      grad_norm  fastpath
#   -------  ------  --------  ---------------  -----------------  ------------------  --------  ---------  --------------------------------
#   6        24      4         0.0001787        1.28852            0.000639733         -4.17178  1.21151    local_pauli_zz_chain_padded_scan
#   12       48      4         0.000217         2.36052            0.000741233         -8.63694  3.89897    local_pauli_zz_chain_padded_scan
#   30       120     4         0.0002462        6.7991             0.001326            -20.7049  12.5415    local_pauli_zz_chain_padded_scan
#   60       240     4         0.0003308        12.399             0.00193577          -37.5472  27.986     local_pauli_zz_chain_padded_scan

# --jax-cache-dir .fq_jax_cache
# n_wires  params  max_bond  compile_setup_s  first_loss_grad_s  steady_loss_grad_s  loss      grad_norm  fastpath
# -------  ------  --------  ---------------  -----------------  ------------------  --------  ---------  --------------------------------
# 6        24      4         0.0002549        0.606605           0.000551133         -4.17178  1.21151    local_pauli_zz_chain_padded_scan
# 12       48      4         0.0002035        0.896491           0.0006904           -8.63694  3.89897    local_pauli_zz_chain_padded_scan
# 30       120     4         0.0002849        1.95799            0.00120327          -20.7049  12.5415    local_pauli_zz_chain_padded_scan
# 60       240     4         0.000325         3.9979             0.0025746           -37.5472  27.986     local_pauli_zz_chain_padded_scan
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

- `01_vqe_statevector.py`: VQE with a PyTorch-native default and optional JAX quantum kernel.
- `02_quantum_classifier.py`: a tiny quantum classifier with PyTorch-native and optional JAX execution.
- `03_mps_training.py`: MPS quantum model training for larger local circuits, with optional JAX acceleration.
- `04_jax_kernel_torch_layer.py`: PyTorch training interface with a FlagQuantum JAX quantum kernel.
- `05_mps_1000q_dimer_training.py`: a structured 1000-qubit dimerized MPS
  teacher-student task. It is intentionally favorable to FlagQuantum's
  PyTorch-facing JAX/MPS path and intentionally unfavorable to dense
  statevector simulators that need O(2^n) memory.

Each script prints a correctness reference and a backend speed comparison:

- VQE/MPS/JAX-kernel examples report an exact dense ground-state energy for
  the small Hamiltonian, plus the final energy gap.
- The MPS example also provides `--scale-report` to separate JAX setup time,
  first loss+gradient time, and warmed-up steady-state loss+gradient time for
  large local MPS workloads.
- The classifier uses a known teacher quantum circuit as the theoretical
  solution and reports final loss/accuracy against that teacher-generated data.
- Speed sections report the selected backend. On the JAX path, PyTorch-native
  comparison timing is enabled with `--compare-torch`.
- The 1000q dimer example is a structure-aware MPS benchmark, not a claim about
  arbitrary 1000-qubit circuits. It reports `single_device_fast_path` and does
  not make distributed scalability claims.
