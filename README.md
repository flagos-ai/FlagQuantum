<div align="center">
  <img src="assets/logo.png" alt="FlagQuantum Logo" width="380">
</div>

# FlagQuantum

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org/)

A high-performance distributed quantum statevector simulator built on PyTorch, enabling quantum circuit simulation across multiple GPUs with automatic sharding and resharding.

## ✨ Features

- **Distributed Statevector Simulation**: Leverage multiple GPUs to simulate large quantum circuits using `DTensor` from `torch.distributed`
- **Automatic Resharding**: Intelligently redistributes statevectors to minimize communication overhead during gate operations
- **Comprehensive Gate Set**: Includes Pauli, Clifford, rotation, and controlled gates with parameterized support
- **Invertible Backpropagation**: Memory-efficient gradient computation for trainable quantum circuits
- **Custom Gate Registration**: Extend the library with your own gates without modifying the core
- **Post-Selection & Noise Models**: Built-in support for measurement post-selection and depolarizing noise
- **Flexible Encoding**: Multiple encoding schemes (angle, amplitude, basis) for classical data embedding
- **OpenQASM 3.0 Export**: Run circuits on real quantum hardware (IBM, AWS Braket, Azure Quantum, IonQ, Rigetti)

## 🏗️ Architecture

```
flagquantum/
├── devices/          # Quantum device implementations
├── drawer/           # Quantum circuit visualization
├── ops/              # Quantum operations (gates, matrices, operators)
├── encoding/         # Data encoding methods
├── measure/          # Measurement utilities
└── utils/            # Helper functions (DTensor, interchange, OpenQASM)
```

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.5 or higher

### Install from source

```bash
# Clone the repository
git clone https://github.com/flagos-ai/FlagQuantum.git
cd flagquantum

# Install in development mode
pip install -e .
```

### Install with pip

```bash
pip install flagquantum
```

### Verify Installation

```python
import flagquantum as fq
print(fq.__version__)
```

## 🚀 Quick Start

### Basic Usage
```python
import flagquantum as fq
import torch

# Create a distributed quantum device
qdev = fq.DistributedQuantumDevice(n_wires=4, bsz=2, world_sz=1, device='cpu')

# Apply gates (functional style)
fq.h(qdev, wires=[0])
fq.rx(qdev, wires=[1], params=0.5)
fq.cx(qdev, wires=[0, 1])

# Measure all qubits
expectations = fq.measure_allZ(qdev)
print(expectations.shape)  # (2, 4)
```

### Parameterized Gates with Trainable Parameters
```python
# Create a gate with trainable parameter
rx_gate = fq.RX(wires=[0], trainable=True)
rx_gate(qdev)  # Apply to quantum device

# Optimize the parameter
optimizer = torch.optim.Adam([rx_gate.params])
for _ in range(100):
    optimizer.zero_grad()
    qdev.reset_states()
    rx_gate(qdev)
    loss = fq.measure_allZ(qdev).sum()
    loss.backward()
    optimizer.step()
```

### Quantum Encoding
```python
# Angle encoding
x = torch.randn(2, 4)  # batch=2, features=4
fq.angle_encoder(qdev, x, wires=[0, 1, 2, 3])

# Amplitude encoding
amplitudes = torch.randn(2, 16)  # 2^4 = 16 amplitudes
fq.amplitude_encoder(qdev, amplitudes)

# Custom encoding circuit
encoder = fq.GeneralEncoder([
    {"func": "ry", "wires": [0], "input_idx": 0},
    {"func": "ry", "wires": [1], "input_idx": 1},
    {"func": "cx", "wires": [0, 1]},
])
encoder(qdev, x)
```

### Export to Real Quantum Hardware

FlagQuantum circuits can be exported to OpenQASM 3.0 and run on **all major quantum computing platforms**:

```python
# Build your circuit
qdev = fq.DistributedQuantumDevice(n_wires=3, record_op=True)
fq.H(wires=[0])(qdev)
fq.RX(wires=[1], init_params=torch.tensor([0.5]))(qdev)
fq.CNOT(wires=[0, 1])(qdev)
fq.measure_allZ(qdev)

# Export to OpenQASM 3.0
fq.export_to_qasm(qdev, "circuit.qasm", version=3.0)

# Now run on ANY platform:
# - IBM Quantum (via Qiskit)
# - AWS Braket (IonQ, Rigetti)
# - Azure Quantum
# - IonQ direct
# - Rigetti direct
```

**Supported Platforms:**

| Platform | Support | Description |
|----------|---------|-------------|
| **IBM Quantum** | ✅ Native | Run on real IBM quantum processors |
| **AWS Braket** | ✅ Full | Submit to IonQ, Rigetti, and more |
| **Azure Quantum** | ✅ Full | OpenQASM as core intermediate representation |
| **IonQ** | ✅ Native | Direct hardware submission |
| **Rigetti** | ✅ Native | Superconducting qubit systems |
| **Q-CTRL Fire Opal** | ✅ Full | Hardware optimization services |

### Register Custom Gates
```python
from flagquantum.ops import register_gate
import torch

# Define custom gate matrix
my_gate = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
register_gate("my_gate", my_gate)

# Now available as:
# - fq.ops.registry.my_gate (functional)
# - fq.ops.registry.my_gate_inv (inverse)
# - fq.ops.registry.MY_GATE (operator class)
```

### Distributed Multi-GPU Execution
```bash
# Run with 4 GPUs
torchrun --nproc_per_node=4 your_script.py
```

```python
# In your script, world_sz is set automatically via torchrun
qdev = fq.DistributedQuantumDevice(n_wires=20, bsz=32, world_sz=4)
```

### Invertible Mode (Memory Efficient)
```python
qdev = fq.DistributedQuantumDevice(n_wires=10, bsz=64, invertible=True)
# Uses less memory during backpropagation
```

## 📚 Tutorials

Explore our tutorial series to learn how to use FlagQuantum effectively:

| # | Tutorial | Description |
|---|----------|-------------|
| 00 | [Understanding States](examples/tutorials/00_understanding_states.ipynb) | Quantum state representations and initialization |
| 01 | [Basic Operations](examples/tutorials/01_basic_operations.ipynb) | Single-qubit and two-qubit gates |
| 02 | [Measurement](examples/tutorials/02_measurement.ipynb) | Quantum measurement and expectation values |
| 03 | [Parameterized Gates](examples/tutorials/03_parameterized_gates.ipynb) | Trainable gates with automatic differentiation |
| 04 | [Quantum Circuit Builder](examples/tutorials/04_quantum_circuit_builder.ipynb) | Building and visualizing circuits |
| 05 | [Quantum Machine Learning](examples/tutorials/05_quantum_machine_learning.ipynb) | End-to-end QML training pipeline |

[→ View all tutorials](examples/tutorials/)

## 🧪 Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
python run_tests.py
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

Apache License 2.0

## 🙏 Acknowledgments

We would like to thank the following projects and organizations for their inspiration and reference:

- **[NVIDIA CUDA-Q](https://github.com/NVIDIA/cuda-quantum)** - For insights on GPU-accelerated quantum circuit simulation and distributed quantum computing
- **[MIT TorchQuantum](https://github.com/mit-han-lab/torchquantum)** - For inspiration on PyTorch-native quantum circuit representations
- **[IonQ's TQD](https://github.com/ionq/torchquantum-dist)** - For ideas on efficient state representations
- **[Xanadu's PennyLane](https://github.com/PennyLaneAI/pennylane)** - For the elegant functional API design and seamless integration with classical ML frameworks
- **[IBM's Qiskit](https://github.com/Qiskit/qiskit)** - For foundational concepts in quantum circuit construction and statevector simulation
- **[OpenQASM](https://github.com/openqasm/openqasm)** - For the industry-standard quantum circuit representation enabling cross-platform compatibility

This project is built with PyTorch's `DTensor` for distributed tensor operations, enabling scalable quantum state simulation across multiple devices. We are grateful to the broader quantum computing community whose open-source efforts continue to bridge classical and quantum machine learning.
