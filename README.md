<div align="center">
  <img src="assets/logo.png" alt="FlagQuantum Logo" width="380">
</div>

# FlagQuantum

A high-performance distributed quantum statevector simulator built on PyTorch, enabling quantum circuit simulation across multiple GPUs with automatic sharding and resharding.

## Features

- **Distributed Statevector Simulation**: Leverage multiple GPUs to simulate large quantum circuits using `DTensor` from `torch.distributed`
- **Automatic Resharding**: Intelligently redistributes statevectors to minimize communication overhead during gate operations
- **Comprehensive Gate Set**: Includes Pauli, Clifford, rotation, and controlled gates with parameterized support
- **Invertible Backpropagation**: Memory-efficient gradient computation for trainable quantum circuits
- **Custom Gate Registration**: Extend the library with your own gates without modifying the core
- **Post-Selection & Noise Models**: Built-in support for measurement post-selection and depolarizing noise
- **Flexible Encoding**: Multiple encoding schemes (angle, amplitude, basis) for classical data embedding

## Architecture

```
flagquantum/
├── devices/          # Quantum device implementations
├── ops/              # Quantum operations (gates, matrices, operators)
├── encoding/         # Data encoding methods
├── measure/          # Measurement utilities
└── utils/            # Helper functions (DTensor, interchange)
```

## Installation

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.5 or higher

### Install from source

```bash
# Clone the repository
git clone https://github.com/flagquantum/flagquantum.git
cd flagquantum

# Install in development mode
pip install -e .
```

### Install with pip (when available)

```bash
pip install flagquantum
```

### Verify Installation

```python
import flagquantum as fq
print(fq.__version__)
```

## Quick Start

### Basic Usage
```python
import torch
import flagquantum as fq

# Create a distributed quantum device
device = fq.DistributedQuantumDevice(n_wires=4, bsz=2, world_sz=1)

# Apply gates (functional style)
fq.h(device, wires=[0])
fq.rx(device, wires=[1], params=0.5)
fq.cx(device, wires=[0, 1])

# Measure all qubits
expectations = fq.measure_allZ(device)
print(expectations.shape)  # (2, 4)
```

### Parameterized Gates with Trainable Parameters
```python
# Create a gate with trainable parameter
rx_gate = fq.RX(wires=[0], trainable=True)
rx_gate(device)  # Apply to device

# Optimize the parameter
optimizer = torch.optim.Adam([rx_gate.params])
for _ in range(100):
    optimizer.zero_grad()
    device.reset_states()
    rx_gate(device)
    loss = fq.measure_allZ(device).sum()
    loss.backward()
    optimizer.step()
```

### Quantum Encoding
```python
# Angle encoding
x = torch.randn(4, 4)  # batch=4, features=4
fq.angle_encoder(device, x, wires=[0, 1, 2, 3])

# Amplitude encoding
amplitudes = torch.randn(4, 16)  # 2^4 = 16 amplitudes
fq.amplitude_encoder(device, amplitudes)

# Custom encoding circuit
encoder = fq.GeneralEncoder([
    {"func": "ry", "wires": [0], "input_idx": 0},
    {"func": "ry", "wires": [1], "input_idx": 1},
    {"func": "cx", "wires": [0, 1]},
])
encoder(device, x)
```

### Register Custom Gates
```python
import torch
from flagquantum.ops import register_gate

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
device = fq.DistributedQuantumDevice(n_wires=20, bsz=32, world_sz=4)
```

### Invertible Mode (Memory Efficient)
```python
device = fq.DistributedQuantumDevice(n_wires=10, bsz=64, invertible=True)
# Uses less memory during backpropagation
```

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
python run_tests.py
```

## License

Apache License 2.0

## Acknowledgments

We would like to thank the following projects and organizations for their inspiration and reference:

- **[NVIDIA CUDA-Q](https://github.com/NVIDIA/cuda-quantum)** - For insights on GPU-accelerated quantum circuit simulation and distributed quantum computing
- **[MIT TorchQuantum](https://github.com/mit-han-lab/torchquantum)** - For inspiration on PyTorch-native quantum circuit representations
- **[IonQ's TQD](https://github.com/ionq/torchquantum-dist)** - For ideas on efficient state representations
- **[Xanadu's PennyLane](https://github.com/PennyLaneAI/pennylane)** - For the elegant functional API design and seamless integration with classical ML frameworks
- **[IBM's Qiskit](https://github.com/Qiskit/qiskit)** - For foundational concepts in quantum circuit construction and statevector simulation

This project is built with PyTorch's `DTensor` for distributed tensor operations, enabling scalable quantum state simulation across multiple devices. We are grateful to the broader quantum computing community whose open-source efforts continue to bridge classical and quantum machine learning.
