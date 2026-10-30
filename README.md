<div align="center">
  <img src="assets/logo_flagquantum.png" alt="FlagQuantum" width="520">

<p><strong>Quantum computing, built for learning.</strong></p>
<p>A PyTorch-first framework for differentiable quantum computing and quantum AI.</p>

[Qubit ordering](#qubit-ordering-qubit-0-is-the-most-significant-bit) · [Quick start](#install-and-train-your-first-quantum-model) · [Documentation](docs/README.md) · [Wiki](https://github.com/flagos-ai/FlagQuantum/wiki) · [Examples](examples/README.md)

</div>

Turn quantum circuits into trainable models. FlagQuantum brings PyTorch learning,
multiple simulation representations, and hardware execution into one workflow.
Its long-term goal is a continuous path from local scientific exploration to
distributed training, device modeling, and fault-tolerant quantum computing research.

- **Train with PyTorch.** Compose quantum and classical layers with autograd and
  familiar optimizers.
- **Choose the representation.** Statevector, matrix product state (MPS), and
  tensor-network simulation for different circuit structures and resource budgets.
- **Connect simulation to hardware.** Keep the circuit and requested observable
  explicit as you move between supported execution targets.

Support is specific to each backend and workload. Local training and selected
distributed paths have correctness evidence.
See the [validation scope](docs/reference/PUBLICATION_VALIDATION.md) for what has
been tested and what remains a research goal.

## Qubit ordering: qubit 0 is the most-significant bit

> [!IMPORTANT]
> **FlagQuantum uses big-endian qubit order.** For an `n`-qubit register,
> basis states are written as `|q0 q1 ... q(n-1)>`; qubit `0` is the
> most-significant bit (MSB), and qubit `n-1` is the least-significant bit (LSB).

By default, a circuit starts in `|00...0>`, with amplitude `1` at statevector
index `0` and `0` everywhere else. For example, applying `X` to qubit `0` of
`|000>` produces `|100>` at index `4`; applying it to qubit `2` produces
`|001>` at index `1`.
This matches the tensor-product notation
`|q0> ⊗ |q1> ⊗ ... ⊗ |q(n-1)>`, keeping equations, matrices, and qubit
lists in the same left-to-right order.

| Framework or service | Exposed qubit/result order | Compared with FlagQuantum |
| --- | --- | --- |
| **[FlagQuantum](https://flagquantum.com)** | Qubit `0` is the MSB | Native convention |
| [Cirq](https://quantumai.google/reference/python/cirq/StateVectorMixin) | The first qubit in `qubit_order` is the MSB | Same when ordered `[q0, q1, ...]` |
| [PennyLane](https://docs.pennylane.ai/en/stable/code/api/pennylane.BasisState.html) | Basis-state digits follow the supplied qubit sequence | Same when ordered `[0, 1, ...]` |
| [Qiskit](https://quantum.cloud.ibm.com/docs/en/guides/bit-ordering) | Qubit `0` is the LSB | Opposite significance |
| [NVIDIA cuStateVec](https://docs.nvidia.com/cuda/cuquantum/24.08.1/custatevec/overview.html#bit-ordering) | Index bit `0` is the LSB | Opposite significance |
| [Jiuding QuantumAI Cloud](https://cloud.flagquantum.com) | Runs FlagQuantum remotely and returns FlagQuantum-ordered results | Same |
| [Quafu](https://quafu.com.cn) | Provider count strings are converted to logical qubit order, with qubit `0` on the left | Normalized to the same order |
| [Amazon Braket](https://aws.amazon.com/braket/) | Ascending qubit order, with qubit `0` as the MSB | Same |
| [Azure Quantum](https://learn.microsoft.com/azure/quantum/overview-azure-quantum) | Depends on the selected Azure target and hardware provider | No platform-wide convention |

Endianness affects statevector, matrix, and bitstring indexing, not the
`(control, target)` order of gate arguments. Convert representations at framework
boundaries instead of reversing gate arguments. Cloud services commonly expose
measurement results rather than raw statevectors, so the service rows describe
the order visible at FlagQuantum's result boundary.

## Install and train your first quantum model

Requires Python **3.10–3.12**. Choose the command for your computer.

**macOS (Apple Silicon):**

```console
python -m pip install flagquantum
```

This release publishes macOS wheels for Apple Silicon only; Intel macOS is
not a supported binary installation target.

**Linux or Windows, CPU only:**

```console
python -m pip install "torch>=2.13,<2.15" --index-url https://download.pytorch.org/whl/cpu
python -m pip install flagquantum
```

FlagQuantum's compiled extension uses PyTorch's Stable ABI and is tested with
PyTorch 2.10.x through 2.14.x in compatibility CI. New CPU/CUDA environments
should use the recommended 2.13.x or 2.14.x runtime above. The first CPU install downloads about 200 MB of
PyTorch. On a clean GitHub Ubuntu runner it took about 16 seconds; slower
networks will take longer.

**NVIDIA GPU:** Select a PyTorch 2.13.x or 2.14.x CUDA command from the
[PyTorch installer](https://pytorch.org/get-started/locally/), then run:

```console
python -m pip install flagquantum
```

Installing from source? Follow the short setup in [CONTRIBUTING.md](CONTRIBUTING.md).

**Native accelerator support:** FlagQuantum runs directly on the following
vendor-provided PyTorch runtimes.

|  |  |
| --- | --- |
| **NVIDIA CUDA** | Upstream PyTorch |
| **Hygon DCU** | Vendor PyTorch runtime |
| **Moore Threads MUSA** | Vendor PyTorch runtime |
| **MetaX** | Vendor PyTorch runtime |
| **Biren** | Vendor PyTorch runtime |
| **Huawei Ascend 910C** | Torch-FL with the [pinned Ascend and FlagCX fixes](docs/guides/ASCEND_910C.md) |

Keep the vendor runtime installed, then run `python -m pip install flagquantum`.
The core requirement accepts `torch>=2.10,<2.15`, including local version labels
such as `2.10.0+metax3.8.1.0`; do not replace a managed runtime with an upstream
CPU/CUDA wheel. Avoid `--upgrade`/`--upgrade-strategy eager` when retaining a
managed runtime. The `invertible` extra still requires PyTorch 2.13.x and is not
a vendor-runtime installation shortcut.

**Huawei Ascend 910C support through Torch-FL:** Native complex64/complex128
statevector, MPS, tensor-network and density-matrix execution, autograd and hybrid
training, noise, TEBD, native dynamics and QEC predecoder training have hardware
validation. Sharded statevector/MPS/TN training, two-host communication and
checkpoint restart are also covered. Install the fixed source configuration in
the [910C guide](docs/guides/ASCEND_910C.md); previously published wheels alone
do not contain the required fixes.

```console
FLAGQUANTUM_TEST_ASCEND_910C=1 python -m pytest \
  tests/integration/test_ascend_circuit_workflows.py \
  tests/integration/test_ascend_native_dynamics.py -q
```

Quantum state, gradients and optimizer updates remain on `flagos`. The
[acceptance record](artifacts/ascend_910c_20261030.json) states the measured
functions and topology. This is hardware development support; production,
performance and capacity certification remain outside the claim. Optional
third-party runtimes retain their own hardware requirements.

To build a CPU or GPU environment with QSteed and optional JAX, follow the
[container guide](docker/dev/README.md).

Build a two-qubit circuit and learn its rotation angle by minimizing ⟨Z₀⟩. `fq.Module` exposes the quantum model to PyTorch; `outputs` selects what
to measure after training.

```python
import torch
import flagquantum as fq


def circuit(parameters):
    return fq.Circuit(2).ry(0, parameters[0]).cx(0, 1)


model = fq.Module(circuit, n_parameters=1, init=torch.tensor([0.25]))
training = fq.train(
    model,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.05),
    objective=lambda z: z.mean(),
    steps=10,
)

trained_circuit = circuit(next(model.parameters()).detach())
measurement = fq.expectation(fq.Z(0))
result = fq.run(trained_circuit, outputs=measurement)
print(result.expectation())
```

For a complete classical–quantum model, follow the
[hybrid training example](examples/quick_start.py). The published wheel contains
the `flagquantum` package only, so `examples/` comes from a clone of this
repository: `git clone https://github.com/flagos-ai/FlagQuantum.git` and run the
examples from that working copy.

## Same circuit. Different execution targets.

The experimental adapters can evaluate the same observable on a
[Jiuding QuantumAI Cloud](https://cloud.flagquantum.com) GPU workspace or Quafu
quantum hardware. Configure the
[Jiuding QuantumAI Cloud workspace and credentials](docs/guides/JIUDING.md) or
the [Quafu token](docs/guides/QUAFU_BACKEND.md) before running the corresponding
call.

```python
# Managed GPU simulation on Jiuding QuantumAI Cloud
jiuding_result = fq.run(
    trained_circuit, target="jiuding:gpu", outputs=measurement,
)

# Quantum hardware: service compilation and estimation from measured shots
quafu_result = fq.run(
    trained_circuit, target="quafu:Baihua",
    outputs=measurement, shots=1024,
)
```

The Jiuding QuantumAI Cloud call above needs `JIUDING_WORKSPACE` set to the
running workspace name when it is invoked from outside that workspace; inside
the workspace the name is discovered automatically. See
[repeated low-latency workspace execution](docs/guides/JIUDING.md#repeated-low-latency-workspace-execution).

Direct Quafu submission is available in the 0.3 release line. Install
`flagquantum==0.3.0` for this path. With the older 0.2.0 release, use the
documented local QSteed compilation path instead: pass `compiler="qsteed"` after installing the
separate
[FlagQuantum Compiler QSteed](https://github.com/FlagQuantum/FlagQuantum-Compiler-QSteed)
plugin, as described in
[optional local compiler plugin](docs/guides/QUAFU_BACKEND.md#optional-local-compiler-plugin).
The `quafu` extra does not install that plugin.

Jiuding QuantumAI Cloud computes a simulated expectation; Quafu estimates it
from hardware measurements. The training example runs on your local machine;
these calls evaluate the trained circuit remotely. Live provider access is
required and is not certified by the local or A800 checks.

## Go further

**Connect simulation with device observations.** Use [QPU digital twins](flagquantum/twin/README.md)
to compare calibration-based model predictions with measured counts. See the
experiment guide for task binding and the scope of hardware validation.

**Toward fault-tolerant quantum computing.** Start with a local
[QEC memory experiment](flagquantum/qec/README.md) connecting syndrome extraction,
decoding, and correction. Logical operations and hardware feedback are longer-term
research goals.

[Quantum AI tutorials](examples/single_machine_quantum_ai/README.md) ·
[Distributed statevector](examples/distributed_statevector_topologies/README.md) ·
[Distributed MPS](examples/distributed_mps/README.md) ·
[ARCHITECTURE.md](ARCHITECTURE.md)

## Documentation and Wiki

Start with the [Wiki home](https://github.com/flagos-ai/FlagQuantum/wiki) or browse the
[Wiki Directory](https://github.com/flagos-ai/FlagQuantum/wiki/Wiki-Directory) for the
full learning path. Find a guide by topic:

| Topic | Wiki guides |
| --- | --- |
| Get started and understand the basics | [Installation and Compatibility](https://github.com/flagos-ai/FlagQuantum/wiki/Installation-and-Compatibility) · [Quick Start](https://github.com/flagos-ai/FlagQuantum/wiki/Quick-Start) · [Core Concepts](https://github.com/flagos-ai/FlagQuantum/wiki/Core-Concepts) · [Qubit Ordering and Result Conventions](https://github.com/flagos-ai/FlagQuantum/wiki/Qubit-Ordering-and-Result-Conventions) |
| Build circuits and train models | [Building and Simulating Circuits](https://github.com/flagos-ai/FlagQuantum/wiki/Building-and-Simulating-Circuits) · [Training Parameterized Circuits](https://github.com/flagos-ai/FlagQuantum/wiki/Training-Parameterized-Circuits) · [Hybrid Classical-Quantum Models](https://github.com/flagos-ai/FlagQuantum/wiki/Hybrid-Classical-Quantum-Models) |
| Choose simulation and compute resources | [Choosing a Simulation Method](https://github.com/flagos-ai/FlagQuantum/wiki/Choosing-a-Simulation-Method) · [Accelerator Setup](https://github.com/flagos-ai/FlagQuantum/wiki/Accelerator-Setup) · [Distributed Simulation and Training](https://github.com/flagos-ai/FlagQuantum/wiki/Distributed-Simulation-and-Training) |
| Compile circuits and model noise | [Circuit Compilation and Interoperability](https://github.com/flagos-ai/FlagQuantum/wiki/Circuit-Compilation-and-Interoperability) · [Noise and Open-System Simulation](https://github.com/flagos-ai/FlagQuantum/wiki/Noise-and-Open-System-Simulation) |
| Explore QEC, cloud execution, and device models | [Quantum Error Correction Experiments](https://github.com/flagos-ai/FlagQuantum/wiki/Quantum-Error-Correction-Experiments) · [Cloud and Quantum Hardware Execution](https://github.com/flagos-ai/FlagQuantum/wiki/Cloud-and-Quantum-Hardware-Execution) · [QPU Digital Twins](https://github.com/flagos-ai/FlagQuantum/wiki/QPU-Digital-Twins) |
| Reproduce results, troubleshoot, and contribute | [Reproducibility and Benchmarks](https://github.com/flagos-ai/FlagQuantum/wiki/Reproducibility-and-Benchmarks) · [Troubleshooting and FAQ](https://github.com/flagos-ai/FlagQuantum/wiki/Troubleshooting-and-FAQ) · [Development and Contribution Guide](https://github.com/flagos-ai/FlagQuantum/wiki/Development-and-Contribution-Guide) · [Documentation Guide](https://github.com/flagos-ai/FlagQuantum/wiki/Documentation-Guide) |

The Wiki guides currently target **FlagQuantum 0.3.1** and state their verification
scope. For versioned API contracts, executable examples, and support boundaries,
use the [repository documentation](docs/README.md), [examples](examples/README.md),
and [capability catalog](docs/generated/CAPABILITIES.md) matching your installed
release; `main` can describe newer development.

## Citing FlagQuantum

If you use FlagQuantum in work you publish, cite the software through
[`CITATION.cff`](CITATION.cff). GitHub renders it under **Cite this
repository** in APA and BibTeX, and `cffconvert` produces other formats from
the same file, so a citation does not have to be assembled by hand.

Cite the version you ran rather than `main`: a result is reproducible against
the release it was measured on. The file records 0.3.1; adjust the version and
date if you ran an earlier release.

<!-- BEGIN GENERATED CAPABILITY_SUMMARY -->
Support varies by execution path. See the [capability catalog](docs/generated/CAPABILITIES.md) for maturity and limitations.
<!-- END GENERATED CAPABILITY_SUMMARY -->

<!-- BEGIN GENERATED PERFORMANCE_CLAIMS -->
[Benchmarks and validated results](docs/generated/CAPABILITIES.md#validated-public-performance-claims)
<!-- END GENERATED PERFORMANCE_CLAIMS -->

---

[Contributing](CONTRIBUTING.md) · [Apache License 2.0](LICENSE)
