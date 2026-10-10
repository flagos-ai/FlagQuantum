<div align="center">
  <img src="assets/logo_flagquantum.png" alt="FlagQuantum" width="520">

<p><strong>Quantum computing, built for learning.</strong></p>
<p>A PyTorch-first framework for differentiable quantum computing and quantum AI.</p>

[Quick start](#install-and-train-your-first-quantum-model) · [Documentation](docs/README.md) · [Examples](examples/README.md)

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

## Install and train your first quantum model

Requires Python **3.10–3.12**. Choose the command for your computer.

**macOS (Apple Silicon):**

```console
python -m pip install flagquantum
```

PyTorch 2.13 does not publish macOS Intel (`x86_64`) wheels, so this
FlagQuantum release cannot provide a working Intel macOS installation.

**Linux or Windows, CPU only:**

```console
python -m pip install "torch>=2.13,<2.15" --index-url https://download.pytorch.org/whl/cpu
python -m pip install flagquantum
```

FlagQuantum's compiled extension uses PyTorch's Stable ABI and is tested with
PyTorch 2.13.x and 2.14.x. The first CPU install downloads about 200 MB of
PyTorch. On a clean GitHub Ubuntu runner it took about 16 seconds; slower
networks will take longer.

**GPU:** Select a PyTorch 2.13.x or 2.14.x command matching your accelerator from the
[PyTorch installer](https://pytorch.org/get-started/locally/), then run:

```console
python -m pip install flagquantum
```

Installing from source? Follow the short setup in [CONTRIBUTING.md](CONTRIBUTING.md).

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
[hybrid training example](examples/quick_start.py).

## Same circuit. Different execution targets.

The experimental adapters can evaluate the same observable on a Jiuding GPU
workspace or Quafu quantum hardware. Configure the [Jiuding workspace and credentials](docs/guides/JIUDING.md)
or the [Quafu token](docs/guides/QUAFU_BACKEND.md) before
running the corresponding call.

```python
# GPU simulation in a running Jiuding workspace
jiuding_result = fq.run(
    trained_circuit, target="jiuding:gpu", outputs=measurement,
)

# Quantum hardware: service compilation and estimation from measured shots
quafu_result = fq.run(
    trained_circuit, target="quafu:Baihua",
    outputs=measurement, shots=1024,
)
```

The Jiuding call above needs `JIUDING_WORKSPACE` set to the running workspace
name when it is invoked from outside that workspace; inside the workspace the
name is discovered automatically. See
[repeated low-latency workspace execution](docs/guides/JIUDING.md#repeated-low-latency-workspace-execution).

Direct Quafu submission is available in the 0.3 release line, including the
`0.3.0rc1` prerelease. With the older 0.2.0 release, use the documented local
QSteed compilation path instead: pass `compiler="qsteed"` after installing the
separate
[FlagQuantum Compiler QSteed](https://github.com/FlagQuantum/FlagQuantum-Compiler-QSteed)
plugin, as described in
[optional local compiler plugin](docs/guides/QUAFU_BACKEND.md#optional-local-compiler-plugin).
The `quafu` extra does not install that plugin.

Jiuding computes a simulated expectation; Quafu estimates it from hardware
measurements. The training example runs on your local machine; these calls
evaluate the trained circuit remotely. Live provider access is required and is
not certified by the local or A800 checks.

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

<!-- BEGIN GENERATED CAPABILITY_SUMMARY -->
Support varies by execution path. See the [capability catalog](docs/generated/CAPABILITIES.md) for maturity and limitations.
<!-- END GENERATED CAPABILITY_SUMMARY -->

<!-- BEGIN GENERATED PERFORMANCE_CLAIMS -->
[Benchmarks and validated results](docs/generated/CAPABILITIES.md#validated-public-performance-claims)
<!-- END GENERATED PERFORMANCE_CLAIMS -->

---

[Contributing](CONTRIBUTING.md) · [Apache License 2.0](LICENSE)
