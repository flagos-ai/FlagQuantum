<div align="center">
  <img src="assets/logo_flagquantum.png" alt="FlagQuantum" width="380">

<h1>FlagQuantum</h1>

<p><strong>A PyTorch-first framework for differentiable quantum computing and quantum AI.</strong></p>

<p>Build quantum circuits, train hybrid models, select simulation representations, and package trained programs for supported quantum providers.</p>

[Quick start](#quick-start) ·
[Capabilities](#current-capabilities) ·
[Architecture](#architecture) ·
[Examples](#examples-and-guides) ·
[Documentation](docs/README.md)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-red.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-blue.svg)](https://python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>

FlagQuantum connects quantum programs to ordinary PyTorch learning workflows.
`fq.Circuit` describes a program, `fq.Module` makes its parameters trainable,
and FlagQuantum IR carries its semantics through compilation, planning,
execution, and deployment.

This repository contains the **vNext development line**. Local CPU execution is
the starting point; accelerator, distributed, provider, and experimental paths
have separate support boundaries. The [capability catalog](docs/generated/CAPABILITIES.md)
records their maturity and evidence.

## Quick start

Use Python **3.10–3.12**. From the repository root, install the package into your
Python environment:

```console
python -m pip install -e .
python examples/quick_start.py --mode sv
```

The maintained example trains a classical `torch.nn.Linear` layer and an
`fq.Module` quantum layer in one optimizer loop, then checks an analytical
reference. It runs locally without cloud credentials. To explore the same
example with MPS or tensor-network execution:

```console
python examples/quick_start.py --mode mps
python examples/quick_start.py --mode tn
```

A minimal API workflow trains a circuit and checks it locally. The trained
program can then be executed on Jiuding or Quafu using the same `fq.run` entry
point:

```python
import flagquantum as fq
import torch

def build_program(parameters, inputs=None):
    return fq.Circuit(n_qubits=2).ry(0, parameters[0]).cx(0, 1)

model = fq.Module(build_program, n_parameters=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
training = fq.train(
    model,
    optimizer=optimizer,
    objective=lambda value: value.mean(),
    steps=10,
)
trained_program = build_program(next(model.parameters()).detach())
result = fq.run(trained_program)
```

### Execute on Jiuding or Quafu

After local training, choose a remote target. These calls execute real remote
work and require the corresponding environment to be configured first.

**Jiuding GPU simulation:** configure Jiuding credentials and SSH access, start
a compatible GPU workspace, and set `JIUDING_WORKSPACE` to its name as described
in the [Jiuding guide](docs/guides/JIUDING.md). Then run the trained circuit on
its resident GPU executor:

```python
jiuding_result = fq.run(
    trained_program,
    target="jiuding:gpu",
    outputs=fq.counts(),
    shots=1024,
)
print(jiuding_result.measurement("counts"))
```

This reuses a running workspace. For a separately scheduled CPU or GPU batch
task, use the [Jiuding submission example](examples/remote/jiuding_submit.py).

**Quafu quantum hardware:** install the `flagquantum-compiler-qsteed` plugin,
configure `QUAFU_API_TOKEN`, and select an available backend following the
[Quafu guide](docs/guides/QUAFU_BACKEND.md):

```python
quafu_result = fq.run(
    trained_program,
    compiler="qsteed",
    target="quafu:Baihua",
    shots=1024,
)
print(quafu_result.measurement("counts"))
```

The Quafu call compiles, packages, submits, and waits for the hardware result.
Use `flagquantum.deployment.create_deployment_package` separately when you need
to inspect or save a deployment artifact before submission. The examples above
train locally and execute the bound circuit remotely; they do not perform
remote training.

Continue with the [annotated quick start](examples/quick_start.py),
[training examples](examples/single_machine_quantum_ai/README.md), and
[execution result contract](docs/reference/RUNTIME_RESULT_CONTRACT.md).

### Optional dependencies

PyTorch is the required numerical dependency. Install extra environments only
for the workflows you use:

```console
python -m pip install -e '.[dev]'       # tests, lint, typing, and builds
python -m pip install -e '.[jax]'       # optional JAX quantum kernels
python -m pip install -e '.[cuda]'      # optional Triton kernels
python -m pip install -e '.[examples]'  # model and dataset examples
python -m pip install -e '.[viz]'       # Matplotlib visualization
```

Provider and interoperability extras are defined in [pyproject.toml](pyproject.toml).
Some have narrower Python requirements than the core package.

## Current capabilities

<!-- BEGIN GENERATED CAPABILITY_SUMMARY -->
Selected core capabilities; maturity applies only within each documented scope.

| Capability | Maturity |
| --- | --- |
| [Unified circuit API and FlagQuantum IR](docs/reference/API.md) | `release_certified` |
| [Local statevector simulation and training](examples/single_machine_quantum_ai/README.md) | `production_supported` |
| [Sharded statevector training](examples/distributed_statevector_topologies/README.md) | `production_supported` |
| [Differentiable and sharded MPS training](examples/distributed_mps/README.md) | `development_evidence` |
| [Tensor-network execution and training](docs/reference/KNOWN_LIMITATIONS.md) | `experimental` |
| [Circuit packaging and cloud deployment](docs/reference/API.md) | `development_evidence` |

See the [full capability catalog](docs/generated/CAPABILITIES.md) for support
boundaries, hardware evidence, and experimental paths including FlagOS,
interoperability, noise, and precision research. Levels are generated from the
[capability matrix](capability-maturity.toml); they do not certify every device
or workload, and development evidence does not establish production support.
<!-- END GENERATED CAPABILITY_SUMMARY -->

## Architecture

The user workflow is centered on a small set of entry points:

```text
fq.Circuit / fq.Module
          |
          v
   FlagQuantum IR
          |
          +-- compilation and export
          +-- runtime planning and execution
          |      +-- local statevector / MPS / tensor network
          |      +-- supported sharded execution paths
          +-- deployment packages and remote providers
```

The implementation separates program semantics, execution policy, numerical
algorithms, and resource access:

| Domain | Responsibility |
| --- | --- |
| [Core](flagquantum/core/README.md) | IR, operators, parameters, target capabilities, and shared contracts |
| [Compiler](flagquantum/compiler/README.md) | Program transformation, routing, legalization, and export |
| [Runtime](flagquantum/runtime/README.md) | Planning, execution orchestration, and result evidence |
| [Simulation](flagquantum/simulation/README.md) | Numerical algorithms and quantum kernels |
| [Compute](flagquantum/compute/README.md) | Devices and resources controlled directly by the current process |
| [Remote](flagquantum/remote/README.md) | External task submission, status, and result adapters |
| [Ecosystem](flagquantum/ecosystem/README.md) | Framework conversion and extension adapters |
| [Services](flagquantum/services/README.md) | Reusable application workflows over framework APIs |

See the [architecture documentation](docs/architecture/README.md) for domain
boundaries and migration details, and the [stable API inventory](docs/generated/STABLE_API.md)
for protected public interfaces. Internal compiler experiments do not expand
the public API.

### FlagQuantum and FlagOS

FlagOS is the target integration for heterogeneous accelerator execution.
FlagQuantum owns quantum semantics, simulation algorithms, differentiation,
and execution planning; the Compute boundary adapts directly controlled
resources, including the optional FlagOS/Torch-FL path.

CPU and native PyTorch paths remain first-class. FlagOS is not required for
the quick start. JAX is an optional kernel path behind the PyTorch interface.
Hardware portability and distributed support require evidence for the specific
route; a CUDA-backed FlagOS run does not certify a domestic accelerator.

Follow the [FlagOS release train](docs/roadmap/FLAGOS_ALIGNED_RELEASE_TRAIN.md)
and [FlagOS 2026 workshop](workshops/flagos2026/README.md) for integration work.

## Examples and guides

| Goal | Start here |
| --- | --- |
| Build and train a first circuit | [Quick start](examples/quick_start.py) |
| Optimize and compile a circuit | [Compiler guide](flagquantum/compiler/README.md) |
| Run VQE or train a hybrid model | [Single-machine quantum AI](examples/single_machine_quantum_ai/README.md) |
| Explore low-entanglement systems | [1,000-qubit MPS training](examples/single_machine_quantum_ai/05_mps_1000q_dimer_training.py) |
| Compare simulation representations | [Switch VQE across SV, MPS, and TN](examples/vqe_switch_sv_mps_tn.py) |
| Partition a workload across devices | [Distributed statevector](examples/distributed_statevector_topologies/README.md) and [distributed MPS](examples/distributed_mps/README.md) |
| Package a trained circuit | [Training to deployment](examples/train_parameterized_circuit_then_deploy.py) |
| Execute on a Jiuding GPU workspace | [Jiuding guide](docs/guides/JIUDING.md) |
| Submit to Quafu quantum hardware | [Quafu backend guide](docs/guides/QUAFU_BACKEND.md) |
| Check support before an experiment | [Capability catalog](docs/generated/CAPABILITIES.md) and [known limitations](docs/reference/KNOWN_LIMITATIONS.md) |

## Verified results

Distributed capacity claims require one logical workload to be partitioned
across ranks. Replicated execution and process count alone do not establish
scalability. The record below retains its exact workload scope and recorded
environment; it is not a general performance or production-readiness claim.

<details>
<summary><strong>Artifact-backed capacity result and evidence identity</strong></summary>

<!-- BEGIN GENERATED PERFORMANCE_CLAIMS -->
Every value below is read from a hash-bound raw artifact. Missing or changed
evidence makes the source-of-truth check fail closed.

| Claim | Maturity and scope | Artifact-derived result | Recorded environment | Evidence identity |
| --- | --- | --- | --- | --- |
| **Sharded MPS exact-workload capacity**<br>`mps-capacity-131072-chi768-20260806` | `development_evidence`<br>One batch-one, complex64, χ768 MPS training step for the checked-in all-rank and all-boundary workload. This is not arbitrary statevector capacity, fixed-plan strong scaling, or release evidence. | **Sites:** 131,072<br>**Logical MPS state:** 1,236,780,012,864 bytes (1,151.84 GiB)<br>**Maximum elapsed time:** 367.37 s<br>**Maximum peak allocated memory per rank:** 77,745,407,488 bytes (72.41 GiB)<br>**Cumulative discarded weight:** 8.39e-06 | **Ranks:** 16<br>**Reported device memory per rank:** 85,093,777,408 bytes (79.25 GiB)<br>**CUDA allocator policy:** expandable_segments:True<br>**Topology fingerprint:** c74a91e3a224a4dd414cfbfbcb31da7570880d42e6aa4eaccecf4b85427940a5<br>**Metadata boundary:** The artifact records rank count, per-rank device memory, topology fingerprint, and CUDA allocator policy. It does not record the exact GPU model or Python, PyTorch, CUDA, NCCL, driver, host, and operating-system versions, so the claim is restricted to the recorded environment fields. | [raw JSON](benchmarks/results/local/mps_capacity_131072q_chi768_16xa800_repeat_complete_20260806.json)<br>SHA-256 `df8c19b74cc173799e3894025fb8e72de72a786d4542a9aa5e67687298f482f5`<br>code `9d56a6ecd78b06f11b9ee6e8aadcbe9644f2c708` |
<!-- END GENERATED PERFORMANCE_CLAIMS -->

</details>

The benchmark interface provides discoverable workloads and structured output:

```console
flagquantum-benchmark list
flagquantum-benchmark info statevector_local
flagquantum-benchmark run environment_probe \
  --json-output benchmarks/results/local/environment.json
```

See the [benchmark guide](benchmarks/README.md) for local, distributed, and
release-evidence workflows.

## Development

Install the development extra and run the default test tier:

```console
python -m pip install -e '.[dev]'
python tools/ci_tier.py pr-default
```

The README API example has a focused executable check:

```console
python -m pytest tests/test_readme_golden_path.py -q
```

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md) before editing.
Use the owning domain's README for its shortest change path and the
[testing manual](docs/development/TESTING.md) for additional runtime,
distributed, accelerator, and release checks.

Capability tables and performance records in this README are generated from
repository manifests and evidence. Validate them with:

```console
python tools/docs_source_of_truth.py --check
```

## License

FlagQuantum is licensed under the [Apache License 2.0](LICENSE).
