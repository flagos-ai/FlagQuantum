# FlagQuantum package map

This file is the short guide to the source tree. The architecture contract and
long-term rationale live in
[`docs/architecture/LONG_HORIZON_ARCHITECTURE.md`](../docs/architecture/LONG_HORIZON_ARCHITECTURE.md).

## Normal execution path

```text
fq.Circuit / fq.Module
        |
        v
Core IR -> Compiler -> Runtime -> Simulation -> result
                         |  |
                         |  +-> Compute (direct CPU/GPU/NPU)
                         +----> Remote (QPU, GPU/HPC service, or cloud platform)
```

Core supplies the shared vocabulary throughout this path; it is not an
orchestration service. Runtime chooses and organizes execution. Simulation owns
numerical methods. Compute isolates resources controlled by the current process;
Remote isolates external task control planes.

## Source domains

| Directory | Owns | Does not own | Start here |
| --- | --- | --- | --- |
| `core/` | IR, operator semantics, artifacts, capabilities, shared configuration | execution policy, kernels, vendor SDKs | `core/README.md` |
| `compiler/` | validation, optimization, lowering, routing, code generation | device lifecycle, execution, simulation | `compiler/README.md` |
| `runtime/` | planning, execution lifecycle, backend selection, distributed coordination, results | compiler passes, numerical algorithms, vendor integration | `runtime/README.md` |
| `simulation/` | statevector, MPS, tensor-network, noise and precision kernels | resource policy, credentials, remote jobs | `simulation/README.md` |
| `compute/` | direct CPU/GPU/NPU lifecycle, capabilities, precision and communication | remote jobs, scheduling policy, simulator algorithms | `compute/README.md` |
| `remote/` | external target discovery, submission, status and result decoding | direct device lifecycle, scheduling policy, simulator algorithms | `remote/README.md` |
| `ecosystem/` | PyTorch, JAX, Qiskit and format boundary adapters | a second IR or runtime | `ecosystem/README.md` |
| `services/` | capability discovery and composite preflight workflows | pass-through API wrappers, MCP transport or LLM policy | `services/README.md` |
| `algorithms/` | user-facing algorithm composition | runtime or backend internals | `algorithms/README.md` |
| `benchmarking/` | reproducible measurements and evidence generation | alternate execution paths | `benchmarking/README.md` |
| `testing/` | reusable conformance and certification helpers | production execution | package modules and `tests/` scenarios |

## Public facades and supporting namespaces

The package root is deliberately small: normal use starts with
`import flagquantum as fq`. The root-level `circuit.py`, `training.py`,
`models.py`, `operators.py`, `gradients.py`, `dynamic.py`, and `errors.py`
preserve reviewed user-facing concepts; they are not general implementation
directories.

These explicit namespaces also remain intentional:

- `backends/` is the stable expert facade for backend-native results. Its
  implementations live in Runtime or Simulation.
- `noise/` owns backend-neutral channel and noise-model semantics.
- `deployment/` owns sealed execution packages and target-neutral submission
  contracts; concrete external adapters live in Remote.
- `drawer/` owns visualization and IR-to-drawing adaptation.
- `experimental/` contains APIs with no compatibility guarantee.
- `utils/` contains the existing QASM/QCIS exporters until Compiler covers
  their full behavior and an approved migration can remove the namespace.

Do not create another top-level domain to hold code that already has an owner.
Generated directories such as `__pycache__` are not part of the architecture.

## Where to make a change

- Change a gate's meaning or IR validation in Core.
- Change program transformation, routing or emission in Compiler.
- Change target selection, retries, distributed ownership or result assembly
  in Runtime.
- Change tensor math, precision kernels or simulator behavior in Simulation.
- Add a directly controlled accelerator integration in Compute.
- Add a QPU, remote GPU/HPC service, or cloud integration in Remote.
- Add support for an external framework or format in Ecosystem.
- Add a reusable multi-step preflight workflow in Services; call stable APIs
  directly when no orchestration is needed.

An ordinary feature should normally change one primary domain. If it repeatedly
needs edits across four or more domains, stop and review the boundary instead of
adding pass-through objects.

## Required checks

Run focused scenario tests for the changed behavior, then run:

```bash
python tools/check_architecture.py
python tools/check_dependency_policy.py
PYTHONPATH=. python tools/public_api_snapshot.py
```

Public API and serialized-contract changes require the process in
[`docs/development/PUBLIC_API_PROTECTION.md`](../docs/development/PUBLIC_API_PROTECTION.md).
