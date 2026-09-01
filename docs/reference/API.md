# FlagQuantum API Reference

FlagQuantum exposes one curated Python interface:
`import flagquantum as fq`. Build a circuit, inspect its runtime plan, execute
it through a stable result contract, and train parameterized programs with
PyTorch.

Exact stable names are defined by
[`public_api_v1.json`](../public_api_v1.json), verified by executable contract
tests, and rendered in the
[stable API inventory](../generated/STABLE_API.md).

## API map

| Task | Primary interface | Result |
| --- | --- | --- |
| Build a program | `fq.Circuit` | Circuit backed by FlagQuantum IR |
| Inspect execution | `fq.plan`, `Circuit.runtime_plan` | Explainable runtime plan |
| Execute | `fq.run` | `fq.ExecutionResult` |
| Define a trainable quantum layer | `fq.Module` | PyTorch module |
| Train | `fq.train` | `fq.TrainingResult` |
| Package for a target | `flagquantum.deployment.create_deployment_package` | Sealed deployment package |

## Build and execute

```python
import flagquantum as fq

circuit = fq.Circuit(n_qubits=2).h(0).cx(0, 1)
options = fq.ExecutionOptions(mode="auto", precision="complex64")
plan = fq.plan(circuit, options=options)
result = fq.run(plan)

print(plan.identity)
print(plan.summary()["recommended_mode"])
print(result.plan.identity)
print(result.state)
```

`n_qubits` is the preferred public name for circuit size. Positional
`Circuit(2)`, `n_wires=2`, and the legacy `nqubits=2` remain compatible;
conflicting aliases fail during construction. Runtime, compiler, and IR
internals continue to use *wire* for logical mappings.

Generated gate methods keep their concise positional form and also accept
semantic qubit keywords. For example, `h(0)` and `h(qubit=0)` are equivalent;
`cx(0, 1)` and `cx(control=0, target=1)` are equivalent. Symmetric two-qubit
gates use `qubit1=` and `qubit2=`, while the generic `Circuit.gate(...)` and
FlagQuantum IR continue to use `wires=`.

`fq.run(...) -> fq.ExecutionResult` is the single recommended execution entry
point. `ExecutionOptions` owns backend-neutral execution configuration;
measurement requests and an optional `flagquantum.noise.NoiseModel` are semantic
program inputs. Unknown keywords fail before planning. `Circuit.run(...)` and
`fq.run(circuit, ...)` are equivalent.

For an inspectable and reproducible execution, pass the result of `fq.plan`
directly to `fq.run`. The supplied plan is validated and executed without
replanning or recompiling, and `result.plan is plan` holds in the same process.
Plan identity covers the canonical IR, resolved execution semantics, compiler
pipeline, required environment, and selected decision. JSON round trips verify
all fingerprints and the final SHA-256 identity before execution:

```python
text = plan.to_json()
restored = fq.ExecutionPlan.from_json(text)
result = fq.run(restored)
assert result.plan.identity == plan.identity
```

An existing plan is closed to semantic overrides: passing `options`,
`measurements`, or `noise_model` alongside it raises `TypeError`. Environment
or world-size incompatibility fails before kernel launch rather than silently
replanning or falling back. Provider submission and signed portability remain
the responsibility of `DeploymentPackage`.
`flagquantum.backends.run_native`, `flagquantum.backends.run_mps`, and
`flagquantum.backends.run_tensor_network` are advanced interfaces for callers
that explicitly need native backend result objects or backend-specific controls.

## Train with PyTorch

Execution and training are intentionally separate. A complete trainable program
looks like ordinary PyTorch code:

```python
import flagquantum as fq
import torch

def build_circuit(parameters, inputs=None):
    return (
        fq.Circuit(n_qubits=2)
        .ry(0, theta=parameters[0])
        .cx(0, 1)
        .ry(1, theta=parameters[1])
    )

module = fq.Module(
    build_circuit,
    n_parameters=2,
    policy=fq.RuntimePolicy(observable_wires=(1,)),
)
optimizer = torch.optim.Adam(module.parameters(), lr=0.01)

training = fq.train(
    module,
    optimizer=optimizer,
    objective=lambda value: value.mean(),
    steps=100,
)

print(training.losses[-1])
```

`module(inputs)` and `module.forward(inputs)` always return an autograd-compatible
Tensor. Use `module.execute(inputs)` when the caller needs an
`ExecutionResult`, provenance, runtime diagnostics, or explicit backend
compatibility information. `fq.run` accepts Circuit, IR, or ExecutionPlan—not
Module—and never updates parameters.

`fq.train` is intentionally a minimal, caller-owned PyTorch optimizer loop. It
performs `zero_grad`, `backward`, and `step`, then returns `fq.TrainingResult`.
Checkpoint and resume belong to `Module.save_checkpoint()` and
`Module.load_checkpoint()` or to an application-owned training loop; they are
not hidden options of `fq.train`. Training lifecycle types such as
`PrecisionPolicy`, `SeedContract`, and checkpoint errors live in the stable
`flagquantum.training` namespace. Distributed training remains explicitly
experimental under `flagquantum.experimental.distributed`.

`ExecutionResult.require_value()` returns the Module value or fails clearly.
`ExecutionResult.diagnostics()` returns a versioned envelope containing
`metrics`, `provenance`, `runtime`, and `compatibility`; keys inside those four
sections may grow compatibly. `TrainingResult.final_loss` and its versioned
`summary()` provide stable training output access.

The [examples index](../../examples/README.md) provides runnable statevector,
MPS, JAX, distributed, and deployment workflows.

## Errors

Catch stable lifecycle categories from `flagquantum.errors`:

```python
import flagquantum.errors as fqe

try:
    result = fq.run(fq.plan(circuit, options=options))
except fqe.ValidationError:
    ...  # invalid semantic input
except fqe.PlanningError:
    ...  # stale, tampered, or incompatible plan
except fqe.CapabilityError:
    ...  # requested capability is unavailable
except fqe.ExecutionError:
    ...  # execution or training failure
```

All categories inherit `FlagQuantumError` and their compatible Python built-in
exception (`ValueError`, `RuntimeError`, or `NotImplementedError`). Wrong Python
types and unknown keyword arguments continue to raise `TypeError`. Existing
specific errors such as `IRValidationError`, `IRSerializationError`, and
`flagquantum.training.TrainingStateError` remain available and now belong to
the corresponding stable category.

## Measurements

Pass ordered `MeasurementNode` requests to `fq.plan` or `fq.run`. They are
embedded in the executable plan before identity is computed. Requests may
instead already exist in `CircuitIR.measurements`, but supplying both forms is
an error: FlagQuantum never silently replaces or appends measurements.

```python
requests = (
    fq.MeasurementNode("expectation_z", (0, 1)),
    fq.MeasurementNode(
        "expectation_ps",
        (0, 1),
        metadata={"x": (0,), "z": (1,)},
    ),
    fq.MeasurementNode("sample", (0, 1), shots=1024, metadata={"seed": 7}),
)
plan = fq.plan(circuit, measurements=requests)
result = fq.run(plan)

z_values = result.expectation(0)
xz_value = result.expectation(1)
bit_samples = result.require_samples()
```

Supported measurement kinds are `expectation_z`, `expectation_ps`,
`probabilities`, `sample`, and `counts`. Sampling and counts are in the
computational basis. A `sample` request is also projected to
`ExecutionResult.samples` for consumers of the original result contract.
Unsupported measurement kinds and missing shot counts fail explicitly.

Use `result.measurement(index_or_name)` for a specific request,
`result.statevector()` for a required statevector, and `result.native()` only
when intentionally depending on an unstable backend-native object. Backend
attributes are not implicitly forwarded through `ExecutionResult`.

## Noise

Stable noisy execution accepts a `flagquantum.noise.NoiseModel` during planning:

```python
import flagquantum.noise as fqn

noise = fqn.NoiseModel().add("x", fqn.bit_flip_channel(0.01))
plan = fq.plan(circuit, noise_model=noise)
restored = fq.ExecutionPlan.from_json(plan.to_json())
result = fq.run(restored)
```

The versioned model payload and its SHA-256 identity are verified as part of
the plan. The first public alpha candidate supports stable noisy planning for
`mode="auto"` and `mode="density_matrix"`; unsupported mode combinations fail
during planning.

`probabilities` computes an exact joint marginal over the requested wires from
Pauli-Z contractions. The default limit is eight wires because the cost is
`2**len(wires)` contractions; callers must set `max_marginal_wires` explicitly
to accept a larger exponential calculation.

Sampling and counts support bounded postselection without materializing a
dense state:

```python
request = fq.MeasurementNode(
    "counts",
    (1,),
    shots=1024,
    metadata={
        "seed": 7,
        "postselect": {0: 1},
        "max_postselection_draw_multiplier": 1024,
    },
)
result = fq.run(
    circuit,
    options=fq.ExecutionOptions(mode="mps"),
    measurements=(request,),
)
print(result.measurements[0].statistics["acceptance_rate"])
```

Shot-based results report bit probabilities, binomial standard errors, draw
counts, and postselection acceptance rates in `MeasurementResult.statistics`.
If the requested number of conditioned samples cannot be retained within the
explicit draw bound, execution fails instead of returning a biased or
short-count result.

## Hardware Pauli measurements

Use `create_pauli_measurement_plan` to measure a Hamiltonian containing X, Y,
and Z terms on shot-based hardware. It greedily groups qubit-wise-commuting
terms, appends the required basis rotations, and creates one sealed deployment
package per group:

```python
import flagquantum.deployment as fqd

plan = fqd.create_pauli_measurement_plan(
    circuit,
    hamiltonian,
    backend=backend,
    shots=4096,
)

results = tuple(provider.run(package) for package in plan.packages)
energy = plan.expectation(tuple(result.counts for result in results))
```

X measurements append H; Y measurements append S-dagger followed by H.
Aggregation validates the number of groups, shot totals, bitstring widths, and
real Hamiltonian coefficients before producing an expectation value. Each
package records its group index, term indices, basis, routing evidence, and
sealed deployment identity.

### Amazon Braket IQM dynamic submission

The experimental IQM deployment path derives qubit count, native gates,
connectivity and the dynamic dialect from an `AwsDevice`. If dynamic qubit
groups are not present in the device-property schema, provide the current
groups published for that device. Install the optional SDK integration with
`pip install -e '.[braket]'`.

```python
from flagquantum.deployment import AmazonBraketProvider

provider = AmazonBraketProvider(
    device_arn,
    dynamic_qubit_groups=published_groups_for_device,
)
preview = provider.dry_run(package)
assert preview.compatible, preview.blockers
print(preview.program)
result = provider.run(package)  # Creates a Braket quantum task.
```

`dry_run()` validates the sealed package, device ARN, OpenQASM version, IQM
dialect and dynamic groups without calling `AwsDevice.run()`. See
[`braket_iqm_dynamic_preflight.py`](../../examples/braket_iqm_dynamic_preflight.py)
for the complete guarded-submission example. This integration has been
validated locally through the real Braket SDK serializer and mocked task
contract only; no real IQM QPU execution is claimed.

### Provider-neutral dynamic conformance

The experimental conformance API makes backend differences explicit and runs
the same deterministic active-reset, conditional-flip, and qubit-reuse vectors
against multiple implementations:

```python
local = fq.experimental.dynamic.run_dynamic_conformance()
aer = fq.experimental.dynamic.run_dynamic_conformance(
    fq.experimental.dynamic.run_qiskit_aer_dynamic,
    implementation="qiskit_aer",
)
assert local.passed and aer.passed
```

Install the second local backend with `pip install -e '.[qiskit]'`.
`assess_dynamic_features()` can compare a circuit against the declared local,
Qiskit Aer, or Braket IQM feature set before transport-specific validation.
`DynamicExecutionResult` exposes `final_samples`, `classical_register`,
`mid_circuit_measurements`, availability metadata, and
`to_execution_result()` for projection into the canonical result contract.
Local trajectory results also expose a `statistics` mapping with trajectory,
measurement, reset, conditional-branch, observed-branch and elapsed-time
counters. The same mapping is retained as `runtime["dynamic_statistics"]` by
the canonical projection. `run_dynamic(..., strategy="auto")` uses batched
statevector trajectories for eligible workloads of at least 32 shots and
falls back to the reference trajectory path when batching would exceed
`max_batched_bytes` (256 MiB by default) or the input is already batched.
Callers may explicitly request `strategy="trajectory"` or `"batched"`.
`statistics["gate_execution_strategy"]` records the selected path for
benchmark attribution.
The Qiskit path also validates a full `DynamicCircuit → OpenQASM 3 → Qiskit →
Aer` round trip and statistical agreement for random measurement branches.

Use the development microbenchmark to measure shot and mid-circuit-measurement
scaling locally. Add `--backend qiskit_aer` after installing the optional
Qiskit dependencies for a same-workload comparison:

```bash
python benchmarks/dynamic_trajectory.py \
  --shots 100 1000 --mid-circuit-measurements 1 2 4 \
  --json-output benchmarks/results/smoke/dynamic-trajectory.json
```

Use `--flagquantum-strategy trajectory` and `batched` in separate runs for a
direct reference-versus-vectorized comparison.

This payload is development evidence only and does not support scalability or
provider-performance claims.

Optional integration suites can be selected independently:

```bash
pytest -m qiskit
pytest -m pennylane
pytest -m braket
```

### Interoperability adapter contract

External framework adapters implement one candidate-stable, framework-neutral
protocol under `flagquantum.interop`. The default registry stores import-safe
descriptors and loads an adapter implementation only when requested:

```python
from flagquantum.interop import available_adapters, get_adapter

assert available_adapters() == ("pennylane", "qiskit")
adapter = get_adapter("qiskit")
result = adapter.import_program(external_circuit)
flagquantum_ir = result.ir
```

`InteropConversionIssue`, `InteropConversionReport`, `InteropImportResult`,
and `InteropExportResult` define the common diagnostics boundary. Registries
are immutable: adding a descriptor returns a new registry and cannot alter the
process-wide default. Resolving the Qiskit descriptor imports no Qiskit module;
the external dependency is loaded only when conversion is requested. Adapter
API mismatches and registered/loaded identity mismatches fail before use.
`InteropRegistry.to_dict()` provides a machine-readable inventory for tooling
and review without probing or importing dependencies. The default registry
instance and each Qiskit/PennyLane adapter remain experimental implementation
details; they are not part of the candidate-stable export list.

### PennyLane QuantumScript interoperability

PennyLane is an optional control-plane adapter and is never a FlagQuantum
runtime dependency. On Python 3.11 or newer, install it with
`pip install 'flagquantum[pennylane]'` and convert only at the immutable
`QuantumScript` boundary:

```python
from flagquantum.interop.pennylane import from_pennylane, to_pennylane

ir = from_pennylane(quantum_script)
round_trip = to_pennylane(ir)
```

The v1 adapter is intentionally static and complex128-first. It supports the
gate map recorded in `contracts/pennylane-interop-contract.toml`, bound real scalar
parameters, and contiguous integer wires. QNodes, devices, execution, shots,
measurement processes, autograd bridges, and symbolic parameters remain out of
scope and fail closed. Nonstandard wire labels can only be flattened with an
explicit `allow_lossy=True` report. PennyLane objects do not cross into the
compiler, PyTorch runtime, Torch-FL, CUDA, vendor accelerator, or QPU layers.

Adapter authors use `InteropRoundTripCase`, `InteropRejectionCase`, and
`run_adapter_conformance()` to apply the same framework-neutral identity,
lossless round-trip, fail-closed, and explicit-loss checks to every adapter.
The returned `InteropConformanceResult.to_dict()` payload uses the versioned
`flagquantum_interop_conformance_v1` schema. See
[Interoperability adapter development](../development/INTEROP_ADAPTERS.md) for
the required adapter layout and evidence boundary.

### Qiskit IR interoperability

Qiskit is an optional control-plane adapter, not a FlagQuantum runtime
dependency. Install it with `pip install 'flagquantum[qiskit]'`, then convert at
the versioned IR boundary:

```python
from flagquantum.interop.qiskit import from_qiskit, to_qiskit

ir = from_qiskit(qiskit_circuit)
round_trip = to_qiskit(ir)
```

These Qiskit-specific functions and result types implement the common adapter
protocol rather than defining a parallel framework architecture. They remain
experimental and have their own tested dependency-version window.

Both directions fail closed when an operation, control-flow construct, or
parameter expression cannot be represented losslessly. Use `import_qiskit()`
or `export_qiskit()` to receive the converted object together with a
machine-readable `QiskitConversionReport`. `allow_lossy=True` must be explicit
and records every skipped operation; it is intended for inspection, not silent
execution fallback. Importing `flagquantum` or `flagquantum.interop.qiskit`
does not import Qiskit.

The supported bidirectional gate set, parameter names, bit-index mapping,
statevector endianness, loss policy, unsupported boundary, and certified
Qiskit/Aer version lanes are pinned in `contracts/qiskit-interop-contract.toml`. Run the
same deterministic semantic certification used by CI when qualifying a new
environment:

```python
from flagquantum.interop.qiskit import run_qiskit_conformance

result = run_qiskit_conformance()
assert result.passed
```

The implementation is partitioned under `flagquantum.runtime.dynamic` into
`circuit`, `result`, `execution`, `routing`, `deployment`, and `dialects`
boundaries. Existing `fq.experimental` names and the
`flagquantum.runtime.dynamic` import path remain compatible.

The stable surface also includes circuit and IR construction, backend
compilation, runtime planning, and deployment helpers. The generated table is
authoritative for exact names.

## Stability boundaries

- `fq.experimental` has no compatibility guarantee.
- Compatibility imports are migration aids and are not implied stable.
- A planner result describes intent and estimates; it is never runtime or
  benchmark evidence.
- Operator/backend support comes from the executable lowering registry in the
  [generated capability table](../generated/OPERATOR_CAPABILITIES.md).
- Runtime evidence must satisfy the [typed contracts](RUNTIME_CONTRACTS.md).

Examples in stable documentation are executed by documentation contract tests.
New public names must first be importable, snapshot-tested, and added to the
stable API manifest.
