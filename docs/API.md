# FlagQuantum API Reference

The stable package boundary is defined by [`public_api_v1.json`](public_api_v1.json),
verified by an import/snapshot test, and rendered in
[`generated/STABLE_API.md`](generated/STABLE_API.md). An object merely reachable
through a compatibility module is not a stable API.

## Quick start

```python
import flagquantum as fq

circuit = fq.Circuit(n_qubits=2)
circuit.h(0)
circuit.cx(0, 1)
result = fq.run(circuit)

print(result.state)
print(result.plan)
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
point. It provides the same stable result contract for local, MPS,
tensor-network, and distributed modes. `Circuit.run(...)` is a compatibility
shortcut whose default return remains backend-native. `fq.run_native`,
`fq.run_mps`, and `fq.run_tensor_network` are advanced interfaces for callers
that explicitly need native backend result objects.

Training is intentionally separate from execution:

```python
module = fq.Module(build_circuit, n_parameters=2)
optimizer = torch.optim.Adam(module.parameters(), lr=0.01)

training = fq.train(
    module,
    optimizer=optimizer,
    objective=lambda value: value.mean(),
    steps=100,
)
```

`fq.run` performs one forward execution and never updates parameters.
`fq.train` owns the PyTorch optimizer loop (`zero_grad`, `backward`, and
`step`) and returns `fq.TrainingResult`. The specialized
`train_distributed_statevector` and `train_distributed_mps` functions remain
advanced interfaces for explicit distributed lifecycle and evidence control.

## Measurements

Pass ordered `MeasurementNode` requests to `fq.run`. The same requests can be
stored in `CircuitIR.measurements`; executors consume them without a separate
measurement API:

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
result = fq.run(circuit, measurements=requests)

z_values = result.measurements[0].value
xz_value = result.measurements[1].value
bit_samples = result.measurements[2].value
```

Supported measurement kinds are `expectation_z`, `expectation_ps`,
`probabilities`, `sample`, and `counts`. Sampling and counts are in the
computational basis. A `sample` request is also projected to
`ExecutionResult.samples` for consumers of the original result contract.
Unsupported measurement kinds and missing shot counts fail explicitly.

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
result = fq.run(circuit, mode="mps", measurements=(request,))
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
plan = fq.create_pauli_measurement_plan(
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
[`braket_iqm_dynamic_preflight.py`](../examples/braket_iqm_dynamic_preflight.py)
for the complete guarded-submission example. This integration has been
validated locally through the real Braket SDK serializer and mocked task
contract only; no real IQM QPU execution is claimed.

The stable surface also includes circuit and IR construction, backend
compilation, runtime planning, and deployment helpers. The generated table is
authoritative for exact names.

## Stability boundaries

- `fq.experimental` has no compatibility guarantee.
- Compatibility imports are migration aids and are not implied stable.
- A planner result describes intent and estimates; it is never runtime or
  benchmark evidence.
- Operator/backend support comes from the executable lowering registry in the
  [generated capability table](generated/OPERATOR_CAPABILITIES.md).
- Runtime evidence must satisfy the [typed contracts](RUNTIME_CONTRACTS.md).

Examples in stable documentation are executed by documentation contract tests.
New public names must first be importable, snapshot-tested, and added to the
stable API manifest.
