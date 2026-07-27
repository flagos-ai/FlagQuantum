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
