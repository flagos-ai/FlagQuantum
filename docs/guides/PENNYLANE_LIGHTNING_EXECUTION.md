# Explicit PennyLane Lightning execution

FlagQuantum can execute a FlagQuantum circuit on local PennyLane
`lightning.qubit` without changing the native runtime default. Install the
existing optional dependency and select the bridge explicitly:

```bash
pip install "flagquantum[pennylane]"
```

```python
import flagquantum as fq
from flagquantum.ecosystem.pennylane import run

circuit = fq.Circuit(3).h(0).cx(0, 1).cx(1, 2)
shot_options = fq.ExecutionOptions(shots=1_000, seed=7)

state = run(circuit)
samples = run(
    circuit,
    outputs=fq.samples(qubits=(2, 1, 0)),
    options=shot_options,
)
counts = run(
    circuit,
    outputs=fq.counts(qubits=(2, 1, 0)),
    options=shot_options,
)
```

These are the same `outputs`, `ExecutionOptions`, `samples`, and `counts`
expressions used by native `fq.run`, Qiskit Aer, and Cirq Simulator. Selecting
the framework changes only the imported `run` implementation.

Every result is a FlagQuantum `ExecutionResult`. The `runtime`, `provenance`,
and `compatibility` mappings identify PennyLane Lightning, both dependency
versions, the source IR hash, the conversion report, and that no fallback
occurred. PennyLane scripts, devices, and native result objects do not cross the
ecosystem boundary.

QuantumScript does not retain idle wires. The bridge constructs Lightning with
the complete FlagQuantum wire range so statevector dimensions and wire
positions remain unchanged; this preservation is recorded in provenance.

## Initial support boundary

- local CPU `lightning.qubit` statevector, computational-basis samples, and
  counts;
- one fully bound, unbatched FlagQuantum circuit;
- `complex64` and `complex128` statevectors;
- explicit ordered wire selection for samples and counts;
- no gradients, QNodes, noise model, dynamic circuit, automatic routing, GPU,
  alternative PennyLane device, or fallback.

Unsupported requests fail before device execution. Native `fq.run` remains the
default and does not consult this bridge implicitly. A missing Lightning plugin
is an explicit dependency error; the bridge never switches to `default.qubit`.
