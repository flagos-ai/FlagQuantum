# Explicit Qiskit Aer execution

FlagQuantum can execute a FlagQuantum circuit on local Qiskit Aer without
changing the native runtime default. Install the existing optional dependency
and select the bridge explicitly:

```bash
pip install "flagquantum[qiskit]"
```

```python
import flagquantum as fq
from flagquantum.ecosystem.qiskit import run

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
expressions used by native `fq.run`; selecting the framework changes only the
imported `run` implementation.

Every result is a FlagQuantum `ExecutionResult`. The `runtime`, `provenance`,
and `compatibility` mappings identify Qiskit Aer, the Qiskit and Aer versions,
the source IR hash, the conversion report, the CPU thread policy, and that no
fallback occurred. Qiskit circuits, jobs, and result objects do not cross the
plugin boundary.

The same implementation is discoverable as the experimental
`backend.qiskit_aer` extension. Direct `run` is the shortest user journey;
extension discovery is intended for higher-level backend selection code.

## Initial support boundary

- local CPU statevector, computational-basis samples, and counts;
- one fully bound, unbatched FlagQuantum circuit;
- `complex64` and `complex128` statevectors;
- explicit ordered wire selection for samples and counts;
- no gradients, noise model, dynamic circuit, automatic routing, or fallback.

Unsupported requests fail before Aer execution. Native `fq.run` remains the
default and does not consult this plugin implicitly.
