# Explicit Cirq Simulator execution

FlagQuantum can execute a FlagQuantum circuit on local Cirq Simulator without
changing the native runtime default. Install the existing optional dependency
and select the bridge explicitly:

```bash
pip install "flagquantum[cirq]"
```

```python
import flagquantum as fq
from flagquantum.ecosystem.cirq import run

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
expressions used by native `fq.run` and the Qiskit Aer bridge. Selecting the
framework changes only the imported `run` implementation.

Every result is a FlagQuantum `ExecutionResult`. The `runtime`, `provenance`,
and `compatibility` mappings identify Cirq Simulator, the Cirq version, the
source IR hash, the conversion report, and that no fallback occurred. Cirq
circuits and result objects do not cross the ecosystem boundary.

Cirq circuits do not retain idle qubits. The bridge supplies the complete
FlagQuantum wire order explicitly to Cirq so that statevector dimensions and
wire positions remain unchanged; this preservation is recorded in provenance.

## Initial support boundary

- local CPU statevector, computational-basis samples, and counts;
- one fully bound, unbatched FlagQuantum circuit;
- `complex64` and `complex128` statevectors;
- explicit ordered wire selection for samples and counts;
- no gradients, noise model, dynamic circuit, automatic routing, device
  selection, or fallback.

Unsupported requests fail before simulation. Native `fq.run` remains the
default and does not consult this bridge implicitly.
