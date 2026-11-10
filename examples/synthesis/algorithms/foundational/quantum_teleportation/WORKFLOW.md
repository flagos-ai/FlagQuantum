# Teleportation with explicitly deferred private measurements

Run from the repository root:

```sh
python -m examples.synthesis.algorithms.foundational.quantum_teleportation.main
python -m examples.synthesis.algorithms.foundational.quantum_teleportation.comparisons.circuit
```

The frozen program measures Alice's two qubits, uses the private outcomes for
Pauli corrections on Bob, and drops Alice's qubits. Because those outcomes never
escape, the same output channel can be implemented by coherent corrections and
retaining the outcome qubits until they are traced out. This native example uses
that deferred-measurement implementation explicitly. It performs no physical
mid-circuit measurement and sends no classical messages. It is not a generic
measurement or reset implementation.

```python
import flagquantum as fq
from examples.synthesis.algorithms.foundational.quantum_teleportation.main import build_program

program = build_program(theta=0.73, phi=-0.41)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("bob_qubit",))))
```

The input state is `cos(theta/2)|0> + exp(i*phi)sin(theta/2)|1>`. Bob receives that
state, including its relative complex phase. The default input is the original
`|+>` state. The two dropped Alice qubits remain a quantum environment; they are
not reset, freed or reported as clean work.

The independent ideal-channel Circuit prepares Bob directly and prepares the
retained outcome qubits separately. Additional acceptance explicitly enumerates
all actual measurement branches and applies classical X/Z corrections, then
compares the resulting density matrix. A Bell input entangled with an external
reference verifies preservation of entanglement rather than only basis-state
probabilities. Full absolute purified states, nonzero complex inputs, branch
probabilities and every work amplitude are checked.
