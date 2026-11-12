# Native Simon oracle composition

The retained quantum wrapper prepares five input bits, evaluates
`min(x, x ^ 6)` into a fresh result, then applies Hadamard to the input again.
The result register stays part of the joint quantum state.

```python
import flagquantum as fq
from examples.synthesis.algorithms.foundational.simon.main import build_program

program = build_program()
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x", "res"))))
print(program.report["resources"])
```

The original's concrete oracle callback is a named native `Const`/`Output`
helper. The explicit H/oracle/H sequence preserves its within/apply behavior.
This specialization adds no general quantum callable parameters. The user writes
arithmetic and preparation without choosing gates, work registers or wire numbers.

An independent Circuit uses a classical reversible XOR table for all input and
result values, surrounded by explicit Hadamards. Both the exact original and
native program match all complex joint amplitudes within 1e-12, including absolute
phase and register order, with hidden work probability below 1e-24. All 32 inputs
have the retained two-to-one relation for secret 6. Actual qubit, operation, depth
and memory ceilings refuse before execution.

Both programs use ten qubits, zero work bits, fifteen operations and depth four.
This is the original quantum wrapper only. Classical secret recovery and the
complete associated Notebook workflow remain pending.
