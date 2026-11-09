# Declare an exponential probability distribution

The native example computes the normalized classical probabilities proportional
to exp(-0.1 * value), then prepares their positive real square-root amplitudes.
The bound probability array is explicit high-level input to the captured program.

```python
import flagquantum as fq
from examples.synthesis.functions.state_preparation.prepare_exponential_state.main import build_program

program = build_program()
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x",))))
print(program.report["resources"])
```

`prepare_state` initializes its fresh Output directly; it does not require a
preceding allocate. This example matches the frozen original's fresh five-bit
state. It does not promise the original exponential rotation's unitary action
on arbitrary nonzero or entangled existing input.

An independent Circuit builds a Householder reflection from a separately
computed softmax distribution. Original and native complete complex amplitudes,
absolute phase, bit order and all 32 probabilities are checked within 1e-12,
with hidden-work probability below 1e-24 and physical budgets enforced.

The general native preparation currently uses five qubits, zero work bits,
56 operations and depth 52; the original product preparation uses five parallel
RY operations and depth one. Detecting separable probability distributions is
a general compiler optimization opportunity. No equal-resource claim is made.
The accepted scope is the quantum program.
