# Equal superposition

A reusable quantum function prepares all eight three-bit values with equal
amplitude. `allocate` supplies zero-initialized storage; `hadamard_transform`
expresses the preparation without physical wire numbers.

```python
from flagquantum.synth import Output, QArray, QBit, allocate, hadamard_transform, qfunc

@qfunc
def prepare_uniform(register: QArray[QBit]) -> None:
    hadamard_transform(register)

@qfunc
def main(register_a: Output[QArray[QBit]]) -> None:
    allocate(3, register_a)
    prepare_uniform(register_a)
```

Run from the repository root:

```sh
python -m examples.synthesis.community.basic_examples.superposition.main
python -m examples.synthesis.community.basic_examples.superposition.comparisons.circuit
```

The first command constructs the program, prints its native text circuit diagram
and reads named probabilities. The result is eight outcomes of probability 1/8;
every complex amplitude is +1/sqrt(8), with no work qubits.

This implements the complete computational workflow of the pinned upstream
Notebook: a composed three-bit preparation, synthesis and visualization. The
FlagQuantum text drawer replaces the upstream cloud viewer. The independent
Circuit comparison also checks the analytical state, absolute phase and named
readout. No upstream source file is loaded at runtime.

The [provenance record](provenance.json) identifies the reviewed Notebook and its
code cell. This Notebook has no corresponding file in the 216-source Qmod corpus
and is counted separately from native quantum-program counterparts.
