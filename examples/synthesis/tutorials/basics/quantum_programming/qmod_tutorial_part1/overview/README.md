# Complete native quantum programming tutorial

Run every solution, independent Circuit comparison and expected ownership error:

```sh
python -m examples.synthesis.tutorials.basics.quantum_programming.qmod_tutorial_part1.overview.main
```

Open `walkthrough.ipynb` from the repository root for all 22 original code stages.
The Notebook includes executable native qfunc definitions. TODO exercises are
labeled reference solutions; their subsequent solutions repeat the same teaching
steps, rather than establishing additional algorithm coverage.

The workflow preserves the original Bell/GHZ preparation, signed three-bit view,
eight-entry probability distribution, 10,000 actual draws for each probability
preparation stage, full-turn rotations, classical parameter values, two-bit
sum/product, accumulator initially four and coherent equality control. The
unallocated borrowed-port example remains an expected synthesis refusal; the
correction makes the initializer own its Output.

Classical alpha/beta values bind during synthesis. Re-synthesize for a different
binding. `QArray[QBit]` uses `allocate(size, target)` instead of a second generic
array argument. Array index zero denotes the low bit. Named distribution readers
handle register decoding; `numeric_probabilities` gives signed numeric values.

The independent Circuit references check complete complex states in declared
register order without phase alignment and require clean hidden compiler storage.
The arithmetic reference enumerates the mathematical truth table into gates;
preparation references use a mathematical normalized vector. Resource estimates
are actual synthesized instruction counts before Runtime compilation, with no
claim of optimality or equality to Classiq resources.

The frozen sibling quantum counterpart is a different polynomial-phase program;
this complete Notebook is a separate source artifact. Provenance pins every
original teaching cell. No Qmod importer, Classiq SDK, authentication or remote
hardware submission is needed.

[Workflow mapping](WORKFLOW.md) · [Third-party license](../../../../../CLASSIQ_LICENSE.txt)

Local acceptance checks all 20 executable solution states (including repeated
exercise solutions) with maximum absolute amplitude error below `6.2e-16` and
zero dirty compiler workspace. The direct accumulator example synthesizes to
six physical qubits, ten instructions and depth seven; one qubit is clean MCX
workspace. The independent sum/product example uses 13 physical qubits,
26 instructions and depth 20. Counts are before Runtime compilation. Focused
and adjacent regression tests pass; the full default-tier result and its
existing failures are recorded separately in the delivery ledger.

The accumulator is written directly as a native quantum program:

```python
import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QNum, allocate, hadamard_transform, qfunc


@qfunc
def main(y: Output[QNum], acc: Output[QNum]) -> None:
    allocate(2, y)
    hadamard_transform(y)
    acc |= 4
    acc += y


program = synth.synthesize(main)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("y", "acc"))))
# {(0, 4): 0.25, (1, 5): 0.25, (2, 6): 0.25, (3, 7): 0.25}
```
