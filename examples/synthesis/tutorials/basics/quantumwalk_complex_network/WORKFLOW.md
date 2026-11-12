# Nonuniform coined walk on a complex network

Run from the repository root:

```sh
python -m examples.synthesis.tutorials.basics.quantumwalk_complex_network.main
python -m examples.synthesis.tutorials.basics.quantumwalk_complex_network.comparisons.circuit
```

The program prepares a position register in uniform superposition, then prepares
a different coin state for each position. Each of three rounds applies the
conditional reflection `2|s_x><s_x|-I` and swaps the two logical registers.
The four probability vectors preserve the frozen quantum program. `within/apply`
with inverse preparation expresses reflection using existing high-level
operations. No physical wire numbers or handwritten preparation gates appear in
`main.py`. QArray indices identify logical elements; a helper borrows the complete
array as a raw unsigned QNum for comparisons.

```python
import flagquantum as fq
from examples.synthesis.tutorials.basics.quantumwalk_complex_network.main import build_program

program = build_program()
result = fq.run(program)
print(dict(program.probabilities(result, registers=("x",))))
print(program.report["resources"])
```

`build_program(steps=2, angle=0.37)` additionally permits bounded teaching variants.
The angle applies relative phases to position and coin after preparation. These
variants are independently verified and do not change the frozen source.

The coin is dropped from the public output, but remains a potentially entangled
quantum environment. Dropping it does not reset it or reclaim it as clean work.
Acceptance compares all sixteen position-and-coin complex amplitudes, including
absolute global phase, and independently checks every work amplitude is zero.
The reference Circuit uses a Householder map only to prepare the initial state,
then explicit block-diagonal `2|s_x><s_x|-I` matrices and swaps. It does not call
synthesis preparation or controlled-effect expansion.

Tests cover rounds0..3, phase values0/0.37/-0.61, both control polarities, arbitrary
complex one-qubit target states, conditional global-phase interference and inverse
preparation. Alias, Const, nested Output escape, allocation and budget refusals
are checked. Measured feedforward, ownership-changing callbacks and arbitrary
scratch-producing callbacks are not admitted by this profile.
