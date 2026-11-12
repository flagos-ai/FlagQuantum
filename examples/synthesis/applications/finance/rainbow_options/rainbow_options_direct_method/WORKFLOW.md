# Direct rainbow payoff: quantum program

This native example retains the original finite model. Users supply classical
iteration and exponent-rate values; qfunc expresses state preparation, exact
payoff arithmetic and controlled rotations. Synthesis owns gate design, temporary
registers, inverse generation and physical placement.

```python
import flagquantum as fq
from examples.synthesis.applications.finance.rainbow_options.rainbow_options_direct_method.main import build_program

program = build_program(k=1)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("indicator",))))
print(program.report["resources"])
```

`physics.py` contains the source model constants and classical rotation angles.
`operations.py` contains reusable preparation, exponential loading and reflection.
The affine maximum remains a quantum expression inside within/apply; no host
truth table replaces the program. The score has five bits and two fractional
bits, declared explicitly so accidental truncation is rejected. Classical angle
calculation happens before quantum capture.

`main.py` retains the complete amplitude-amplification loop. Dropping the private
asset and auxiliary registers removes them from named readout; it does not erase
their entanglement or turn them into zero workspace. Only the score and synthesis
scratch are uncomputed to zero. The sole public output is the indicator.

The independent Circuit builds branch amplitudes from exact rational payoffs,
then uses a dense preparation/reflection reference for the ten model qubits.
Acceptance compares every complex amplitude including absolute phase, all private
environment and zero scratch, not just success probability. Tests also replay
the unchanged original source at k=0/1/2, check two further positive exponent
rates, and reject insufficient width/work/operation/memory budgets.

The native k=0/1/2 resource measurements are recorded independently of the
original importer baseline. The related financial estimation Notebook workflow
has not been migrated or certified by this quantum-program example.
