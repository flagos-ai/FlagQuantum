# Integration Rainbow payoff: quantum program

Users write source preparation, exact payoff arithmetic, controlled probability
integration and amplitude amplification as composable Python quantum functions.
Synthesis owns the gate sequence, physical placement, inverses and clean scratch.

```python
import flagquantum as fq
from examples.synthesis.applications.finance.rainbow_options.rainbow_options_integration_method.main import build_program

program = build_program(k=1)
result = fq.run(program)
print(dict(program.probabilities(
    result, registers=("indicator",), max_amplitudes=2**26,
)))
print(program.report["resources"])
```

`physics.py` contains the original finite-model constants and checked classical
rotation parameters. `operations.py` expresses the affine maximum as a quantum
expression in within/apply. The explicit five-bit unsigned word with two
fractional bits retains the comparison's exact numeric meaning.

For scores above 1.5 the program prepares all 32 reference values with their
normalized exponential weights, then XORs `score >= reference` into the
indicator. Equality is included. The other branch rotates the indicator by the
original low-payoff angle. `reflect` reverses the complete preparation and
reflects the original ten model bits around zero before preparing again. A
temporary conjunction divides the wide zero predicate into bounded pieces;
its own computation is reversed. The scalar phase retains the exact reflection
sign, including in subsequent coherent composition.

`drop` removes the two source words and reference word from named readout. Those
bits remain physical environment, potentially entangled with the indicator.
Only computed scores and synthesis scratch are required to return to zero.
Named marginal readout scans the physical state, so this example explicitly
permits up to `2**26` amplitudes, matching its synthesis width ceiling. The
default readout budget refuses this larger state even for a one-bit marginal.

The independent Circuit calculates branch amplitudes using rational payoffs and
the probability law, without reusing native loading angles or generated gates.
It uses dense preparation/reflection matrices on the full ten-bit model. The
acceptance suite compares every complex amplitude at k=0/1/2, including private
environment, absolute phase and clean scratch, and checks two additional nonzero
exponent rates and insufficient resource budgets. This is quantum-program
acceptance; it does not certify the complete estimation Notebook workflow.

Native resource measurements are separate from the frozen original importer
baseline. Current native measurements are:

| Amplification rounds | Physical qubits | Clean workspace | Operations | Depth |
| --- | --- | --- | --- | --- |
| 0 | 25 | 15 | 393 | 340 |
| 1 | 25 | 15 | 1,111 | 948 |
| 2 | 25 | 15 | 1,829 | 1,556 |

The other ten physical bits are the complete model: indicator, two two-bit
source words and the five-bit reference. Private environment is not counted as
clean workspace, and dropping its names does not remove its simulation cost.

Mixed arithmetic selection retains supported Boolean segments and
uses checked conservative arithmetic where the bounded Boolean candidate refuses
a segment. Reasons remain in candidate diagnostics; exhausted construction
budgets propagate. This is not a claim of optimal resource usage.
