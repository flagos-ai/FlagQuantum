# Native lattice-folding energy helpers

Six `fold` decisions encode three lattice directions; `active` is the retained
interaction decision. Named short-contact, long-contact and overlap helpers
compose the energy phases. All five cost/RX layers, integer penalties, squared
residuals and the activity-dependent 2600 offset are preserved.

```python
import flagquantum as fq
from examples.synthesis.applications.chemistry.protein_folding.protein_folding_with_qaoa.protein_folding.main import build_program

params = tuple((-1) ** i * (0.003 + 0.001 * i) for i in range(10))
program = build_program(params)
result = fq.run(program)
print(dict(program.probabilities(result, registers=("fold", "active"))))
print(program.report["resources"])
```

The first five angles drive energy phases, and the final five drive RX mixers.
Two-bit array views pass consecutive directions to the helpers. Users choose
logical decision names and phases without assigning physical wires or work bits.

An independent classical model computes one-hot lattice directions and their
contact and overlap energies. All 128 costs are compared with a separate
arithmetic interpreter of the frozen original. A reference Circuit transforms
the independent seven-bit energy table to spin parities and applies explicit RX
mixers. Original and native programs must match complete complex amplitudes for
two nonzero parameter sets and both objectives within 1e-12, including absolute
phase, declared field order and hidden-work probability below 1e-24.

The native decomposition currently uses seven qubits, zero work bits, 40,847
operations and depth 33,409. The original uses 22,312 operations and depth 18,286.
Separate helper phases are lowered independently; this retained semantic example
exposes a future compiler optimization requirement to combine adjacent compatible
phase expressions across helper boundaries. It makes no resource-optimality claim.
Physical qubit, operation, depth and memory budgets remain enforced.
The complete associated Notebook workflow remains pending.
