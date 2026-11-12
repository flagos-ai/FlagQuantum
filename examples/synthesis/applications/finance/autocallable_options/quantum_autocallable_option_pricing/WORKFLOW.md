# Native autocallable quantum program

This program implements the pinned finite autocallable preparation and bounded
amplitude amplification. The surrounding Notebook, financial estimator and
external execution workflow require separate migration.

```python
import flagquantum as fq
from examples.synthesis.applications.finance.autocallable_options.quantum_autocallable_option_pricing.main import (
    build_program,
)

program = build_program(k=1)
result = fq.run(program)
probabilities = program.probabilities(
    result, registers=("indicator",), max_amplitudes=2**24
)
print(dict(probabilities))
```

The source model predicts indicator probability approximately `0.7601369491`
after one iteration. This is a finite quantum-model result, not an estimated
market price. Iteration counts are integers in `0..4`; the independent complete
state acceptance covers `0`, `1` and `2` with the original shift `0.1274075` and
the alternative nonzero shift `0.22`.

The quantum model keeps three two-bit source words, a four-bit integration
reference and an indicator. Each source word uses the original four-outcome
probabilities. At the source's two-fractional-bit precision, affine coefficients
are individually truncated toward zero. The resulting raw increment is
`grid - 2` at the original shift and `grid - 1` at the alternative shift.
`physics.py` derives comparison bounds from this rule; it does not evaluate
quantum data on the host. The default and alternative cumulative sums both fit
the original signed four-bit format.

`operations.py` tests the cumulative call and barrier conditions coherently.
The two early binary payoffs retain their source rotations. The final payoff
either applies its fixed rotation or prepares the normalized exponential
reference distribution and XORs the indicator with the comparison result. The
comparison threshold preserves the original sign-bit flip. Temporary threshold
and reflection predicates are uncomputed by `within`/`apply`; deterministic
cumulative values and branch flags need not persist as extra quantum registers.

`main.py` composes preparation, phase marking and the exact signed Grover
reflection. Private source and reference names are dropped after the program;
that does not reset or discard their entangled state. Native acceptance compares
the complete joint purification and the public indicator density, not just the
indicator probability. Additional tests use a nonzero indicator and a complex
coherent indicator so XOR behavior and relative phase remain observable.

The independent reference uses five-bit Circuit branches for all 64 source
words. It constructs the complete eleven-bit purification from their complex
amplitudes, then evaluates `2|prepared><prepared| - I` as an exact rank-one
reflection. This avoids a dense 2048-square reference matrix. It uses neither
the native predicate bounds nor native construction or source-importer code.
Tests separately execute the original fixture and the changed-shift source.

Current native synthesis uses 24 physical bits: eleven model bits and thirteen
clean work bits. Reading one named output still scans the full dense state, so
the example explicitly admits `2**24` amplitudes. The default read limit remains
unchanged and is tested to refuse this scan. The one-GiB synthesis memory limit
bounds the planned state buffer; it is not a process peak-memory guarantee.
Native costs and the frozen original costs are reported separately. No resource
optimality or full Notebook-workflow claim is made.
