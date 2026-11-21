# iQuHack state-preparation workshop

Open [walkthrough.ipynb](walkthrough.ipynb) or execute the workshop locally.

```bash
python -m examples.synthesis.community.hackathons.iquhack_2025.workshop.overview.main
```

The runnable reference exercises retain all 64 terms of the coherent
arithmetic example, the seeded 32-entry probability array and the five-bit
exponential warm-up. The signed exercise computes `5 + (-7.5) = -2.5`.
It uses a 14-qubit Circuit with sign extension and controlled increments
in units of one half. In preflight, the direct signed qfunc required a wider
working register than the local statevector budget. The Circuit exercise
checks all 256 input pairs in one coherent witness.

The November 21, 2026 native preflight retains the same input formats:
`QNum[3]`, `QNum[5, True, 1]` and result `QNum[6, True, 1]`. Immutable input-bit
borrowing first reduced retained lowering from 36 work bits / 286 instructions
to 25 / 264, with recomputation changing from 25 / 304 to 25 / 264. Those
historical measurements remain unchanged.

Equal-width modular-sum fusion now emits the terminal XOR result directly,
while read-only sign aliases avoid copying an extended signed word. Both
schedules use **11 work bits and 49 arithmetic instructions**. The visible
native preparation of `5` and `-7.5` adds four instructions, so its compile-only
report is **25 total qubits, 53 instructions and depth 24**. A fourteen-qubit /
zero-workspace native request still refuses. The original-budget Circuit
execution remains in place; the Notebook does not execute a 25-qubit state.
Compiler checks cover all 256 inputs and 64 arbitrary XOR targets, inverses,
shared negatives, unequal moduli, and smaller complete coherent witnesses with
absolute phase and clean workspace. This is a general modular arithmetic rule,
not an optimization keyed to this Notebook or these constants.

The amplitude exercise uses native indexed rotations to load `x²` on the
indicator, with `x = word / 32`. It draws the 2,048 shots observed in the
source transcript and derives both joint and marginal counts from them.
The original scalar parameters marked `a` and `b` are unused by this
exercise. The warm-up applies the positive-sign QFT to a probability grid
that decays as `exp(-0.1 * word)` and executes one seeded bonus draw.
All local checks compare complete complex states with independent
arithmetic, probability or Fourier references.
