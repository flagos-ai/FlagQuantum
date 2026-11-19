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

The amplitude exercise uses native indexed rotations to load `x²` on the
indicator, with `x = word / 32`. It draws the 2,048 shots observed in the
source transcript and derives both joint and marginal counts from them.
The original scalar parameters marked `a` and `b` are unused by this
exercise. The warm-up applies the positive-sign QFT to a probability grid
that decays as `exp(-0.1 * word)` and executes one seeded bonus draw.
All local checks compare complete complex states with independent
arithmetic, probability or Fourier references.
