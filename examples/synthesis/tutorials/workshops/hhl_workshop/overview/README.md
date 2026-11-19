# Native HHL Workshop

Run all 21 original code stages with native FlagQuantum qfunc and local CPU execution:

```bash
python -m examples.synthesis.tutorials.workshops.hhl_workshop.overview.main
```

Open [walkthrough.ipynb](walkthrough.ipynb) for the complete teaching workflow.
It computes the original 4x4 matrix exponential, prepares normalized [1, 2, 4, 3],
uses four fractional phase bits and 16 reciprocal amplitudes, and retains actual
postselection, five-decimal rounding, fidelity and the final comparison plot.

[WORKFLOW.md](WORKFLOW.md) describes translation and numerical limits.
No provider account or notebook-time dependency installation is required.
