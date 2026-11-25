# One-step circle-walk circuit generation

Compile one controlled walk step at each of the original seven position widths,
five through eleven. Compare a minimum-width circuit with a bounded search for
a smaller CX count, measured in the exact CX/U3 basis.

```bash
python -m examples.synthesis.tutorials.technology_demonstrations.discrete_quantum_walk_circle.overview.main
```

Open [the Notebook](walkthrough.ipynb) for the complete sweep and the original
published comparisons. Figure labels separate current local timings from the
archived Qiskit and Classiq measurements.
