# Native QSVT from the Qiskit reference

This runnable FlagQuantum workflow preserves the commented source circuit's six
data bits, four exact phase angles, reflection sign and controlled arithmetic.
It needs no foreign SDK or Qmod source.

```bash
python -m examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.qiskit_qsvt.overview.main
```

Open [walkthrough.ipynb](walkthrough.ipynb) for the inline native qfunc and named
readout, [program.py](program.py) for the composable quantum logic, and
[WORKFLOW.md](WORKFLOW.md) for the source mapping and explicit limits.
[comparisons/circuit.py](comparisons/circuit.py) independently verifies full
complex states. Target conversion reuses caller-owned exact CX/U3 rules.
