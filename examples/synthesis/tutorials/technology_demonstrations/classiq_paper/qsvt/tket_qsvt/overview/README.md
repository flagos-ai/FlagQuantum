# Native reconstruction of the Tket QSVT reference

This FlagQuantum workflow retains six data bits, three block bits, one auxiliary,
the four source phases, literal half-turn rotations, the non-diagonal reflection,
and positional box wiring. It needs no foreign SDK or Qmod source.

```bash
python -m examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.tket_qsvt.overview.main
```

Open [walkthrough.ipynb](walkthrough.ipynb) for the inline native qfunc,
[program.py](program.py) for the composable quantum logic, and
[WORKFLOW.md](WORKFLOW.md) for source mapping and explicit limits.
[comparisons/circuit.py](comparisons/circuit.py) independently verifies complete
complex states. Target conversion uses [exact_basis.py](exact_basis.py).
