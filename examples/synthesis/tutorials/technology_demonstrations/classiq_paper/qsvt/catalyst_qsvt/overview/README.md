# Native reconstruction of Catalyst QSVT

This FlagQuantum workflow preserves the source's six data bits, four phases,
positive block controls, negative-zero reflection and big-endian modular shifts.
It explicitly corrects the commented source's nine-device-wire declaration to
ten logical wires. No foreign SDK or Qmod source is required.

```bash
python -m examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.catalyst_qsvt.overview.main
```

Open [walkthrough.ipynb](walkthrough.ipynb) for an inline native qfunc and named
readout, [program.py](program.py) for the quantum composition, and
[WORKFLOW.md](WORKFLOW.md) for source mapping and the disclosed corrections.
[comparisons/circuit.py](comparisons/circuit.py) provides an independent runnable
Circuit. Reported CX/U3 costs belong to the explicit local target, not the
source's CX/RZ/RY/RX and qjit pipeline.
