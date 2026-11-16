# Quantum double-slit workflow

`main.py` declares the position and output registers, prepares the wave source
and calls the QSVT program. `operations.py` expresses periodic lattice shifts,
scatterers, absorbing boundaries, selector preparation and projectors as reusable
quantum functions. `physics.py` retains all 192 source phases and the absorbing
phase. The user entry has no physical wire numbers or bit-string decoding.

The complete counterpart uses all 95 QSVT cycles. The one-cycle option is a
finite diagnostic only. Both pass complete complex-state comparison against an
independent mathematical gate-tape reference and a separate public Circuit
construction with explicit wires and clean Toffoli ladders. Absolute phase,
little-endian numeric registers and all nine zero work bits are checked.

The Circuit comparison uses the ordinary target-independent optimizer, then an
independently checked workspace-layout hint. It imports neither source text nor
synthesis internals. The native program stays within the unchanged 24-qubit,
9-workspace, 100000-operation and 256 MiB budgets. Resources for the native and
Circuit implementations are measured separately; equality or optimality is
not claimed. This acceptance covers the quantum program, not the full physical
simulation Notebook, plotting or application-accuracy workflow.

```sh
python -m examples.synthesis.applications.cfd.double_slit_experiment.quantum_double_slit_experiment.main
python -m examples.synthesis.applications.cfd.double_slit_experiment.quantum_double_slit_experiment.comparisons.circuit
python -m pytest tests/integration/test_synthesis_native_double_slit.py -q
```
