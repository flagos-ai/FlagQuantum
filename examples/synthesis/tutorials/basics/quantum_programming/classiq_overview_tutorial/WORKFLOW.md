# Native quantum programming overview

The user writes `y |= x**2 + 1` after allocating a three-bit input and preparing
its uniform superposition. Synthesis constructs reversible arithmetic and manages
its workspace. No physical wire indices or bitstring decoding are needed.

The reviewed Notebook defines this program, synthesizes it, opens the circuit
viewer and samples the results. `main.py` implements the same local computational
workflow: native synthesis, text drawing, execution, named probabilities and
1000 seeded samples from the executed distribution. The text drawer replaces
the cloud viewer and local sampling replaces the remote sampling session.

All eight named pairs `(x, x*x + 1)` have probability 1/8. The independent Circuit
comparison checks every complex amplitude, including absolute phase and zero
hidden workspace. The workflow test additionally runs drawing and sampling;
this is local simulation and does not represent a hardware measurement.

The [provenance record](provenance.json) identifies the unchanged quantum source,
the Notebook and every reviewed code cell. Run the commands in [README](README.md).
