# Complete generalized signal processing notebook migration

Q-0653 maps all twelve cells of frozen Qualtran v0.7.0
`qsp/generalized_qsp.ipynb`, blob
`bdc342ca7bb678eac2c98065d198b035a19bc293` at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Google LLC 2023 Apache-2.0
attribution remains in `examples/notebooks/generalized_signal_processing.ipynb`.
Two additional cells demonstrate ordinary append and independent resource counts.

| Original cells | Native responsibility |
|---|---|
| 0–1 | Scientific title and domain-owned Algorithm, Compiler, Core, Simulation and Drawer imports. |
| 2–3 | Complete P/Q block-encoding, Laurent polynomial and precision contract. Existing native named unitary emitter and operation replace the external Bloq object. |
| 4–7 | All three original X factories, coefficient pairs and Laurent powers 0/1/5. |
| 8–9 | All three graphical signatures through existing typed Core signatures and Drawer. |
| 10–11 | Original one-level graph, two signal-rotation calls and one open-controlled X, with the same multiplicities. |
| Added 12–13 | Complete coherent complex input through ordinary append, exact zero idle and physical/model resource separation. |

The native test executes both complete notebooks and compares coefficient arrays,
all phase factors, original factory precisions and powers, complete port names,
widths, shapes and sides, graph child identities/multiplicities, complete complex
output and idle zero. Native execution rejects Qualtran, Cirq and OpenFermion
imports. Source code/assertions and the frozen notebook hash remain unchanged.
The four SVG graphs render to PNG and have been visually checked for complete
labels and edges. The native notebook emits seven display objects.

The notebook append emits 6 X,4 phase,4 Rz,2 Rx,2 H and1 controlled phase, zero
measurements and zero scratch. It has one unitary branch and one verified idle
zero qubit. Its separate declared model has 6 continuous rotations, 3 Clifford and
peak 2 logical qubits. Existing small complete-operator tests and provider branch
evidence are reused, with numerical precision/source defects retained in the
[complete file audit](GENERALIZED_QSP_FILE_AUDIT.md).

The concentrated batch passes 458 cases. One IPython read-only home warning
selects a temporary directory. Strict mypy covers every extracted code cell,
both tests and both changed owners. Black/Ruff, Algorithms scope, architecture,
IR and protected API checks pass. The original isolated notebook test requires
local Jupyter kernel ports and runs with that environment permission. Its final
result is recorded below.

The unchanged original notebook test passes in an isolated kernel, with 141
other cases deselected and seven upstream marker/collection warnings. Its first
sandbox attempt was blocked by local port binding. The permitted isolated-kernel
run passed. After the initializer hash check and documentation edits, 32 focused
cases pass with zero warnings and strict checking of all five Python surfaces
passes again. Q-0653 closes its complete notebook responsibility.
