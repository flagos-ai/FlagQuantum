# Complete Kaiser phase-window notebook audit

The frozen source is `phase_estimation/kaiser_window_state.ipynb`, blob
`c108729f0a6c6bf738d1cca65c91677e17380750`, at Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389`. All 16 original cells map to
`examples/notebooks/kaiser_phase_window.ipynb`, retaining source-cell indices.
Two additional cells demonstrate the separate quantized physical child.
Copyright and Apache-2.0 provenance are retained. Algorithms owns the notebook,
which composes existing window, QPE, drawing, local simulation and sampling
interfaces. No new numerical engine or protected export is introduced.

## Complete original scope

The five-bit alpha-two instance and symbolic instance retain original metadata.
The two signature diagrams use shared concrete rendering or explicitly labeled
symbolic declaration annotations. The original one-child graph retains its
32 coefficients, five-bit precision and weight one. Original and native cells
execute independently, with tests blocking Qualtran, Cirq and OpenFermion imports
throughout native execution. All three diagrams are rendered and inspected.

The complete second half remains present. It compares six-bit rectangular,
Luis–Perina and factory-generated Kaiser windows, at theta=0.51234, with 100,000
samples per window. The source composites a target OneState, QPE and OneEffect
and contracts the exact window tensor. The native notebook explicitly loads
the same ideal coefficients as simulation input, appends shared controlled
powers and inverse QFT through TextbookPhaseEstimation, and projects the retained
target-one amplitude. The identity PREPARE marks this injected tensor boundary.
It is explicitly separate from physical window preparation.

Full absolute complex output vectors match all three untouched source composite
tensors. No phase alignment is performed. The unused target-zero column remains
zero. The repeated control/QFT body contains 87 primitives, with 63 controlled
unitary calls, 15 QFT phases, six H and three swaps. Tensor input construction
is excluded from this execution inventory.

Sampling uses the existing `Circuit.sample` interface and a fixed native Torch
generator. It retains big-endian phase indices by removing the final target bit.
Cirq and Torch generators are independent. Every source and native histogram
is checked against the same exact distribution, within six binomial standard
deviations plus two events per bin. Identical random sample sequences are not
an acceptance claim. The original Holevo expression, including its cosine-angle
convention, is compared directly on identical supplied samples. It is preserved
without introducing a revised phase-variance convention.

All three sampled curves, original log-y scale, phase marker, sample count,
labels, variances and annotations are retained. The complete chart is rendered
and visually inspected. A headless Agg backend is selected by the test and
rendering workflow after the local GUI backend failed to initialize Tk. The
notebook retains ordinary Matplotlib display behavior. Original escape-sequence
warnings and the local IPython temporary-directory warning remain visible.

## Separate quantized physical execution and model

The added two-bit example executes `KaiserWindowState.make_on` via ordinary
`circuit.append(op)` on the existing DynamicCircuit, with an explicit negative
phase-gradient input. It uses five clean work qubits, two reusable slots and
seven measurement events. All work amplitudes outside zero vanish. The source
file's already-accepted exact stream, all 128 histories in each direction and
inverse evidence are reused.

The actual inventory totals 158 primitives, with 67 CX, 21 H, 14 T, 14 T-dagger,
ten X, seven S, seven CZ, seven measurements, seven resets and four Rx.
The independent declared model reports four And, 26 Clifford, four measurements
and peak nine. Both accounts are shown separately. The phase-preserving maximum
joint-amplitude error relative to the ideal tensor is 0.2561492360899099,
matching the original quantized child evidence. The source lacks a physical
Kaiser decomposition and those ideal/quantized and precision boundaries remain
open in Q-0622. The notebook does not close them.

## Verification

Every original and native cell executes, including all original sampling and
plotting. Tests verify metadata, exact coefficients, child calls, all three full
QPE tensors, sampling distributions, original Holevo convention, plot structure,
physical work cleanup and actual/model accounts. Strict mypy covers the test
and all extracted notebook code. Black, Ruff, Algorithms scope, architecture,
multi-level IR and protected API checks are run with direct dependency
regression. Evidence logs use `g2-kaiser-notebook-` in the reference workspace.
Q-0620 covers the complete notebook responsibility mapping.

The concentrated batch reports 148 passed. Strict mypy passes for both the test
and the full extracted code. All format, lint, scope, architecture, IR and API
checks pass. Original escape and environment warnings remain visible.
