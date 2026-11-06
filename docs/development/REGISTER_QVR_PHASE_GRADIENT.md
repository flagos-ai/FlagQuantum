# Quantum variable rotation through a phase gradient

The shared register compiler ports the concrete unsigned-cost
QvrPhaseGradient planner from locked Qualtran v0.7.0. It selects the
coefficient format from the cost integer width and requested epsilon.
The gradient width is the smaller of the source formula and the
source's fixed-point search below 54 cost bits. Wider cost registers
use the formula. The search visits the original width bounds and
compares the per-term accumulated fraction on the all-ones cost word
with the fractional product. The optimizer uses the source's
floating-point alignment before converting negative shifts to raw
integers. The circuit uses the separate per-term truncating gate
schedule of AddScaledValIntoPhaseReg.

The native QVR emitter binds the selected formats to the shared
scaled phase addition. Callers own the negative-exponent gradient,
the clean carries and their measurement slots. The adjoint reverses
the same schedule. This slice supports concrete finite coefficients
and positive epsilon below one full turn.

The source precision choices are checked for fractional and
integer-containing cost words, both coefficient signs, wide search
witnesses and the 54-bit formula cutoff. The source Soquet graph
provides the scaled-add register binding. Its classical action
provides the expected kickback, and the previously verified
scaled-add emitter supplies the native gates. Tests compare complete
complex vectors, all small raw histories, sparse wider histories and
And-channel certificates. Each numerical test checks the maximum
complex phase error over all cost words for its case.

The original selection reaches log2(0) when the coefficient
quantizes to zero. The native planner rejects that case explicitly.
The locked environment also overflows its fixed-point library for
some integer-heavy input formats. One witness is cost (47,4),
coefficient 2.393029727807961 and epsilon 0.005779784227676086.
The native search reports no accepted gradient width for this
witness. The original recursive qubit estimator fails for cost
(4,1), coefficient -0.375 and epsilon 0.1 because its scaled leaf's
full-width fractional dtype differs from the cost dtype. The source
gate model and the outer classical action still provide evidence
for that case.

| Cost format, coefficient, epsilon | Gradient bits | Physical wires | Clean work | Literal T-family | Literal CX | Literal measurements | Source model Toffoli / qubits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| (2,2), 0.5, 0.5 | 3 | 7 | 2 | 8 | 20 | 2 | 1 / 5 |
| (3,3), 0.75, 0.25 | 4 | 10 | 3 | 24 | 61 | 6 | 4 / 7 |
| (4,1), -0.375, 0.1 | 4 | 11 | 3 | 20 | 54 | 5 | 4 / unavailable |
| (8,8), 0.625, 0.03 | 9 | 25 | 8 | 64 | 178 | 16 | 14 / 17 |

The original model charges no measurements for its cost-only
arithmetic leaves. The literal native counts include measured
carry uncomputation. The eight-bit cost case's maximum complex
phase error is about 0.009204.

Run `python -m examples.algorithms.register_qvr_phase_gradient`
to select a gradient, apply the operation through
`circuit.append(op)`, and verify gradient and work cleanup.
