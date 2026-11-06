# Precision-limited phase-gradient QFT

The emitter follows the [locked ApproximateQFT](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/qft/approximate_qft.py)
loop. At data position i it retains the min(i, precision - 1) nearest
earlier qubits, reverses them for the big-endian adder, and uses the
current qubit as the control. The add targets a gradient prefix and is
shifted right once. A Hadamard follows each addition. The optional final
swaps reverse the result order. The adjoint reverses that order and
subtracts each addend. An empty retained window needs no addition.

Data and gradient are caller-owned. The negative gradient eigenstate
gives the truncated Fourier transform and returns to its initial state.
The shared controlled adder restores the borrowed work. A one-qubit
data register uses only a Hadamard and can take an empty gradient.
The concrete epsilon formula follows the locked source.

With five data qubits and three gradient qubits, the emitted arithmetic
uses 56 T-family gates, 14 measurements, four clean work qubits and
four reusable slots. The source estimates 24 T for the same loop.
Its AddIntoPhaseGrad leaves have executable arithmetic and a resource
model, with no gate-level decomposition. Tests compare the original
window and operand bindings, the full small source joint operator,
raw measurement histories, and the truncated kickback with a prepared
and subsequently cleared gradient.
