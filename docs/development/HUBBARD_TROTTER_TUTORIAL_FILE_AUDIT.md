# Hubbard Trotter tutorial file audit

Q-0439 covers the Hubbard operator notebook in Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The original 42-cell notebook
matched blob `0d46f45fd50d00c60daa9de6fc7a32dbd9b75815` and passed in an
independent kernel. The corresponding native asset is
`examples/notebooks/register_hubbard_trotter.ipynb`. Its kernel test
rejects runtime imports from Qualtran, Cirq and Quimb.

The five original examples use the matched shaped through signatures,
immediate calls and weighted leaf totals. The model comparisons check
each exact original fixture, including leaf resource categories and peaks.

| Case | Immediate declaration |
| --- | --- |
| Hopping tile, side 8, angle 0.5 | 32 plaquette calls |
| Plaquette, kappa 8, synthesis tolerance 0.15 | Four two-bit Fourier calls and two Rz calls |
| Interaction, side 8, angle 0.5, coupling 4 | 64 Rz calls |
| HWP hopping tile, side 8, angle 0.15 | 128 two-bit Fourier calls, two 32-mode HWP batches |
| HWP interaction, side 8, angle 0.52728, coupling 4 | Two 32-mode HWP batches |

The original plaquette fixture passes positionally. Its first argument is
the angle, even though the Python variable is named `length`; its second
argument is the synthesis tolerance. The native tutorial preserves the
example's model angle and records the source tolerance convention. The
native graph uses the shared renderer's labels and the concrete model's
gate categories.

The executable two-by-two example appends a hopping color and an
interaction to one dynamic circuit. The native bindings share three
work wires and allocate distinct feedback slots for the two append calls.
The combined circuit emits 32 T-family instructions and four
measurements on 11 physical wires. The source HWP interaction model
independently reports a peak of five logical qubits. The operator audits
retain the complex-state and cleanup evidence and source decomposition
gaps for Q-0437 and Q-0440.
