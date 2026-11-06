# Recursive phase-gradient QFT

The emitter follows the [locked QFTPhaseGradient](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/qft/qft_phase_gradient.py)
recursion. Transform the left half without swaps, accumulate the product
of the bit-reversed left half and the right half into the full gradient,
then transform the right half. Each child uses the appropriate gradient
prefix. Only the outer call may reverse the result order. The inverse
reverses the stages and subtracts the product.

The caller binds equal-width big-endian data and gradient registers.
The negative gradient eigenstate is preserved and induces the textbook
QFT. The joint operator is also defined for other gradient states.
The shared product accumulator borrows a clean pool sized at the root,
reuses it at every recursion level, and restores it after every product.

At width four the emitted circuit uses 52 T-family gates, 13 measurements,
nine clean work qubits and three classical slots. The recursive source
leaves estimate 48 T in total. The locked aggregate graph estimates 40 T
because the equal child entries collide in its dictionary. Its child
entries also default to bit reversal, while the executable children omit
it. The source product leaf supplies an arithmetic gate and model without
a primitive decomposition. Tests compare the executable parent bindings,
the native composed stream, the full source operator for small registers,
and cleanup of a prepared gradient through the append entry point.
