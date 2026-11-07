# Classical exchange values

Simulation owns `classical_swap.py`, adapted from the unchanged Qualtran v0.7.0
`basic_gates/swap.py` and `swap_network/cswap_approx.py`, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. Compiler retains the existing exact
swap, seven-T Fredkin and relative-phase four-T decomposition; no gate emitter
or production SDK dependency is added.

The uncontrolled callback simply exchanges the original objects. It neither
copies arrays nor interprets register widths. The controlled callback compares
with zero, then one, returns a literal Python integer control, and rejects any
other value. Array ambiguous-truth errors, symbolic comparisons and unused
width metadata retain their original behavior. It does not adopt the fanout
callback's broader nonzero-control policy.

The same controlled numerical rule serves TwoBitCSwap, CSwap and CSwapApprox.
For CSwapApprox it describes only the computational-basis permutation. Its
executable circuit has relative phases, which remain present in the native IR.
Tests explicitly demonstrate that the raw callback cannot predict the full
complex output of that circuit, while the native circuit matches the original
source stream and complex execution. The inverse restores the complete input.
This is an evidence boundary, not a repair to the reference.

Run `python -m examples.algorithms.classical_swap_values` to use
`circuit.append(op)` with exact and relative-phase exchange, occupied targets,
and a spectator. Both require no work or measurements. The exact one-bit
exchange emits seven T instructions; the source CSwap model defaults to four
T equivalents, or seven when the conversion weight is explicitly selected.
The relative exchange emits four T and its source model has four T equivalents.
Existing source tests verify both resource bases and the original instruction
dependencies. Raw callback equality does not establish quantum phase equality.

Reference tests cover object identity and aliasing, scalar control type changes,
invalid controls, arrays and symbolic widths. Coherent valid-bit tests cover all
input amplitudes and inverse restoration. Existing predicate-controlled tests
retain every small measured branch and work-cleanup verification. No complete
source-file acceptance is claimed by this callback slice.
