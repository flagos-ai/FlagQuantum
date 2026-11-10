# Four-slice quantum time marching

Run from the repository root:

```sh
python -m examples.synthesis.algorithms.quantum_differential_equations_solvers.time_marching.main
python -m examples.synthesis.algorithms.quantum_differential_equations_solvers.time_marching.comparisons.circuit
```

This program retains all four slices of the source's duration-two interval,
seven even/eight odd QSVT phases and a three-bit counter initialized to four.
Each slice decrements the counter modulo eight only when time, block, signal
and selector lie in the exact success subspace. The unused amplification bit
is preserved. The native example exposes named outputs for the original packed
struct fields and a clean success marker; it does not load a final-state table.

The time-dependent block combines commuting Y rotations through an equivalent
phase polynomial. The selector's in-place preparation is RY(pi/2), undone by
RY(-pi/2), rather than assuming that H has the same borrowed-unitary behavior.
Projector rotations retain both branch and absolute phases. A success marker
is coherently computed with H/phase/H, used for decrement, then uncomputed.
No production API, ownership or construction budget is changed.

Independent Circuit validation uses controlled-Y matrices, explicit projector
permutations and RZ matrices, reverse inverse order and a controlled counter
permutation. All complex amplitudes are compared with native and frozen source
states after an explicit packed-source/named-native axis permutation. Three
durations/phase sets exercise the complete algorithm. Complex nonzero inputs
exercise every success/counter branch, modular wrap and borrowed-step inverse;
alias, invalid input and budget refusals are tested. Compiler work and the
explicit marker are checked separately for zero leakage.

Native default: 13 physical bits including one compiler work bit and one
explicit marker, 18,344 operations, depth 15,762. Frozen source: 15 physical
bits, 5,118 operations, depth 4,355. Phase polynomial expansion increases native
gate count substantially; this remains an optimization gap. Source resource
records are retained. This is quantum-program acceptance, not certification
of the complete Notebook workflow or a performance claim.
