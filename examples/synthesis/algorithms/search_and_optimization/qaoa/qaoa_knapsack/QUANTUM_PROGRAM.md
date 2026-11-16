# Constrained Knapsack quantum program

All three layers and all six parameters are retained, ordered gamma values then
beta values. The cost phase is `-3*a-5*b` exactly when `2*a+3*b<=12`.
Native numeric ports `a: QNum` (three bits) and `b: QNum` (two bits) explicitly
replace the source fields `v.a` and `v.b`; this is not general qstruct support.
The inclusive boundary and all 32 basis assignments are preserved.

The captured cost helper reads both words without changing them. Each mixer is
`exp(-i*beta*sum(X)/2)`, which equals RX(beta) on every bit because its X terms
commute. No source importer or physical wire numbers appear in the native program.

Two nonzero six-parameter vectors and both width/depth objectives agree with
independent classical predicate costs and explicit Circuit parity/RX rotations.
The comparison uses raw complex128 amplitudes in logical `(a,b)` order, preserving
absolute phase despite different physical layouts, and checks zero workspace.
Measured native resources are 5 qubits, 848 static instructions, depth 721 and
maximum instruction width 2. The separately retained source has 5 qubits,
856 instructions and depth 721. These are IR instruction counts, not a hardware
basis-gate or optimality claim; local evolution fusion may change counts.

Run `main.py` and `comparisons/circuit.py` using the module commands in [README](README.md).
The shared native cost and ansatz helpers live in the parent QAOA example.
Classical optimizer training and the related complete Notebook workflow remain
pending; this acceptance covers the complete retained quantum program.
