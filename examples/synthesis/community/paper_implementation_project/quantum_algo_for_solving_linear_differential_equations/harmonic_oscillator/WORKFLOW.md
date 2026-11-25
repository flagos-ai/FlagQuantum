# Harmonic oscillator encoding and evolution

The native quantum function separates `encoding` and `evolution` and composes
them using `within`/`apply`. `physics.py` retains all 64 entries of the source's
eight-dimensional encoding matrix as ordinary immutable Python data. The
original two rounded input amplitudes and error bound 0.01 are retained.
The named `x,ancilla,y` outputs preserve all five logical bits and their order.

`control(~y, lambda: unitary(ENCODING, ancilla))` acts only on the zero-selector
branch. Its complete controlled matrix, conditional global phase and indexed
locations use the existing unitary verifier and decomposition. The other branch
remains identity. `within`/`apply` reverses the encoding, including the sign on y.

The eight original evolution matrices are exactly RY(-pi*k), with k the three-bit
ancilla word. Their binary weights permit three controlled rotations by
-pi, -2pi and -4pi. This uses the general composition law for rotations about
one axis, retains controlled minus identity and does not remove any branch.
The optional finite classical scale parameter changes every rotation
consistently; its default one reproduces the untouched source.

The independent Circuit constructs the controlled encoding blocks and all
eight analytic RY matrices directly. Acceptance checks the original, native
and independent complete state, all 32 input columns (including both selector
branches and nonzero targets), two nonzero scale/signed-amplitude bindings,
absolute phase, bit order, actual work cleanup and resource refusal. Primitive
control tests separately check complex matrices, conditional negative identity,
indexed/explicit-list order, the five-qubit combined bound and illegal aliases.

The native example uses 7 physical qubits, 2 workspace qubits, 387 operations
and depth 335. The original uses the same widths, 3821 operations and depth 3339.
The reduction follows from the verified rotation composition above; it is not
a claim of global circuit optimality or a general differential-equation solver.
Complete Notebook workflows remain separate migration work.
