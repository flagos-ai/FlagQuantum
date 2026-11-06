# Phasing via a cost function

The compiler binds the pinned Qualtran v0.7.0
`PhasingViaCostFunction` evaluation, phase and evaluation-adjoint
sequence. The phase receives the named cost outputs and its own extras.
Evaluation and erase share the input, cost and scratch bindings. Both
Hamming phasing emitters use this composition through their existing
entry points, including the `circuit.append(op)` example.

The acceptance test reads the original outer graph, including the
adjoint's inputs and the retained gradient output. It compares the
direct source ZPow leaves and uses the verified scaled-add emitter for
the gradient leaf. On two input bits, both phase paths cover integer
and fixed-point costs and project all measurement histories. The
complex output includes a coherent spectator and checks restoration
of the input, gradient and scratch. The pinned Hamming composition's
Cirq simulation is marked xfail upstream. The native branch oracle
and the original graph checks are separate.

| Cost interpretation | Native T-family / CX / phase / measurements / wires | Source Toffoli / And / rotations / measurements |
| --- | --- | --- |
| Integer, direct | 4 / 18 / 2 / 1 / 5 | 0 / 1 / 2 / 1 |
| Fixed point, direct | 4 / 18 / 2 / 1 / 5 | 0 / 1 / 2 / 1 |
| Integer, gradient | 12 / 38 / 0 / 3 / 10 | 1 / 1 / 0 / 1 |
| Fixed point, gradient | 16 / 44 / 0 / 4 / 12 | 2 / 1 / 0 / 1 |

Native phase instructions are unsynthesized. Counts exclude the
spectator and caller-owned gradient preparation and unpreparation.
Source counts are the locked resource model and count the Hamming
measurement separately from abstract Toffolis.

The locked Square and its adjoint expose classical values, tensors
and resource counts. They raise DecomposeNotImplementedError when
asked for a gate decomposition. Square-based gate parity is tracked
separately from the verified composition.
