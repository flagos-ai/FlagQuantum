# Concrete fixed-point amplitude amplification

Run `python -m examples.synthesis.algorithms.amplitude_amplification.qsvt_fixed_point.main`.
The program keeps the original 26 phase constants, six state bits and live
auxiliary output. `a`, `b`, and `c` explicitly replace the original record's
2/1/3-bit fields in that order. The source block encoding is identity.

A marked predicate is computed into a fresh `Annotated[QNum, 1]` local.
`within`/`apply` automatically reverses that computation and releases its zero
storage. The projected rotation uses
`phase(aux - 0.5, angle)` and `phase((1 - 2 * aux) * marked, angle)`;
the half-angle term preserves absolute phase for both auxiliary values.
The target condition `a-c == -2` is expressed as `a+2 == c` to avoid unsigned
subtraction. Two bounds express `a+b == 3` with a statically typed quantum
predicate. The 13 alternating pairs include the source's initial rotation,
12 paired steps and final rotation.

The independent Circuit enumerates every projected diagonal amplitude and
conjugates initial-state projections by six Hadamards. Tests compare the
untouched source, native program, two nonzero phase-sequence variations and
phase-weighted superpositions of every state/auxiliary branch. Hidden work is
required to return to zero; operation, width, depth and memory limits still fail
closed. This validates the concrete quantum program, not a generic QSVT callback
or the complete associated Notebook workflow.

Current native lowering uses 21 physical qubits (14 workspace), 14015 operations
and depth 10218. The untouched source uses 10 physical qubits, 20542 operations and
depth 16850. Fewer operations here cost additional workspace; there is no universal
resource-optimality claim. The direct predicate-controlled phase formulation
exceeded the 100000-operation construction cap; explicit compute/phase/uncompute
uses the existing arithmetic and Boolean lowering within that cap.
