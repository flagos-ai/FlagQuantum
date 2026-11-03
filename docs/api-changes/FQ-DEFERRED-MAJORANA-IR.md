# Deferred Majorana instruction

Status is proposed. This draft requests review from the IR and compiler owners.

A Majorana operation currently needs a dense matrix to enter the IR. Add a
matrix-free form of `selected_majorana_fermion` using the existing instruction
fields. The ordered operands are control, selection, then target. The
`control_width`, `selection_width` and `target_width` parameters determine
arity and register boundaries. `target_gate` is X, Y or Z.

Existing matrix-bearing instructions retain their execution behavior.
Construction and JSON round-trip preserve the logical form. The compiler and
`Circuit.state()` use the same lowering function to materialize the dense
reference matrix before numerical execution. This lowering supports up to ten
operands. Older readers need the new opcode support to consume the matrix-free
form. The JSON envelope and existing opcodes keep their current shape.

The conformance test covers composition onto reordered qubits, JSON
round-trip, legacy matrix payloads, and the native compile and run paths.
