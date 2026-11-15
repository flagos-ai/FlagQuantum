# Register negation whole-file audit

Q-0210, Q-0211 and Q-0212 cover the negation notebook, implementation and
tests at Qualtran v0.7.0 commit `bcb17bb25360ed68ce3bc084645db3e929799389`.
All three original Git blobs match the frozen inventory.

## Ownership and contracts

Compiler owns big-endian wire binding, complement, increment and feedback.
Simulation owns the original classical composition and its intermediate
validation. The target is THRU. A width-n word is mapped to its additive
inverse modulo 2**n. The caller supplies n clean constant wires, n-1 clean
carry wires and n-1 feedback slots. Every quantum work wire returns to
zero. The feedback slots hold the latest measurement outcomes. Negation
is an involution.

The existing `negate_ir` executes BitwiseNot followed by AddK(1). The
single-control addition owner now also composes controlled negation from
the shared controlled complement and controlled AddK(1). The increment
controls the constant load and unload. The adder itself runs on both
branches. Control-on-zero retains the source wrapper's two outer X gates.
The control and target stay on their original physical wires. No extra
control work is allocated.

| Locked definition | Native disposition |
|---|---|
| Negate dtype and cached signature | Preserve x and both signed factory widths in neutral metadata. The established compiler accepts concrete signed, unsigned and Montgomery bit patterns. |
| Composite build | Existing plain compiler chain and Simulation callback reuse complement and AddK. The original operation order and x binding also match at symbolic width. |
| Generic control of Negate | Both concrete single-control polarities follow the locked wrapper and child order. Generic multi-register control and symbolic primitive lowering remain outside this batch. |
| Two factories and documentation specification | Both signatures and complete one-level graphs match. The asset preserves the original topics and adds coherent negative-control execution through ordinary `circuit.append(op)`. |

The source's one-bit signed AddK leaf cannot load +1. The existing source
failure witness stays in the tests alongside the distinct valid native
one-bit permutation. The source zero-width classical failure and unresolved
symbolic classical failures also remain explicit. The implementation's
integer formats and modulo output are described alongside the original
teaching prose.

## Original tests and native execution

The two original test functions cover both factory autotesters and all
unsigned inputs at widths 1, 2, 3, 4 and 8 with signed decoding. The run
reports 9 passed and 1 skipped. The skipped count-mode fixture reports
that Negate only has counts from decomposition. Native declarations compare
both decompositions and their operand/output connections directly.

Prior plain-negation tests cover three encodings, the signed one-bit
failure, per-wire gate order, coherent complex amplitudes with a spectator,
and complete small measurement histories. New controlled tests cover both
polarities at 1/2/3/5/8 bits against the untouched unsigned source
decomposition. They compare emitted primitive identity, per-wire order,
operands, measurement feedback and output maps. At widths 1–3, exact
postselection checks every measurement history on a superposition of
all inputs, controls and spectator values. The expected full states check
clean work and involution separately for each history.

| Eight-bit account | Actual primitive subset and work | Independent source model |
|---|---|---|
| Plain | 81 CX, 28 T/TDG, 7 measurements, 15 clean work wires | 7 And and 7 measurements |
| Control on one | 91 CX, 28 T/TDG, 7 measurements, 15 clean work wires, 0 X | 7 And, 56 Clifford, 7 measurements, 24 logical qubits |
| Control on zero | 91 CX, 28 T/TDG, 7 measurements, 15 clean work wires, 2 X | 7 And, 58 Clifford, 7 measurements, 24 logical qubits |

The controlled model is the shared controlled increment model plus n
logical complement CX gates. Primitive counts include explicit expanded
And and erasure work. Model Clifford categories are kept separately.

## Teaching asset and remaining boundaries

The unmodified 11-cell notebook and the native 13-cell notebook execute in
independent kernels. The native kernel blocks reference imports. The native
asset retains both original instances, the graphical signatures and the
concrete one-level graph. It adds an executable coherent control example.
Q-0210 closes. Q-0211 and Q-0212 retain the original signed one-bit gate
failure, generic symbolic/multi-register control and the original skipped
count-mode boundary.
