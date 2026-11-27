# Typed binary-field addition

The Compiler binds the complete concrete GF2Addition and GF2AddK
decompositions from Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389` through
`field_addition_program`. The program splits big-endian coefficient words,
uses the existing CNOT or X gate emitter, and joins the owned through
registers with their original field descriptors. The source word is
preserved for register addition. Both operations need zero work wires and
produce zero classical events.

## Verification

The tests compare all ordered direct calls, including Split and Join, and
bind reversed and interleaved wire layouts against the unchanged Qualtran
Cirq operations. For widths 1, 3, and 4, register addition and both zero
and nonzero constants agree in every amplitude of a complex input state
through `Circuit.append`. The bound CircuitIR is serialized as JSON and
reconstructed before execution. The emitted primitives are one CNOT per
coefficient for register addition and one X per set constant bit. Both
totals independently agree with the locked QEC Clifford counts.

The focused regression covers this adapter, the previous whole-file
addition audit, the finite-field package surface, and the Program builder.
All 94 tests passed.

## Historical file boundaries

Q-0510 and Q-0513 use native resolved field descriptors at this entry.
The source foreign QGF constructors remain documented in
[the whole-file addition audit](FIELD_ADDITION_FILE_AUDIT.md).
Q-0511 and Q-0514 retain the upstream generic count and serialization
expected failures. The source declared gate counts omit Split and Join.
The native physical Program codec rejects the logical graph and the bound
CircuitIR codec accepts its executable result. The four inventory rows
remain open for these documented boundaries.


## Current acceptance — 2026-11-27

Q-0510/Q-0513 and their original test parents are accepted after the complete
literal constructor closure in [the whole-file audit](FIELD_ADDITION_FILE_AUDIT.md).
The historical boundary text above is superseded by that disposition. The raw
odd-characteristic degree-prefix circuit, exact scalar alias construction and
source declaration/cache/codec failures are individually mapped there. Numerical
nonbinary addition is not inferred from the raw circuit. Ordinary append,
coherent amplitudes and zero work use the same existing execution entry.
