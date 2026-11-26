# Computational basis constructor closure

The complete frozen z_basis.py source at Qualtran v0.7.0 commit
bcb17bb25360ed68ce3bc084645db3e929799389 is owned as described in
[the full-file audit](Z_BASIS_FILE_AUDIT.md). This batch adds the six previously
missing explicit constructor mappings for ZeroState, OneState, ZeroEffect,
OneEffect, IntState and IntEffect. Existing Z, CZ and measurement owners remain
authoritative. Registration alone does not establish complete catalogue acceptance.

Compiler formats/legacy_basis_vector_factories.py validates metadata and delegates
to the existing integer_basis_function/integer_basis_program owners. Scalar
atoms retain q/QBit ports; integer vectors retain val, QBit at width one and
QAny at other widths. States create RIGHT outputs in caller-provided clean
storage. Effects consume LEFT inputs with an explicit ownership and clean-release
promise. No effect is implemented by reset, trace, or unconditional discard.
Arbitrary input covector contraction remains with the existing tensor owner.

All positive-width constructors at serialized depths zero, one and two resolve
to the same complete Program. Exhaustive values at widths one through four,
and the original eight-bit value55 plus255, check MSB order, reversed physical
bindings, full complex output and a coherent borrowed spectator. The applicable
measurement history is empty. Both an independent forced instruction execution
and the normal circuit.append(op) runtime are checked. Missing effect ownership
retains its refusal. Existing full state/covector, source-call and model evidence
is reused from the shared owner tests and the full-file audit.

## Source boundaries

Original scalar basis constructors refuse n other than one. Symbolic integer
values and widths retain logical declarations without inventing an executable
loader. Width-zero constructors serialize at depth zero, but original state
expansion raises BloqError after Join creates QAny(1) against QAny(0); original
effect expansion raises IndexError. Native resolution retains their declaration
and physical refusal, rather than silently replacing the broken source body.
Tests reproduce both unchanged-source failures separately.

The existing native integer helper validates concrete unsigned loader inputs.
Malformed metadata, out-of-range concrete constants and inconsistent signatures
are refused. Source permissive symbolic-invalid metadata and nominal singleton
Join dtype differences retain their existing separate validation/view boundaries;
this mapping does not claim arbitrary symbolic physical execution or complete
foreign type equivalence. Generic adjoints require a separately supplied exact
owner; no covector is reinterpreted as a unitary inverse.

## Resource levels

Zero/One atoms emit zero/one X, while their original QEC model is zero.
An integer value emits popcount(value) X in either directional checked body;
the original declaration is one abstract ArbitraryClifford on its logical width.
Value55 on eight wires emits five X;255 emits eight X. No measurement, reset
or private scratch is introduced. The spectator is borrowed and unchanged.
Model-only and source-failure declarations have no measured primitive count.
Existing source files and public Core contracts are unchanged.
