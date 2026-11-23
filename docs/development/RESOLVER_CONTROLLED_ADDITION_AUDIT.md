# Controlled-addition constructor dependency audit

## Complete source scope and ownership

The unchanged reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Complete
`arithmetic/controlled_addition.py`, its original tests and direct Add, And,
Cast and resolver dependencies are covered by six frozen Git blobs in
`ftqc-reference-env/g2-cadd-constructor-intake-20261123.json`.
The earlier complete source disposition in CONTROLLED_ADDITION_FILE_AUDIT.md
is reused, including classical callbacks, symbols, specialized outer control,
exact adjoints, call models, both examples and documented source defects.

Compiler `formats/legacy_arithmetic_factories.py` now resolves CAdd through
`registers/arithmetic/controlled_addition_component.py`. The existing arithmetic
catalogue and Ecosystem composition retain their interfaces. No foreign object
enters Compiler. Add, conjunction, Cast, Split and Join keep their existing
owners. The source identifiers remain serialization metadata.

The signature is ctrl/a/b, with omitted b retaining a's descriptor. All words
are MSB first, both control polarities are retained, and a's width cannot exceed
b's. Unsigned, signed, Montgomery and mixed integer words retain dtype metadata
and caller-owned outputs. A symbolic signature is a logical declaration without
a physical body. Private work comprises m loaded conjunction bits and n-1 Add
carries, released after their measured erasure. Each measured bit has a distinct
feedback slot.

## Constructor and dependency boundaries

The original cv validator accepts binary numeric values. Source serialization
converts bool to integer and retains float values 0.0/1.0. Those exact values
are checked before normalization at the constructor boundary. The direct
And dependency uses the same rule. Numerical execution continues to receive
integer controls. Nonbinary, nonfinite and string metadata refuse.

Named source libraries at retained depths, directly decomposed libraries,
default arguments and malformed metadata are covered. Complete original
examples include both the three-bit and thousand-bit constructors. The latter
is lowered and counted without dense simulation. Existing accepted exact-adjoint
and specialized outer-control evidence is reused without claiming generic
serialized Adjoint/Controlled/Power closure.

## Complex execution and separate resource accounts

Named and directly decomposed libraries are compared with untouched source
primitive dependencies, operands and measured feedback. Reversed physical word
placement, unequal widths and both polarities are covered. Every small raw
measurement history preserves the complete unnormalized complex amplitude
against an independent source tensor, including an entangled spectator and
zero private work. Both complex64 and complex128 trajectories run through
ordinary circuit.append(op). Existing complete inverse-history and all-category
resource evidence in the component audit is reused.

| Quantity | Measured primitive inventory | Independent source model |
|---|---|---|
| T and T-dagger | 4(m+n-1) combined | QEC total T count 4(m+n-1) |
| Measurement and reset | m+n-1 of each | QEC measurements m+n-1 |
| Negative-control X | 4m additional X | Source category model queried independently |
| Clean private work | m+n-1 wires | QubitCount agrees for tested small widths |
| Thousand-bit example | 7996 T/T-dagger and 1999 measurements/resets; peak 4000 | Original 1999 Toffoli equivalents; source peak recursion failure retained |

## Preserved defects and remaining scope

Narrow signed CAdd physically zero-extends its loaded a through Add. Its
classical callback can disagree with that circuit. The actual source circuit
is retained. The original full drawing method has an incompatible wire_symbol
signature. Its thousand-bit QubitCount recursively fails at the ordinary
recursion limit. These original failures remain in the complete component audit.
Symbolic physical execution, generic serialized wrappers, pending Proposals
116-119, whole-repository verification and GPU/performance coverage remain open.
The source is unchanged.

This group brings complete built-in constructor coverage to 41/303 identifiers.
The remaining 262 identifiers, including partial OnEach and generic wrappers,
keep Q-0876 open. Batch verification receipts use
`ftqc-reference-env/g2-cadd-constructor-*`.

## Batch verification

Concentrated direct-dependency regression passes 1638 cases. Affected local
integration passes 145 cases. The coverage selection passes 218 cases with
100 percent statements and branches across both affected constructor owners.
All four changed Python files pass strict typing, Black and Ruff. Compiler
scope, language, architecture, multilevel IR and protected API checks pass.
The dependency checker retains the identical 720 baseline findings and exits
one. This is a recorded outstanding repository condition.

The unchanged original test file passes 169 cases, with its one notebook
separately passing. Six source blobs and native Kickmix 0.1 Simulator are
verified. Original protobuf bool deprecations remain visible. Full-repository
and accelerator checks are outside this batch's measured evidence.
