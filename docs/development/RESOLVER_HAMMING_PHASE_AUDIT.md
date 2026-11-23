# Stored Hamming phase source constructor audit

## Scope and interface

Complete frozen `rotations/hamming_weight_phasing.py` contains
`HammingWeightPhasing` and `HammingWeightPhasingViaPhaseGradient`. Both retain
`bitsize`, default `exponent=1`, and default `eps=1e-10`. The direct variant has
an MSB-first unsigned THRU `x`. The gradient variant adds the caller-owned
fractional THRU `phase_grad` selected by the existing QVR sizing owner.

The format module resolves serialized metadata into existing register Programs.
The shared phase owner controls count, phase and exact count erasure. The count
owner retains clean junk and output lifetimes. Existing Runtime binding and
`circuit.append(op)` remain the execution path. Path classification is
`single_device_fast_path`. Frozen Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389` remains untouched.

## Direct dependency closure

The direct root retains count, Split, source LSB-first weighted Z powers, Join,
and exact measured erasure. Symbolic exponents bind through existing Parameter
expressions. Direct synthesis precision remains original model metadata and
adds no physical gate. Symbolic precision therefore retains the same gate body.

The gradient root retains three calls, count, QVR and count erasure. Shared QVR
retains its scaled-addition child when that child's original exact-whole-word
lookup refuses mixed fractional descriptors. The predicate uses the existing
source term schedule and first full input slice. This preserves successful
parent graphs and the original child failure without catching unrelated errors.
Symbolic exponent retains the original finite-width three-call root with a
QVR declaration. Existing source sizing model supplies its exact gradient width.

## Boundaries and evidence

Original gradient precision symbols fail while constructing the fixed-point
signature. Symbolic input widths fail the original bit-length operation.
Zero input direct roots fail at the omitted x port. Source one-bit graphs build,
but original physical SDK lowering fails on empty junk. These remain separate
logical and physical boundaries.

Concrete mixed-format scaled child failures, original finite-precision error
counterexamples, source aggregate model differences and unresolved symbolic
physical binding remain explicit. Original examples and depths 0/1/2 retain
source root call operands. Exact primitive, event and output equality reuses
accepted complex amplitude, all small measurement branch, spectator, inverse,
append and work-zero evidence. Actual primitives and source models are reported
separately in the final acceptance receipt.

The empty gradient source signature retains a one-bit gradient before the root
fails on its omitted input. Shared numerical sizing now admits the zero-width
logical cost case and reuses the same truncation and binary search. This closes
the empty constructor declaration without physical empty-word admission.
One-bit logical roots of both families are compared even though upstream
physical lowering rejects empty junk.

## Acceptance receipt, 2026-11-23

Concentrated regression covers 189 cases across constructors, Hamming, QVR and
source binding. Local affected integration passes 145 cases. Unmodified source
Hamming tests pass 17 cases with 8 skips, 4 expected failures and 15 deliberately
deselected slow/large cases. Existing full native sparse/arithmetic certificates
cover the larger original cases. Seven frozen blobs and actual Kickmix 0.1
Simulator admission are checked by the file fixtures.

Strict mypy checks eight touched Python files. Black, Ruff, team scope,
repository language, architecture, multi-level IR and public API checks pass.
The dependency checker retains the same 720 pre-existing findings.

| Four-bit original example | Actual emitted primitives | Independent original model |
| --- | --- | --- |
| Direct Hamming phase | 12 T/Tdag, 50 CX, 3 phase, 3 measurement/reset | 3 And, 35 Clifford, 3 rotations, 3 measurements, 10 logical wires |
| Gradient Hamming phase | 1108 T/Tdag, 2592 CX, 0 phase, 277 measurement/reset | 561 Toffoli, 3 And, 35 Clifford, 3 measurements; original qubit-count query refuses |

The primitive and model columns retain their different original accounting
conventions. The original finite-precision negative-exponent counterexample and
mixed-format scaled-child refusal remain explicitly tested. Both constructor
identifiers close within the original declaration and decomposition domains.
