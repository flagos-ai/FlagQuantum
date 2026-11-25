# Stored phase-gradient rotation constructor audit

## Scope and ownership

Complete locked `rotations/rz_via_phase_gradient.py` and
`rotations/zpow_via_phase_gradient.py` constructor and direct cast,
controlled-add/subtract, controlled-XOR and fractional-add dependency disposition.
Frozen Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389` remains unmodified.
Compiler format adapters own constructor metadata. Register phase components own
wiring and invoke existing arithmetic and layout owners. Numerical execution
continues through existing runtime entry points. Path classification is
`single_device_fast_path`.

The explicit `phase_rotation_factories()` mapping is supplied to the existing
stored-library resolver. The resolved Program supports the existing
`circuit.append(op)` workflow demonstrated in
`examples/algorithms/phase_gradient_rotation_decomposition.py` and
`tests/unit/test_phase_gradient_rotation_decomposition.py`.

## Interface and logical contracts

Rz retains `angle_dtype` and `phasegrad_dtype`, with THRU ports `q`, `angle`,
`phase_grad` in original order. All word views remain MSB first. Four Cast calls
surround the shared controlled-add/subtract call. Symbolic widths retain all five
root calls with typed unresolved children. Signed fixed-point Cast declarations
retain the same root and now bind through approved Proposal 116 raw storage. The native
exact adjoint exchanges the arithmetic control sectors.

Z power retains `exponent` and `phase_grad_bitsize`, with THRU `q`, `phase_grad`.
Its five root operations allocate a clean angle, conditionally load exponent/2,
add into the gradient, unload and free the angle. The shared addition owns carry
measurement, feedback and erasure. Fractional encoding uses original truncation
and wrap. Concrete exponent with symbolic precision retains the source root.
Symbolic exponent retains its original decomposition refusal and declaration.
No new execution or public API contract is introduced.

## Evidence

`test_legacy_phase_rotation_factories.py` compares serialized depths 0/1/2,
source call operands, reverse wire bindings, exact accepted primitive streams,
events and output ownership. Small full complex operators use original tensor
contraction. The accepted decomposition/file tests are rerun for all small
measurement histories, spectator phases, exact inverse, append, zero work and
40-bit structural/arithmetic certificates.

Frozen file checks and actual Kickmix Simulator checks remain in the accepted
file tests. Resource inventories are recorded separately below after regression.

## Retained boundaries

Both original empty roots fail because zero-width ports are omitted. Declarations
remain serializable, with physical binding refused. Negative concrete Z exponent
is refused by the original unsigned fixed-point encoding and native encoder.
Symbolic Z exponent cannot decompose upstream. Symbolic dimensions cannot bind
physical wires. Signed positive-width fixed-point Rz now retains executable Cast leaves under
approved Proposal 116. Symbolic and empty root boundaries remain explicit. Z power constructor closure is complete
within its original decomposition domain. Source analytical models omit casts
and differ from actual measured primitive decompositions. Original model and
physical counts remain independently labeled.

## Acceptance receipt, 2026-11-23

The complete affected regression passes 68 tests. Local affected integration
passes 145 tests. Strict mypy checks all six touched Python files. Black, Ruff,
team scope, repository language, architecture, multi-level IR and public API
checks pass. Dependency policy retains the identical pre-existing 720 findings.
Both file suites verify frozen Git blob identities and the actual Kickmix 0.1
Simulator. Existing source branches, inverse and 40-bit execution certificates
are reused through exact primitive/event/output equality.

Unmodified upstream tests produce 25 passes, 3 skips and 1 expected failure.
Both notebook tests fail while Jupyter requests a socket forbidden by this
sandbox. This environment failure is retained in
`g2-phase-rotation-original-20261123.log` outside the repository. The slow
seven-bit original tensor case passes in this run.

Four-bit Rz actual inventory is 12 T/Tdag, 41 CX, 3 measurements/resets on 12
physical wires. Its independent source model has 3 And, 28 Clifford and 3
measurements. Forty-bit Z power actual inventory is 156 T/Tdag, 469 CX and 39
measurements/resets on 120 physical wires. Its independent original model has
38 Toffoli and 4 Clifford. Model classifications and physical primitives use
different accounting conventions and remain distinct evidence columns.

Concrete negative exponent with symbolic width succeeds in the original root
because XorK retains the unencoded exponent/2. The native symbolic root now
preserves that domain. Concrete width still invokes the existing unsigned
encoder and rejects negative values.


## Approved signed storage closure, 2026-11-25

The complete Rz source root first reinterprets both data words as unsigned raw
storage, calls ControlledAddOrSubtract, and restores their nominal descriptors.
The stale component-local signed guard has been removed. All four Cast bodies
reuse existing split/join views and the approved Core admission; no signed
arithmetic, signed controls or classical casts are added. Compiler owns only
composition. User code continues to append the bound operation to a circuit.

Signed angle only, signed gradient only and both signed are checked at positive
widths two and three, with negative and positive fractional widths. Actual
source serialized depths zero, one and two resolve identically, retaining the
five root calls, nominal types and original operand identities. Physical output
aliases, primitive instructions, feedback and events equal the previously
accepted unsigned raw-storage path. The complete original signed and unsigned
operators agree without phase alignment. Independent source QEC categories and
peak counts agree; the physical inventory is checked separately.

A complex input entangled with a spectator is compared to the original tensor
on every raw forward history. Every paired inverse history restores all complex
amplitudes and all carry work to zero. Ordinary append and IR JSON roundtrip
are checked. Symbolic roots still retain unresolved declarations and fail finite
binding. Signed-zero storage remains outside this change (Proposal 124).
No locked source or protected API vocabulary changes.

The batch passes 120 concentrated cases, 40 Cirq 1.6 cases and 145 local
integration cases. Strict typing covers three Python files; Black, Ruff,
Compiler scope, language, links, architecture, IR and protected API checks pass.
Dependency policy retains 720 diagnostics: the only textual change from the
previous QFT baseline is an already-existing specialized-control NumPy import
moving from line eight to line nine in the preceding published batch. There
are no added diagnostics. Receipts use `g2-signed-rz-*20261125` in the reference
workspace. The original complete-file blob checks and Kickmix checks run in
the affected source suite. Protobuf bool-argument deprecation warnings belong
to the unchanged source serializer and remain visible.

| Signed storage example | Actual primitive inventory | Original independent QEC model |
| --- | --- | --- |
| Width two | 4 T/Tdag, 13 CX, 3 H, 2 X, 1 S, 1 measurement/CZ/reset; peak 6 | 1 And, 10 Clifford, 1 measurement; peak 6 |
| Width three | 8 T/Tdag, 27 CX, 6 H, 2 X, 2 S, 2 measurements/CZ/resets; peak 9 | 2 And, 19 Clifford, 2 measurements; peak 9 |

The nine resource rows retain angle-only, gradient-only and both-signed nominal
profiles. These are storage views, so signed bit patterns do not change the
primitive stream or original model. Whole-file parents Q-0685 and Q-0688 retain
their existing original empty/symbolic/model boundaries; no parent acceptance
is inferred from this dependency fix.
