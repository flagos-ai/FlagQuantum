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
retain the same root while physical storage awaits Proposal 116. The native
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
physical wires. Signed fixed-point Rz retains a root but its Cast leaves remain
unexecutable pending Proposal 116. This constructor is partial in the catalogue
until that physical boundary closes. Z power constructor closure is complete
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
