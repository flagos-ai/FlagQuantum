# Owned symbolic scalar dependency audit

This dependency batch supports the open complete-source audit Q-0156 for frozen
Qualtran v0.7.0, commit bcb17bb25360ed68ce3bc084645db3e929799389.
The complete parent remains open for wider symbolic expression transport.
The reference sources remain byte-identical to the frozen inventory and GitHub
Git blobs. The environment receipt is g2-owned-scalar-environment.json.

## Ownership and executable interface

Core Program constants retain the existing immutable Parameter and
ParameterExpression representation. Core validates names, immutable operands,
acyclic trees, operation arity and finite numeric leaves. The supported expression
operations are add, sub, mul and neg. Physical numeric inputs retain float
normalization and existing constant hashes. Program verification and canonical
payload comparison reject malformed or altered literals.

Compiler reconstructs owned scalar parameters when binding nested typed calls.
Ecosystem converts symbolic SDK input immediately into owned expressions,
including the real NumberSymbol pi. Exact zero-shift ZPow and CZPow become phase
and cphase. Controlled Rx, Ry and Rz use existing controlled rotation opcodes.
The private register export path admits those leaves for lossless reimport.
Each operand retains its ordered bit position. Measurement conditions retain
native classical slots. This adds no foreign object to Program and no public
expression operations or schema fields.

Users append the bound operation through circuit.append(op), then use the
existing circuit.bind_parameters(values) before numerical execution. Tests use
the existing FieldOperation adapter. Unbound numerical execution reports its
existing named-parameter error. Explicit clean workspace is caller-owned and is
returned through the existing allocation and release events.

## Source and execution evidence

Nine SDK atoms cover Rx, Ry, Rz, ZZ rotation, phase, controlled phase and the
three controlled rotations. The unchanged source graph preserves one immediate
call and reversed operand placement for each input atom. Native import, graph
copy, binding, export and reimport preserve the same leaf boundary. Two parameter
assignments compare complete complex amplitudes on entangled spectator inputs.
No comparison removes an overall phase.

The frozen CZPow and CRz native Bloq wrappers omit the Cirq parameter protocol.
Their exported wrapper reports is_parameterized false even while retaining
symbolic fields. Numerical source comparison therefore resolves input SDK values
before constructing a fresh unchanged source graph. Frozen CZPow export expands
into measured And erasure and cannot supply Circuit.unitary. Its unchanged source
tensor contraction supplies the full exact operator instead. Other resolved
source exports supply their unitary directly. These are distinct source protocols
and are recorded in the tests.

Measured fixtures cover Rz, phase, controlled phase and controlled Rx, both
measurement outcomes, conditional Y feedback and reset. Native export and
reimport match independently computed unnormalized source complex branch
amplitudes. Every branch returns the record bit to zero. The nested-call fixture
returns its clean work bit to zero, preserves its observable scalar phase and
matches the analytic complex state and parameter gradient.

## Resources and boundaries

| Case | Emitted primitive evidence | Independent source model |
| --- | --- | --- |
| One imported SDK atom | One matching native opcode, zero work, zero measurement | No independent QEC estimate supplied by this fixture |
| Nested owned rotations | Two Rz atoms, one clean work bit, zero measurement | No independent QEC estimate supplied |
| Measured symbolic feedback | H, measurement, selected atom, Y, reset, one record | No independent QEC estimate supplied |
| Source CZPow graph export | Measured And decomposition at a deeper boundary | Distinct from the input SDK atom count |

The source accepts nonlinear symbolic Rz(sin(theta)) graph transport. Native
conversion still rejects sin, variable powers and reciprocal parameters with
explicit diagnostics. Generic symbolic global shifts remain unsupported.
Q-0156 remains open for these wider expression protocols. Source symbolic
register element counts are rejected by both implementations and are not an
outstanding native gap. Q-0149 and Q-0154 have already passed their complete
source-file acceptance independently.

## Local acceptance

The concentrated register, graph, scalar, channel and SDK regression passes
1186 cases without skips. The Cirq integration regression passes 96 cases with
two existing deprecation warnings. Ten touched Python files pass strict typing,
Black, Ruff and ownership checks. Architecture and internal IR checks pass.
The dependency scan has 689 whole-repository findings and zero on this batch.
Default smoke/unit with first-failure stopping reproduces the existing
public-docstring example registry failure after 434 passes and 51 skips.
The log is g2-owned-scalar-default-first-failure.log. The earlier unrestricted
run was interrupted after the same failure while it continued into unrelated
QEC cases. Neither run establishes a passing whole-repository result.

The implementation commits are Core 244e0f780, Compiler 27154329a and Ecosystem
ad34f3b8c, preceded by contract f88a890fa.
