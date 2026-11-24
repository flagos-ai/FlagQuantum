# Native definition helper ownership audit

## Scope and interface

The existing Cirq register import boundary accepts native FunctionRecord
implementations through `gate_definitions` and attached definitions. This batch
closes their transitive helper ownership across the pure typed-call path and
shared measurement-history path. It reuses the existing mapping and ProgramRecord
representations. Classification is `single_device_fast_path`.

`ecosystem/cirq/_register_definition.py` collects called helper definitions by
symbol name. Selected roots retain their existing ownership and name-conflict
checks. Only reachable helpers join the executable module. Unused declarations
remain outside the module. Missing or conflicting reachable helpers raise an
explicit ValueError before constructing an incomplete Program.

`ecosystem/cirq/_register_import.py` supplies the complete resolved module to its
existing ComponentProgramBuilder. The measurement-history owner uses the same
collector before binding native definitions and isolating their local keys.
Runtime and Simulation continue to own execution.

## Reproduced failure and exact replacement

A named native CZPow Program was supplied as the implementation of an SDK
CZPowGate. Its complete compute, phase and erasure helper definitions were also
in the existing mapping. The pure import path previously retained only the root,
then failed verification with unknown `compute`, `z_power` and `erase` callees.
The same input now imports the reachable module and binds identically to the
standalone accepted native CZPow Program.

The source-compatible two-QBit signature is THRU and MSB ordered. The existing
And component owns a clean work QBit, measurement feedback and exact cleanup.
The reconstructed gate has the full measured And/phase/erasure stream. Its
emitted primitives, feedback slots, allocations, releases and outputs match the
existing native owner exactly.

A second workflow combines that module with an external measurement. All four
external/erasure histories compare raw complex amplitudes against the exact
conditional channel, and private work is zero in every branch. An unrelated
bodyless declaration is ignored in both paths. Missing and conflicting reachable
helpers have explicit regression cases.

## Remaining integration

This fixes helper ownership through the existing interface. Serialized gate
payloads still use their SDK import path until the named gate conversion table
links the existing native gate owners. That linkage remains the next fidelity
step. The original JSON-expression, arbitrary controlled tensor and large-state
Cirq compatibility boundaries remain separately recorded. Protected public APIs
and serialized vocabularies retain their current contracts.

Explicit native definitions also retain precedence before SDK eigenphase and
control normalization. An identity override of an SDK gate with a nonzero
`global_shift` stays an identity, including in a measured outer circuit. The
SDK's original phase is excluded by the supplied definition's authority.

Default `main` entry names import through the existing component namespace
owner. The importer calls the complete Program instead of defining a second
`main` alongside its own caller. Original definitions remain immutable and
physical output matches standalone binding.

## Acceptance receipt, 2026-11-24

The concentrated regression passes 135 cases without skips. Related local
integration passes 145 cases. Strict mypy checks the three touched Python files.
Black, Ruff, Ecosystem scope, language, architecture, multi-level IR and API
checks pass. Dependency analysis exactly matches the 720 existing findings.
The four frozen interop blobs, Qualtran 0.7.0 and actual Kickmix 0.1 Simulator
are verified by the unchanged reference fixture. The ten original interop tests
accepted in the immediately preceding batch are reused with the same reference
files and environment.

The imported named CZPow emits three H, two T, two Tdag, six CX, one S, one
phase, one measurement, one conditional CZ and one reset on three used wires.
One work allocation and release match standalone native binding exactly.
The separate original model is one And, one Clifford, one rotation, one
measurement and three logical wires. This explicitly supplied named definition
contains the complete And/phase/erasure body. The serialized payload path still
needs its named-gate conversion table to select that definition automatically.
