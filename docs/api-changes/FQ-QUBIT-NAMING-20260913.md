# Qubit terminology migration

Status: implementation candidate; protected signatures and schema changes await API-owner review.

The user explicitly requested gradual retirement of wire terminology and alignment
between executable code and documentation on 2026-09-13. This authorizes preparing
this proposal and PR; it does not record a completed API-owner review.

## Problem and first delivery

New users encounter `n_qubits` alongside `wires` and `observable_wires`. Documentation
must not invent unsupported keyword replacements, but should teach one vocabulary.
This PR renames the quick-start indices and implements the public keyword aliases
and policy schema reader/writer described below. Frozen API snapshots remain
unchanged pending review.

## Proposed public spelling

| Current | Recommended replacement | Compatibility treatment |
| --- | --- | --- |
| Circuit(n_wires=...) / Circuit(nqubits=...) | Circuit(n_qubits=...) | Retain aliases during the published migration window |
| probabilities(wires=...) | probabilities(qubits=...) | Preserve positional selection; deprecated keyword alias |
| samples(wires=...) / counts(wires=...) | samples(qubits=...) / counts(qubits=...) | Preserve Observable overload and positional selection |
| RuntimePolicy(observable_wires=...) | RuntimePolicy(observable_qubits=...) | Constructor alias plus explicit old-payload reader |
| Example local variable wire | qubit | Rename now; no compatibility obligation |

The implementation uses an OMITTED sentinel to distinguish omitted arguments
from explicit None. Contract updates remain a separate review step. Both old and new keywords together must
raise an actionable TypeError; do not guess precedence. Preserve validation of
indices, batch semantics, output ordering, observable selection and gradients.

## Serialized artifacts

RuntimePolicy is serialized in module configuration/checkpoints. Replacing its
field name is not a cosmetic rename. Introduce an explicitly versioned writer
and keep an old-payload reader. Existing payloads containing observable_wires
must retain their meaning. Reject conflicting duplicate fields. Preserve
historical fixtures rather than regenerating them to hide the transition.

This proposal does not rename IR fields, backend-native adapter vocabulary,
plan identity inputs, or every internal occurrence. Those require a separately
inventoried contract migration; a global string replacement is unsafe.

## Schedule and ownership

Owner: integration/API maintainers, with core and runtime implementation owners.
The first deprecation release and removal release must be selected by the release
owner before implementation approval. No calendar date or unapproved release
number is asserted here. Removal must not precede the published compatibility
window required by RELEASE_POLICY.md. A warning must identify the supported
replacement and approved removal release. Compatibility adapters end when that
release and the migration acceptance criteria are met; they are not permanent.

## Documentation alignment

Document the actual reviewed source revision. New examples use qubit terminology;
existing keyword spellings remain explicitly documented until their replacement
is implemented. Once replacement support lands, migrate example code and API
reference in the same change. Keep a migration page for old spellings.

Documentation validation should check the signatures it describes against the
selected source checkout and execute its local examples. A signature difference
must require editorial review, not automatic snapshot regeneration. Remote jobs
are outside the local verification command.

## Required implementation evidence

- Old/new calls return equal probability, expectation, counts and sample results.
- Positional calls remain compatible; generator selections are consumed once.
- Duplicate keywords and invalid indices fail deterministically.
- Deprecated calls warn with replacement, version and a caller-facing stacklevel.
- Old serialized policy/checkpoint fixtures load with unchanged readout semantics.
- New writer round-trips; contradictory fields and unknown versions fail closed.
- Batched hybrid forward values and classical/quantum gradients remain unchanged.
- Approved API contracts, release notes and documentation change together.

## Alternatives

Immediate removal breaks existing notebooks and configuration files. Keeping two
canonical spellings indefinitely creates documentation and maintenance debt.
Renaming only local variables is safe but incomplete; that is why this PR records
the follow-on API and serialization work explicitly.

## Implementation candidate (user continuation)

The user explicitly requested implementation after reviewing the migration steps.
The branch now contains selection aliases, policy schema v2 with an old reader,
constructor deprecations and compatibility/gradient/state-loading tests. API-owner
review remains pending. Proposed release targets are 0.3.x deprecation and 0.4.0
removal; no version bump or frozen-contract regeneration is included. See
`docs/reference/QUBIT_NAMING_MIGRATION.md` for implemented behavior.
