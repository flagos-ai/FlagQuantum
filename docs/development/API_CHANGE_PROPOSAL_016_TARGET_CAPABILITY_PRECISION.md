# API Change Proposal 016: Consistent TargetCapabilities Precision Semantics

## Status

**Approved before the first public alpha.** On 2026-09-06, the API owner explicitly
authorized consolidating precision controls. The repository is unreleased, so no
old-value compatibility or migration layer is retained.

## Problem

TargetCapabilities already separated precision fields, but string values mixed
scalar dtypes with complex quantum-state dtypes. For example,
`precision.effective_dtype` used both `float64` and `complex128`, leaving Runtime
unable to determine whether they were synonyms or distinct precision levels.

## Decision

- `native_dtype`, `storage_dtype`, `parameter_dtype`, and `accumulator_dtype`
  accept only currently supported scalar types `float32` and `float64`.
- `effective_dtype` accepts only logical complex state types `complex64` and `complex128`.
- `software_mechanism` remains an independent mechanism field, not a dtype alias.
- With `software_mechanism=none`, native, storage, and effective precision must
  form a consistent `float32/complex64` or `float64/complex128` path.
- Runtime must still validate evidence levels and applicable scope for software
  precision extensions.

## Compatibility

The field set and TargetCapabilities schema version remain unchanged. Previously
ambiguous values have no public compatibility promise and are rejected directly,
without aliases, automatic conversion, or fallback.

## Acceptance

- Core rejects incorrect dtype categories when constructing requirements and facts.
- Core rejects contradictory precision paths without a software mechanism when
  constructing snapshots.
- Double-Single remains expressible as native/storage `float32`, effective `complex128`.
- Core, Platform, Runtime, serialization, and CPU vertical-path tests pass.
