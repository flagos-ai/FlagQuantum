# ARCH-008: Migration Decisions and Verification Paths

Status: Proposed

Date: 2026-09-03

Basis: Phase 0 inventory of duplicate types and compatibility debt; no authorization for migration, retirement, or public API changes.

## Context

Not every duplicate type warrants immediate migration. Not every local optimization
needs to cross Core, Compiler, Runtime, Simulation, and Provider. Compatibility
without costs or deadlines becomes permanent duplication; requiring full
certification for every change slows local CPU, single-device, and kernel work.

## Decision Candidates

Before acting on a cross-layer or legacy item, record its current/target owner,
callers and authoritative facts, user/architecture value, ongoing maintenance cost,
one-time migration cost, regression risk, impact of retaining the implementation,
compatibility commitments, review date, target version, test evidence, and exit
conditions. Only these dispositions are permitted:

- `migrate`: move to the new authority, with a compatibility plan, replacement
  tests, and zero remaining callers of the old implementation;
- `adapt`: retain the implementation behind a narrow adapter to the authoritative
  contract, documenting information loss, ownership, and exit/review conditions;
- `freeze_legacy`: retain the implementation but prohibit new callers, features,
  and schema expansion. Require an owner, characterization/no-new-caller tests,
  an explicit review date, and a fresh disposition decision at review;
- `retire`: remove only after a replacement path, compatibility window, zero
  callers, and approved removal tests are all in place.

`freeze_legacy` is not indefinite compatibility. An overdue review blocks new
features and release promotion that depend on the item; renewal is not automatic.

### Three Verification Paths

1. **Fast path**: internal changes or local kernel/performance optimizations under
   existing semantics. These do not change public semantics, cross-layer
   contracts, capability ceilings, or fallback. The owner runs scope/architecture
   checks and minimal numerical, gradient, and performance regression tests;
   crossing all five layers is not mandatory.
2. **Standard path**: new or changed cross-layer information, internal versioned
   contracts, adapters, or provider behavior. Requires an ADR/contract fake,
   consumer replacement, compatibility, and relevant integration tests. Only
   changes to public semantics enter the Core/API change process.
3. **Certification path**: hardware, distributed scalability, production,
   performance, or QPU claims. In addition to the standard path, run real targets,
   corresponding GPU/multinode/Provider tests, evidence audits, and release gates.
   Interfaces or mocks cannot substitute for these checks.

## Prohibited Practices

- Do not hide permanent duplication without ownership, cost assessment, or review
  dates behind temporary compatibility.
- Moving files, changing imports, and updating snapshots do not replace
  replacement/compatibility evidence.
- A new internal kernel does not raise public capability. An implementation change
  without semantic effects need not modify Core.
- The fast path cannot bypass existing public API, fail-closed, fallback, or
  numerical correctness gates.

## Compatibility

This ADR defines decision records and verification intensity only. It does not
change deprecation, Stable Core, or release policy. A `migrate`/`retire` action
involving stable exports, signatures, behavior, exceptions, or serialization still
requires an API Change Proposal; ADR approval alone is insufficient. Existing
legacy implementations remain unchanged until individually recorded and approved.

## Migration Sequence

1. Add the required fields and dispositions to Phase 0 D1-D12 and the
   Runtime-to-Compiler inventory.
2. Address valuable adapters and naming conflicts with low regression risk before
   migrating identities and cross-layer authorities.
3. Migrate one vertical slice at a time. After verification, update caller counts,
   maintenance costs, and review dates.
4. Review frozen items when due; remove retired items in the approved order after
   their compatibility windows.

## Acceptance Tests

- Decision-record schemas require all six assessment categories, owner, review
  date, tests, and exit conditions.
- Frozen items gain no importers/callers/exports/features; characterization holds.
- Fake and real implementations for migrate/adapt pass identical conformance
  tests; consumers need no code changes to replace them.
- The fast path proves public contracts/capabilities unchanged; the certification
  path fails closed without real evidence.
- CI identifies overdue freezes, adapters without owners, and retirement actions
  whose exit conditions have not been met.

## Open Questions

- TOML/JSON location for the migration ledger, schema ownership, and the severity
  of overdue-review blocks.
- Whether value/cost/risk use enums, scores, or justified prose.
- Fast-path performance regression thresholds and when observable behavior changes
  automatically require the standard path.
