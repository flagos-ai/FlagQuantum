# FQ-FT-EXECUTION-1131: resource-constrained execution analysis

Status: internal implementation for review; public API approval pending.
Owner: Algorithms, with Compiler owning all program scheduling facts.
Issue: https://github.com/flagos-ai/FlagQuantum/issues/1131

## User journey and boundary

A caller supplies a decomposed logical Clifford+T program and a complete machine
assumption record to compare completion time under finite magic-state storage
and production. The existing logical-layer tally cannot answer that question;
it must retain its meaning and serialized fields. The implementation composes
the existing IR, compiler dependency schedule and logical-resource report.

The implementation is private at
`flagquantum.algorithms._fault_tolerant_execution`. `_MachineAssumptions` and
`_estimate_execution(program, machine)` are internal review entry points, not
public/stable APIs. There is no addition to `fq`, `flagquantum.algorithms`
exports, protected snapshots, compiler contracts, code registries or execution
backends. No API approval is asserted by this document.

## Proposed future public surface

If approved, the domain namespace would expose a machine-assumption input and
an execution-estimation entry rather than extending `LogicalResourceReport` or
overloading runtime `run`/`plan`. Exact public names, signatures, result fields
and serialization must receive API-owner naming and compatibility review before
promotion. Internal JSON is for deterministic source-checkout replay and is not
a versioned public schema. No migration or deprecation of an existing API is
needed. Public promotion would require its own release note, behavior contract
and approved machine-readable API delta; this change does not update snapshots.

## Alternatives and subtraction review

- Extending the old logical report changes the meaning of a protected workflow
  and would mix a dependency-depth tally with feasible execution timing.
- A new scheduler IR or code registry duplicates existing authoritative records.
- A compiler contract change is unnecessary: `CircuitSchedule` already provides
  source indices and predecessor edges. Timing records only annotate those indices.
- General placement, heterogeneous factory types and calibrated error models
  exceed the first verified requirement and are excluded explicitly.

No production dependency or cross-domain implementation change is required.
The module owns only bounded machine assumption validation, resource allocation,
and result evidence. See the [guide](../guides/FAULT_TOLERANT_EXECUTION.md) for
the conventions, executable replay and limitations.

## Acceptance evidence

The focused tests cover hand-solvable unlimited-supply timing, a factory
bottleneck, storage overflow and serialization, capacity infeasibility, explicit
durations/startup, same-tick ordering, independent work during stalls, preserved
wire dependencies, unchanged logical reports, unavailable failure estimates,
unsupported semantics, and byte-for-byte JSON replay. Broader checks and their
results must be reported from the actual implementation revision.
