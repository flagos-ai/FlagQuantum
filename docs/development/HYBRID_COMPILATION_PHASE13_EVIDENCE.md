# Phase 13 bounded measurement-predicate evidence

Date: 2026-09-09

## Accepted profiles

The private hybrid compiler now represents general predicates over previously
assigned measurement booleans as bounded canonical DNF. This includes
disjunction, negated conjunction, measurement equality/inequality, and exact
quantum work in both branches. Simple conjunctions keep the existing runtime
metadata representation.

## Evidence

| Gate | Result |
| --- | --- |
| Program IR verifies `arith.or` as bool, bool -> bool | pass |
| `first or second` emits two canonical single-literal clauses | pass |
| Reference and batched execution implement disjunction shot by shot | pass |
| `not (first and second)` emits the exact De Morgan complement | pass |
| A quantum else branch receives the complementary conjunction | pass |
| Measurement equality emits canonical XNOR clauses | pass |
| Measurement inequality emits canonical XOR clauses | pass |
| Subsumed clauses reduce to the weaker single conjunction | pass |
| Clause expansion beyond the configured ceiling fails before Runtime | pass |
| Malformed and nonminimal DNF metadata is rejected | pass |
| Existing single-conjunction `conditions` behavior remains intact | pass |
| Provider dialects fail closed on unrecognized complex clauses | pass |

## Claim boundary

This evidence covers bounded Boolean predicates over already materialized
measurement bits in the private local dynamic-session path. It does not cover
inline or conditional measurement, measurement-dependent SSA value merging,
finite-shot gradients, provider execution of complex predicates, accelerators,
distributed execution, capacity, or performance.
