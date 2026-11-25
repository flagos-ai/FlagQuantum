# Exact phase read witnesses

Fresh native acceptance exposes five unchanged QAOA programs exceeding the
existing 4096-node model limit. Their models contain 576 to 2688 scalar
zero-angle phase effects emitted solely to witness expression reads. Every
read is currently witnessed again even when an emitted phase monomial already
requires the same initialized quantum location. An isolated in-memory probe
passes the two-binding complete-state comparisons for the first three programs;
the two 24-bit resiliency programs still require bounded full-state validation.
This is exploratory evidence, not production or full-corpus acceptance.

## Contract

Within one existing exact polynomial/factored phase capture, an emitted phase
monomial may also witness the source reads at its selected canonical quantum
locations. Omit a separate zero-angle read witness only for locations selected
by an actually emitted, checked monomial. Retain the witness for every other
source read, including reads removed by algebraic cancellation. An emitted
zero-angle monomial still undergoes the ordinary ownership and initialization
checks; it is not permission to discard its operands before verification.

This changes only redundant internal effect representation. Keep the existing
IR, verifier, phase grammar, numeric formats and source identities. Do not infer
initialization from a preceding helper, assume a clean workspace, move reads
across statements, discard unsupported operands, or use post-verification gate
cancellation as a substitute for verifying the source reads. Factor scratch,
Const, alias, inverse, within/apply, callback and Output-escape checks remain.
Retain every source, expansion, model-node, phase-degree, term, workspace and
operation ceiling. No parameters, layers or original data may change.

## Acceptance

- Compare complete complex amplitudes, absolute phase and clean work against
  independent Circuits for root, borrowed and controlled phases, including
  mixed retained/canceled reads and a zero coefficient.
- Reject uninitialized or unsupported canceled reads, Const mutation and
  overlapping helper actuals. Keep signed/fractional fallback and existing
  phase/predicate bounds; do not weaken a refusal to fit the optimization.
- Demonstrate a generic repeated phase program fitting the unchanged node
  budget through representation reduction, with actual final gates counted.
- Run the five unchanged native programs with both existing nonzero bindings:
  traveling salesman, vehicle routing, maximum induced color subgraph, and
  both resiliency examples. Match independent Circuit full states, declared
  bit order, absolute phase and zero work. Timeouts remain failures.
- Recheck numeric views, bounded callbacks, QPE, arithmetic phase and
  amplification regressions. Replay all 216 original sources and six approved
  revisions, retaining immutable original resource baselines and explaining any
  actual resource change. Run the default tier and archive the exact measured
  sources before crediting the batch. Full native216 acceptance remains a
  separate gate; no integration PR or remote full CI before that gate.
