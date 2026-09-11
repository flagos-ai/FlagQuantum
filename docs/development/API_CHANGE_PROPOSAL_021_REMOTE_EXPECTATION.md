# API Change Proposal 021: Remote expectation outputs

## Status

Approved for the first public alpha on 2026-09-09 by explicit repository-owner
direction. This proposal extends Proposal 020 without changing a public
signature or adding a public type.

## Problem

`fq.expectation(...)` works locally, while the Quafu path accepts only a
full-register `fq.counts()` request. Users must currently leave the root API,
construct grouped measurement packages, submit every package, and aggregate
the counts themselves to evaluate a Pauli Hamiltonian on a QPU.

## Decision

- Allow one `fq.expectation(...)` output in a remote Quafu `fq.run` call.
- Compile the source program once with the explicitly named compiler and
  target. Build all measurement groups from that target-bound IR so every
  group preserves the same logical-to-physical mapping.
- Reuse the existing qubit-wise-commuting Pauli grouping, deployment package,
  provider lifecycle, and grouped-count aggregation implementations.
- Express Y-basis measurement using `RZ(-pi/2)` followed by `H`, keeping the
  post-compilation rotation inside the QSteed plugin's supported native gate
  subset.
- Interpret `shots` as shots per measurement group. Return the total submitted
  shots, group count, per-group shots, and an estimator standard error with the
  expectation result.
- Preserve provider task IDs, deployment identities, submission receipts,
  compiler, target, and physical-qubit mapping in result provenance.
- Keep probabilities, samples, mixed output requests, remote gradients, and
  implicit compiler or provider selection unsupported. They fail before
  compilation or submission.

## Acceptance

- `fq.run(circuit, outputs=fq.expectation(observable), compiler="qsteed",
  target="quafu:<backend>", shots=...)` returns the same stable
  `ExecutionResult` type as local execution;
- one compilation produces one or more independently sealed measurement-group
  packages with an unchanged physical-qubit mapping;
- X, Y, Z, identity, weighted products, and sums aggregate correctly;
- the result exposes an estimator standard error and complete per-group task
  and deployment evidence;
- malformed or unsupported remote output requests fail before provider work;
- no compilation, execution, basis, precision, or CPU fallback is implicit.

## Release note

The Quafu path now evaluates one Pauli expectation request through the ordinary
`fq.run(..., outputs=...)` workflow. `shots` is applied to each commuting
measurement group, and execution evidence remains available through the stable
result metadata and the provider-native result tuple.
