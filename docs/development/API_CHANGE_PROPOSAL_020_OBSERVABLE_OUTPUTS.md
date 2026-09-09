# API Change Proposal 020: Observable and output requests

## Status

Approved for the first public alpha on 2026-09-09 by explicit repository-owner
direction. The API remains pre-freeze until the acceptance suite passes.

## Problem

The root API currently asks users to construct generic `MeasurementNode` values
and spell backend-oriented kinds such as `expectation_ps`. This leaks IR storage
details, makes X/Y/Z observables hard to discover, and gives `ExecutionResult`
no direct accessors for probabilities or counts.

## Decision

- Add the immutable, backend-neutral `Observable` and `OutputRequest` public
  types under `flagquantum.observables`.
- Add root factories `I`, `X`, `Y`, and `Z`. Pauli products use `@`, weighted
  terms use scalar multiplication, and Hamiltonian sums use `+` and `-`.
- Add root output factories `expectation`, `probabilities`, `samples`, and
  `counts`.
- Replace the root `measurements=` keyword with `outputs=` in `fq.run` and
  `fq.plan`. `Circuit.run` and `Circuit.plan` use the same spelling.
- Keep `MeasurementNode` and `ObservableNode` as Core IR implementation types,
  but remove them from the root facade. Output requests lower to existing IR
  measurement nodes before planning; no second execution engine is introduced.
- Add `ExecutionResult.probabilities`, `ExecutionResult.counts`, and
  `ExecutionResult.expectations`. Preserve `expectation(selector=None)` and
  `require_samples()` for explicit selection and validation.
- A Hamiltonian expectation lowers to Pauli-term requests and is recombined by
  the stable result boundary. Identity terms are handled exactly.
- Sampling and counts remain computational-basis requests in this change.
  Observable-basis sampling is not silently approximated and requires a future
  explicit contract.
- Remote execution accepts only outputs the provider workflow can prove. The
  current Quafu path supports one full-register counts request and fails closed
  for unsupported output shapes or kinds.

## Naming and alternatives

`outputs` describes what the caller wants returned; `measurements` remains the
internal IR mechanism. `Observable` describes a mathematical object and avoids
promoting algorithm-layer Hamiltonian implementation classes into the stable
root. Separate `PauliWord`, `Hamiltonian`, builder, registry, and manager public
types are rejected as unnecessary concepts.

## Acceptance

- `fq.expectation(fq.X(0) @ fq.Y(1))` executes on the local CPU path;
- weighted Pauli sums return the correct differentiable expectation;
- `fq.probabilities`, `fq.samples`, and `fq.counts` have direct typed result
  accessors and preserve requested wire order;
- malformed observables, duplicate wires, missing shots, ambiguous result
  access, and unsupported remote outputs fail clearly before execution;
- the public workflow never requires users to construct IR measurement nodes;
- local and remote execution return the same `ExecutionResult` type;
- the existing numerical measurement kernels remain the single implementation
  authority.

## First-public-alpha release note

FlagQuantum now exposes mathematical observables and named output requests at
the root API. Internal measurement nodes are no longer part of the ordinary
user surface.
