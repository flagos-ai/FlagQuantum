# API Change Proposal 098: Identifiable Twin probe banks

## Status

**Implementation authorized; additive to the frozen Twin v1 contract.** This
slice freezes an offline probe design and states exactly which parameters of one
restricted local error family that design can identify.

## Problem

A deterministic probe-refresh plan can choose circuits that cover an evidence
gap, but circuit count and structural coverage do not prove that a model is
identifiable. Six repeated versions of the same experiment may add shots while
leaving the design rank unchanged. Conversely, a smaller, diverse design can
separate the declared parameters.

FlagQuantum therefore needs a framework-level object that binds probe circuits,
physical mapping, exercised directed couplers, parameter names, and an explicit
design matrix. The object must report missing parameters instead of treating a
fixed number of probes or calibration states as proof of sufficiency.

## Decision

- Add immutable `TwinProbeBank` and
  `fq.twin.prepare_probe_bank(target=..., physical_qubits=..., probes=...,
  parameter_family=...)`.
- Support the first explicit family, `local_pauli_readout`, with one readout
  bit-flip and one single-qubit Pauli burden per mapped qubit, plus one
  two-qubit Pauli burden per exercised directed coupler.
- Derive rows from every probe's computational-basis Z marginals and exercised
  directed-coupler ZZ correlations. Coefficients record the declared local
  operation incidence; they are an exact structural design, not fitted noise
  parameters or a physical response Jacobian.
- Compute exact matrix rank with rational arithmetic. A parameter is
  individually identifiable only when removing its column lowers the matrix
  rank. Report every other parameter in `unidentified_parameters`.
- Return `identifiable` only when the matrix has full column rank; otherwise
  return `underdetermined`. An underdetermined bank is an explained rejection
  for later estimation, not partial fitted evidence.
- Canonicalize probe order by circuit identity and bind target, ordered physical
  mapping, directed couplers, design rows, and parameter support into the bank
  identity.

## Public API

This example is complete and runs without credentials, provider access, or a
QPU job:

```python
import flagquantum as fq

probe_circuits = (
    fq.Circuit(2).cx(0, 1),
    fq.Circuit(2).h(0).cx(0, 1),
    fq.Circuit(2).h(1).cx(0, 1),
)

bank = fq.twin.prepare_probe_bank(
    target="quafu:Shenglian",
    physical_qubits=(20, 27),
    probes=probe_circuits,
    parameter_family="local_pauli_readout",
)

print(bank.identifiability_status)
print(bank.parameter_names)
print(bank.unidentified_parameters)
print(bank.matrix_rank)
print(bank.identity)

assert bank.identifiability_status == "identifiable"
assert bank.matrix_rank == len(bank.parameter_names)
assert bank.unidentified_parameters == ()
```

The target uses the same canonical `provider:backend` form as the rest of the
public Twin API. Quafu is a native provider integration, not a semantic special
case in the probe-bank mathematics.

## Result contract

The bank records:

- canonical provider/backend target and ordered physical qubits;
- sorted directed couplers actually exercised by the frozen probes;
- sorted immutable probe circuit identities;
- the exact parameter family and ordered parameter names;
- each probe's represented parameters;
- uniquely labelled design rows and the complete non-negative integer matrix;
- exact matrix rank, individually identifiable parameters, missing parameters,
  status, schema, and deterministic identity.

`to_dict()` returns a canonical JSON-compatible representation. File persistence
is intentionally deferred: this slice does not add a second evidence artifact
or imply that a design has been run on hardware.

## Fail-closed behavior

The factory rejects malformed targets, empty or duplicate mappings, duplicate
probe identities, width mismatches, explicit measurements, channels, dynamic
operations, operations wider than two qubits, and unsupported parameter
families. Direct construction rechecks canonical ordering, matrix dimensions,
row-to-probe binding, support declarations, rank, identifiable parameters, and
status.

The public object has no submission, polling, retry, cancellation, fitting,
estimation, mutation, promotion, or routing method. A later fitting API must
accept only an `identifiable` bank and separately validated observations.

## Compatibility

Additive. Existing Twin models, evidence, releases, histories, provider
adapters, prediction behavior, and serialized v1 artifacts are unchanged. The
Twin v1 candidate contract gains one factory, one frozen runtime type, one
status literal, one method, and one property.

## Acceptance

- Identical probe contents in any input order produce byte-identical payloads
  and the same bank identity.
- Rank and individual identifiability are computed from the frozen design
  matrix rather than probe count or a fixed calibration-state threshold.
- Repeating many structurally equivalent probes cannot create full rank.
- Mapping and directed-coupler identities are explicit and immutable.
- Unsupported parameters remain named and unestimated.
- Tampered matrix rank, support, labels, parameter order, or status fails
  construction.
- Focused Twin tests, public API contracts, strict typing, formatting,
  documentation checks, default CI, and runtime CI pass.

## Non-goals

- No fitted parameter values, intervals, accuracy, uncertainty, fidelity, or
  state reconstruction claim.
- No arbitrary correlated, non-Markovian, leakage, crosstalk, pulse, or
  many-body parameter family.
- No automatic probe synthesis, active learning, calibration-state scheduling,
  model update, or self-evolution loop.
- No QPU submission, provider polling, retry, cancellation, routing, or hardware
  control.
- No Agent, MCP, natural-language, REST, UI, dashboard, or application policy.
