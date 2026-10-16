# API Change Proposal 070 approval packet

Status: API-owner and Services-domain-owner approval recorded on 2026-10-13.
The exact token was supplied by the user as the authority for both roles.

Repository approval record target:
[`FQ-RESEARCH-CAPABILITY-SERVICES-20261013.md`](../api-changes/FQ-RESEARCH-CAPABILITY-SERVICES-20261013.md).
Exact authorization token:
`approve API_CHANGE_PROPOSAL_070_RESEARCH_CAPABILITY_SERVICES`.

## Requested decision

Approve the experimental `flagquantum.services` research-capability namespace
described by API Change Proposal 070 as the authoritative domain boundary for:

- the three versioned P0 capability identities;
- six deterministic scientific operations and two schema-catalog operations;
- closed request, result, validation, error, and Evidence Bundle schemas;
- explicit local provider order, unavailable-only fallback, ordered attempts,
  and content-hashed route explanation;
- the native, Qiskit Aer, and PennyLane Lightning local VQE providers.

Approval does **not** by itself authorize a stable root export, remote or paid
execution, credentials, QPU submission, an MCP tool registration, or a maturity
claim above `experimental`.

## Exact candidate surface

The authoritative machine-readable scope is
`contracts/research-capability-services-v1-candidate.json`. It contains:

- `search_research_capabilities`
- `describe_research_capability`
- `estimate_research_capability`
- `validate_research_capability_input`
- `execute_research_capability`
- `validate_research_evidence`
- `list_research_schemas`
- `describe_research_schema`

The P0 capability IDs are `ground_state.solve.vqe`,
`ground_state.solve.exact`, and `validate.energy_variance`, all at version
`1.0.0`. No natural-language parsing or protocol-specific behavior is included.

## Scientific and routing evidence

The approved candidate scope remains the bounded two-qubit, fixed-Hamming-weight
VQE workload in Proposal 070. Native, Qiskit Aer 0.17.2 with Qiskit 2.5.2, and
PennyLane Lightning 0.45.0 with PennyLane 0.45.1 satisfy the same scientific
contract. The two external providers completed the full 121-point workflow,
reported 361 backend executions, passed independent exact-energy and
energy-variance validation, and agreed in final energy within `1e-6`.

Provider selection remains caller-ordered. Candidate scores express rank only;
the service never silently reorders providers. Fallback requires explicit
authorization and advances only past unknown or locally unavailable adapters.
Execution failures and scientific-validation failures do not trigger fallback.
The complete assessment and attempt history are covered by the Evidence Bundle
digest.

## Verification record

The approval candidate passed:

- 118 focused FlagQuantum service, schema, Qiskit, PennyLane, and real-provider
  parity tests;
- Ruff and mypy for the changed service and provider boundaries;
- architecture, dependency-policy, team-ownership, and public-API baseline
  checks;
- 9 cross-repository MCP candidate-contract tests;
- 49 downstream TaskSpec, evidence, route-explanation, and gateway tests.

The development implementation commits, which are not part of the published
FlagQuantum `0.3.0` wheel, are:

- FlagQuantum `009cf77c`
- FlagQuantum MCP Server `a5ba292`

## Owner checklist

API owner:

- [x] Approve the operation names, signatures, schemas, and namespace.
- [x] Confirm the namespace remains at `experimental` maturity pending the
  separate release/maturity decision.
- [x] Confirm the approved contract remains experimental until the separate
  release/maturity decision records a compatible released version range.

Services domain owner:

- [x] Approve the bounded scientific scope and validator thresholds.
- [x] Approve unavailable-only provider fallback and route evidence semantics.
- [x] Approve native/Qiskit/PennyLane provider conformance as sufficient for P0.

Release/maturity owner, after both approvals:

- [x] Bind every P0 capability to `production_supported` or explicitly reject
  MCP activation for this release.
- [ ] Verify a published FlagQuantum wheel containing the approved research
  service API, then bind MCP Server to a version range that excludes earlier
  incompatible wheels. The published `0.3.0rc3` and `0.3.0` wheels omit this API.

## Release-artifact correction (2026-10-16)

The earlier approval record called `0.3.0rc3` the first compatible release.
Inspection of the PyPI macOS arm64 wheel for `0.3.0rc3` and the macOS arm64,
Linux x86-64, and Windows amd64 wheels for `0.3.0` found no
`flagquantum/services/research_capabilities.py` or
`flagquantum/services/research_execution.py`, and no
`search_research_capabilities` or `execute_research_capability` export in
`flagquantum/services/__init__.py`. The published `0.3.0` wheel has SHA-256
`7b7d5a1c8b45f0188b37bf807675eaee10931198ad34d0a8f78174a1ef65b979`.
Approval of the candidate design remains recorded, but neither published wheel
satisfies the activation gate. A future unique release must pass an installed
API check before an MCP compatibility range is asserted.

## Post-approval activation sequence

Only after the remaining release-artifact gate passes may the MCP repository
register the six tools and three resource patterns. Activation must still prove
released-version compatibility, direct-Python/MCP result equivalence,
lossless structured errors and evidence, wheel/process behavior, and a real
client discovery replay. A downstream client then performs the planned model A/B
replay
against the registered surface rather than importing FlagQuantum directly.
