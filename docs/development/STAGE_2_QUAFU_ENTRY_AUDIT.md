# Stage 2 Quafu Minimum Real Vertical Workflow Entry Audit

Status: **Technical baseline complete — P0 implementation not authorized**
Date: 2026-09-02
Governing contract: [Proposal 012](API_CHANGE_PROPOSAL_012_QUAFU_PROVIDER_CONTRACT.md)

## 1. Existing Capabilities

- `QuafuProvider` implements token validation, backend discovery, chip-info,
  submit, status, cancel, result retrieval, and bounded polling.
- Standard `DeploymentPackage` and direct QASM submission requesting disabled
  provider compilation are supported. Hardware evidence shows that this request
  does not guarantee unchanged circuit execution: the returned `transpiled`
  circuit is the execution identity source.
- Submission receipts, routing evidence, and deployment artifact identity reach results.
- Results perform shot accounting. Portal bit order is preserved by default,
  with an explicit legacy reversal option.
- Mock tests cover tokens, submission, polling, calibration retrieval, physical
  QASM, and basic results.
- Density-matrix probabilities and readout noise were fixed in core Runtime and
  are protected by contract tests.

This evidence establishes that the core components exist, not real QPU capability,
portal compatibility, or production reliability.

## 2. Incomplete Workflow

| Proposal 012 P0 item | Current status | Gap |
| --- | --- | --- |
| Asymmetric bit-order contract | Partial | Machine candidate freezes examples, but Quafu result metadata does not require a bit-order declaration |
| Remove adapter `QPUTwin` dependency | External verification pending | Adapter source is outside this repository; real installation evidence is required |
| Minimum result metadata | Incomplete | Backend, duration, calibration/noise identity, physical qubits, and program digest are not mandatory |
| Separate submission from terminal status | Partial | Asynchronous polling exists, but generic states still allow arbitrary strings |
| Five-state normalization | Incomplete | States such as `Queued` are not stably mapped to `Pending`; original states are not uniformly preserved |
| Error categories | Incomplete | Failed/Cancelled and platform errors still mainly surface as generic `RuntimeError` |
| Idempotent submission | Incomplete | No idempotency key or duplicate-submission protection |
| Capability preflight | Incomplete | Portal facts about shots, QASM profiles, basis gates, and topology are not uniformly validated |
| Real end-to-end evidence | Incomplete | No submit/poll/result evidence using a real registered backend ID |

## 3. Recommended P0 Implementation Boundary

The first implementation should be limited to:

- Internal state normalization with original-state preservation.
- Explicit counts bit-order metadata at the Provider boundary.
- Minimum metadata validation without changing stable dataclass fields.
- Provider-private transport support for idempotency keys.
- Tests for timeout, Failed, Cancelled, empty results, asymmetric counts, and
  shot mismatches.
- Real adapter installation contract scripts or schemas without copying
  FlagQuantum source.
- Negative tests for credential leakage.

Separate API change approval remains required for new stable exception
hierarchies, changed dataclass fields, root exports, a generic Provider API freeze,
or promotion of provider metadata to public schemas.

## 4. Real Evidence Gate

Stage 2 cannot exit on mock evidence alone. At least one controlled real record
must establish:

```text
registered external_backend_id
  -> sealed deployment artifact
  -> provider_job_id
  -> Pending/Running/terminal status
  -> normalized counts + shot accounting
  -> actual backend/calibration/program provenance
```

Real records must be sanitized: do not retain tokens, user information, or private
portal responses. Backend IDs may be deployment bindings, but must not enter
program IR or program identity.

## 5. Approval Checkpoint

Proposal 012 remains Draft. A generic `do` does not authorize changes to protected
Provider behavior. Generate and approve a separate machine-readable candidate
before P0 implementation. Suggested approval phrase:

```text
approve QUAFU-PROVIDER-P0
```

This phrase does not authorize a public Provider API freeze, real paid job
submission, or writes to the adapter repository.
