# API Change Proposal 029: Twin evidence persistence lifecycle

## Status

**Implementation authorized; pending API freeze review.** The user authorized
the next Twin API slice after merging Proposal 028: complete the public offline
evidence lifecycle and include executable usage in the pull request.

## Problem

`fq.twin.load_evidence(path)` safely restores a canonical evidence envelope,
but callers still serialize files themselves. Ad-hoc JSON writing can produce
non-canonical output and can silently replace previously frozen evidence.

## Decision

- Add `fq.twin.dump_evidence(evidence, path)` beside `load_evidence`.
- Emit deterministic canonical JSON with a trailing newline.
- Create a destination only when it does not exist.
- Treat an existing envelope with the same identity as an idempotent success.
- Refuse to overwrite different or invalid existing content.
- Keep the operation offline and provider-neutral.

## Public surface

```python
def dump_evidence(
    evidence: TwinEvidenceEnvelope,
    path: str | os.PathLike[str],
) -> None: ...
```

The complete public workflow is:

```python
fq.twin.dump_evidence(evidence, "twin-evidence.json")
restored = fq.twin.load_evidence("twin-evidence.json")
report = twin.evidence_report(circuit, evidence=restored)
```

## Boundary

The function performs local file I/O only. It does not create evidence, fetch
calibration, contact a provider, submit or poll a QPU task, train a model,
promote a candidate, route a workload, or make an application policy decision.

## Compatibility

The change is additive. The version-1 envelope schema, identity, loader,
prediction, and evidence-report semantics do not change.

## Acceptance

- A dumped envelope loads with the same value and identity.
- Output is deterministic canonical JSON.
- Repeating the same dump is idempotent.
- Different or invalid existing files are preserved and rejected.
- Invalid input types and unwritable destinations fail explicitly.
- Public contracts, documentation, and scenario tests cover the new entry point.
