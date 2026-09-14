# API Change Proposal 044: Resumable Twin candidate submission

## Status

Approved for implementation. The API owner authorized the next Twin increment
after merging the prospective candidate comparison in pull request 42.

## Problem

A prospective candidate comparison may remain in a provider queue longer than
the submitting Python process. `TwinSubmission` safely persists one experiment
and receipt, but it does not retain the incumbent snapshot and prediction needed
to compare the candidate with the incumbent. Saving those records separately
would allow mismatched files to be combined accidentally.

## Additive API

```python
receipt = trial.experiment.submit(provider)  # the only QPU submission
submission = fq.twin.TwinCandidateSubmission.from_receipt(trial, receipt)
fq.twin.dump_candidate_submission(
    submission,
    "twin-candidate-submission.json",
)

# A later process resumes the same task. This path cannot submit a task.
restored = fq.twin.load_candidate_submission(
    "twin-candidate-submission.json"
)
result = provider.fetch_result(restored.receipt)
evaluation = restored.validate_result(result, circuit=circuit)
```

The public additions are:

- `TwinCandidateSubmission`, an immutable binding between one complete frozen
  candidate trial and one existing `TwinSubmission`;
- `TwinCandidateSubmission.from_receipt(...)`, which validates a receipt
  offline and never contacts a provider;
- `dump_candidate_submission(...)`, a mode-`0600`, create-once, idempotent
  writer that refuses to replace different or invalid content;
- `load_candidate_submission(...)`, a strict offline reader;
- `TwinCandidateSubmission.validate_result(...)`, which compares the two
  frozen predictions with the result belonging to the restored receipt;
- strict `from_dict(...)` readers for the candidate trial and submission.

## Safety and ownership

Loading, binding, and validating never submit, poll, cancel, retry, retrain,
promote, route, or modify hardware. Remote continues to own provider contact.
The restored object exposes the retained receipt so the application can query
or fetch through the provider it explicitly constructed.

The file contains the complete incumbent and candidate snapshots, both frozen
predictions, canonical OpenQASM, physical mapping, shot count, task ID, and the
allowlisted receipt provenance required by the existing `TwinSubmission`
contract. Unknown fields, changed identities, changed experiments, invalid
schemas, and mismatched receipts fail closed.

A provider submission and a local file write cannot be one atomic transaction.
If a process exits after the provider accepted a task but before the checkpoint
was written, the operator must reconcile the printed task ID or provider task
history. The example must never recommend blindly running `submit` again.

## Compatibility

The change is additive. It does not modify Twin v1 signatures, existing schemas,
`predict()`, candidate decision semantics, or `TwinSubmission`. It adds no
dependency and no Agent, MCP, dashboard, scheduler, automatic promotion, or
natural-language responsibility to FlagQuantum.
