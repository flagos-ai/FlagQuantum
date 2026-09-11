# Benchmark Results Layout

Benchmark result JSON is grouped by evidence class so human readers do not
mistake local, comparison, or smoke output for release-grade scalability
evidence.

| Directory | Evidence class | Release claim policy |
| --- | --- | --- |
| `local/` | Single-device fast-path results. | Must use `distribution_semantics="single_device_fast_path"` and `scalability_claim_allowed=false`. |
| `comparison/` | Cross-framework, CPU/GPU, or backend comparison results. | Non-release only; may compare performance but must not claim capacity expansion. |
| `smoke/` | Development, rank-local, replicated, or environment smoke checks. | Non-release only; sharded intent may be recorded, but `scalability_claim_allowed=false`. |
| `scalability/` | Release-gate scalability evidence only. | Only place for promoted payloads accepted by `--require-scalability`. |

These four directories are the only evidence classes. Historical experiment
families are retained in the team evidence archive, not under a fifth source
repository class. Published links and capability manifests must point only to
the minimal curated evidence retained here; see
[repository governance](../../docs/development/REPOSITORY_GOVERNANCE.md).

Large raw profiler traces, exploratory result matrices, per-iteration hardware
telemetry, and superseded optimization artifacts are stored outside the source
repository. The repository retains only evidence required by a current public
claim, regression, or release gate. External archives are development records
and are not release evidence until they are restored, normalized, audited, and
promoted through the release gate.

Non-release JSON should make that obvious with fields such as
`benchmark_evidence_class`, `non_release_evidence=true`,
`release_gate_allowed=false`, and a `scalability_blockers` explanation. Historical
smoke payloads are intentionally labeled as non-release evidence even when they
exercise sharded or collective code paths.

Promotion checks:

```bash
python benchmarks/audit_results.py --input benchmarks/results
python benchmarks/audit_results.py --input benchmarks/results/scalability --require-scalability
```

`benchmarks/audit_results.py --require-scalability` is scoped to
`benchmarks/results/scalability/`. Generated audit summary JSON files are not
inputs to release-gate scans.
