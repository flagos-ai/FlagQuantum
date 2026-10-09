# Repository governance

The FlagQuantum source repository is the product integration point. It is not
the long-term store for raw profiler output, exploratory result matrices, or
superseded research artifacts.

This document covers what content belongs in the repository. For who decides what,
and for where to report a vulnerability, see the repository-level
[GOVERNANCE.md](../../GOVERNANCE.md) and [SECURITY.md](../../SECURITY.md).

## What belongs in the main repository

| Content | Canonical location | Retention rule |
| --- | --- | --- |
| Product source | `flagquantum/` | Versioned with the package. |
| Tests and reusable fixtures | `tests/` | Keep fixtures minimal and deterministic. |
| User and maintainer documentation | `docs/` | Current truth; archive superseded narratives outside reference docs. |
| Runnable examples | `examples/` | Must use supported public APIs. |
| Benchmark runners and workloads | `benchmarks/` | Keep maintained, reproducible entry points. |
| Machine-readable capability contracts | `contracts/` | Stable filenames and schema versions. |
| Repository automation | `tools/` | No runtime dependency on repository tools. |
| Curated evidence | `benchmarks/results/{local,comparison,smoke,scalability}/` | Keep only evidence needed by a current claim, regression, or release gate. |

## What does not belong

Store raw profiler traces, repeated experiment matrices, temporary cloud task
payloads, intermediate plots, checkpoints, and superseded result sets in the
team evidence store or the public
[FlagQuantum evidence archive](https://github.com/FlagQuantum/FlagQuantum-evidence).
A retained summary should record provenance, hashes, reproduction commands, the
immutable release tag, and the external archive identifier without claiming a
stronger maturity level.

## Evidence lifecycle

1. A runner writes local or development output outside the promoted evidence
   set.
2. The result is normalized and classified as local, comparison, smoke, or
   scalability evidence.
3. Required provenance and release-gate checks are applied.
4. Only the minimal result and supporting manifest needed for a current claim
   are committed.
5. Superseded or bulky raw material is archived outside the source repository.

New files under `benchmarks/results/` must also be listed in
[`benchmarks/evidence-retention.toml`](../../benchmarks/evidence-retention.toml).
The registry applies only to newly added evidence; it does not make existing
artifacts deletion targets. Each entry explains why the artifact stays and uses
one of these lifecycle classes:

- `temporary`: records a `review_after` date;
- `current_claim`: names the checked-in documentation or contract that consumes it;
- `regression_baseline`: names the checked-in test, contract, or tool that consumes it.

An entry may name the older result it `supersedes`. Remove the entry when its
artifact is archived or removed. The CI and pre-push gates check this relationship
with `python tools/check_evidence_retention.py --base <revision>`.

Historical benchmark families, provider batches, paper workspaces, task
fragments, and agent-specific authoring workflows are stored outside the source
repository. New compact, non-release artifact outputs use
`artifacts/development/`. An external migration must preserve any links used by
published documents and capability manifests by updating them atomically and
recording the source revision plus archive checksum.

The source repository must not add `benchmarks/results/legacy/`,
`benchmarks/development/`, `benchmarks/research/`, `artifacts/legacy/`, paper
submission workspaces, historical PR-readiness records, or project-specific
`.codex/skills/` directories. Reusable workloads belong under
`benchmarks/runners/`; one-off analysis belongs in the external evidence
archive. Git history is a
recovery mechanism, not the long-term evidence store.

## Review budget

Every change adding generated evidence should state:

- why the result must live in the source repository;
- its evidence class and retention period;
- the command and source revision used to create it;
- whether an existing result can be replaced instead of adding another copy.

Generated files larger than 1 MiB or result batches larger than 20 files should
default to external storage unless a release or regression gate requires them.

## Size budgets

`python tools/check_repository_hygiene.py --report` prints the measurements used
by the repository gate. The checked-in budgets deliberately sit just above the
current tree:

| Scope | Budget |
| --- | ---: |
| Any tracked file | 1,800,000 bytes |
| Entire tracked tree | 61,900,000 bytes |
| Entire tracked tree | 3,010 files |
| `benchmarks/results/` | 23,700,000 bytes |
| `benchmarks/results/` | 285 files |
| `docs/development/` | 200 files |

These values are ceilings, not allocations. A change that needs more room must
replace or archive existing material instead of raising a limit. Whenever a
cleanup reduces one of the measurements materially, the same change lowers its
corresponding budget so the removed volume cannot silently return. The budgets
are guardrails against accidental growth, not quotas or instructions to remove
useful, current evidence.

### Phase 2 closeout baseline

The evidence cleanup closed on 2026-10-09 at source revision `252acf67a`. The
post-merge baseline, measured before this note was added, is:

| Scope | Measured value |
| --- | ---: |
| Entire tracked tree | 2,996 files |
| Entire tracked tree | 61,781,105 bytes |
| `benchmarks/results/` | 280 files |
| `benchmarks/results/` | 23,664,402 bytes |
| `docs/development/` | 192 files |

The closeout audit found no byte-identical files under `benchmarks/results/`.
It also found no further artifact that met all removal conditions at once: no
checked-in consumer, clear supersession or obsolescence, reproducibility or a
verified archive copy, and a passing contract suite after removal. Large files
alone were not treated as candidates. In particular, the largest raw GPU sample
CSV is named by its retained capacity result, and the matched-speed calibration
records are consumed by the release manifest and contract tests.

Phase 2 therefore closes with no additional deletion batch. A future cleanup may
reopen an individual artifact only when it has evidence for every removal
condition above; a zero exact-reference search by itself is not sufficient.
