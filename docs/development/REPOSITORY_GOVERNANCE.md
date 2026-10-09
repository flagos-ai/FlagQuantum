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
| Entire tracked tree | 62,000,000 bytes |
| Entire tracked tree | 3,020 files |
| `benchmarks/results/` | 23,800,000 bytes |
| `benchmarks/results/` | 290 files |
| `docs/development/` | 200 files |

These values are ceilings, not allocations. A change that needs more room must
replace or archive existing material instead of raising a limit. Whenever a
cleanup reduces one of the measurements materially, the same change lowers its
corresponding budget so the removed volume cannot silently return.
