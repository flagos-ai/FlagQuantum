# Repository governance

The FlagQuantum source repository is the product integration point. It is not
the long-term store for raw profiler output, exploratory result matrices, or
superseded research artifacts.

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
team evidence store or a dedicated research/evidence repository. A retained
summary should record provenance, hashes, reproduction commands, and the
external archive identifier without claiming a stronger maturity level.

## Evidence lifecycle

1. A runner writes local or development output outside the promoted evidence
   set.
2. The result is normalized and classified as local, comparison, smoke, or
   scalability evidence.
3. Required provenance and release-gate checks are applied.
4. Only the minimal result and supporting manifest needed for a current claim
   are committed.
5. Superseded or bulky raw material is archived outside the source repository.

Historical benchmark families are quarantined under
`benchmarks/results/legacy/`; existing batch directories under `artifacts/`
remain legacy holdings. They may be migrated out in reviewable batches, but new
siblings must not be added. Migration must preserve links used by published
documents and capability manifests, either by updating them atomically or by
retaining a small redirect manifest.

## Review budget

Every change adding generated evidence should state:

- why the result must live in the source repository;
- its evidence class and retention period;
- the command and source revision used to create it;
- whether an existing result can be replaced instead of adding another copy.

Generated files larger than 1 MiB or result batches larger than 20 files should
default to external storage unless a release or regression gate requires them.
