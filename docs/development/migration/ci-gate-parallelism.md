# Parallel CI gates with complete acceptance coverage

The qualified #1354 run has all 394 CPU test partitions below 600 seconds,
including preparation/upload and excluding queue: Runtime 511, coverage 588,
unit 421 and distributed CPU 432 seconds. The preferred limit remains a
per-execution observation, not a guarantee about future runner/network variance.
All 384 original Runtime/coverage reports, selections, resources and original
scientific call phases are retained. Whole-CI work remains: `quality` takes 1,080
seconds, one dependency-bounds cell 1,031 and `package` 701.

## Changes and preserved work

| Existing gate | Execution arrangement |
| --- | --- |
| `quality` | Keep the contract/type/architecture gates. Run formatting/language/legacy-API scanning, performance-provenance tests and evidence-origin validation in separate required jobs. |
| Historical evidence checks | Preserve complete-history checkout, reachable refs, commands, flags and full evidence inputs in the jobs that consume them. Each original verification operation remains required. |
| `package` | Keep reproducible wheel/sdist builds, distribution validation, checksummed manifest and distribution publication in the producer. |
| Package installations | Consume that exact distribution artifact for all existing older-Torch, cold-wheel, sdist and QEC acceptance. Archive digest checking and the manifest's current-commit binding preserve the producer/consumer association. |
| Manylinux | Compile and test the original publishable wheel independently from the same source revision, retaining Ubuntu 22.04/GCC 11 and the existing Stable ABI/Torch versions. |
| Dependency bounds | Retain all seven Python/Torch cells, CPU-only requirements, full smoke/native tests and declared development-floor checks. Use the existing BW CPU pool with a fresh isolated environment and compatible cached wheel bytes; installed SDK dependencies are not inherited. |

The slow dependency cell ran on `fq-a100-cpu10`: isolated-environment preparation
334 seconds, source restoration 189, CPU Torch installation 135, build-tool
installation 106 and editable installation 103; tests took 45. Routing this
fresh compatibility check to the already provisioned BW pool uses idle capacity
without adding runner registrations, labels, quotas or services.
The fresh project install resolves all declared core dependencies under the same
selected Torch constraint, including the Python 3.10 TOML reader.

The workflow keeps its existing hosted concurrency buckets. The terminal
`ci-complete` gate must require every added job and fail for a failed, cancelled
or missing required acceptance path. Existing conditional platform jobs keep
their conditions; no new optional skip replaces a required check.

## Acceptance

Compare the complete old/new workflow verification-step footprint, accounting
for relocated commands and explicitly duplicated setup. Preserve original
scientific selections, resource contracts, precision, optimizer budgets, grids,
draws and clocks. Verify producer/consumer dependencies and artifact identity,
full-history provenance, both success and failure gating, all seven dependency
cells, fresh environments and the hosted concurrency bound.

Run the existing package, hardware-lane, source, concurrency, dependency,
required-check and release-policy tests. Exercise real source/native-installed
workflows and exact-head CI; static YAML checks alone do not qualify the new
execution graph. Retain raw per-job/step intervals and report every execution
above 600 seconds, including non-partition jobs, rather than selecting a favorable
rerun. Keep preceding observations and failures as historical evidence.
