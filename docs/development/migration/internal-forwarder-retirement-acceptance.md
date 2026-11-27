# Internal forwarder retirement acceptance

Baseline: `5f8e97c771925ebaf0a19f8fca26ab16c328ec4a`.
Audit date: 2026-11-26, Asia/Shanghai.
Audited open, draft #1084 head: `9b1a8cd196f9d8b8427b382c404f66705cc57840`.
Review: [draft PR #1331](https://github.com/flagos-ai/FlagQuantum/pull/1331).

## Completed scope

- 283 old tracked files removed, including all 138 root tool command wrappers.
- Old Kernel, JAX, algebra and private Torch frontend trees removed.
- 523 private forwarding statements removed from migration facades, covering
  502 distinct old members; another 29 pre-existing private compiler stage
  aliases and one unused private wrapper removed.
- Ten unused partial numerical reexports removed. Imports used by an actual
  implementation remain ordinary dependencies, not forwarding interfaces.
- Generic module alias registration and reverse owner-to-facade imports removed.
- Existing source, tests, CI, hooks, wheel commands and executable comparison
  harnesses use actual implementation owners. Historical evidence identifiers
  resolve through explicit source mapping without changing their records.
- All 99 retained public entries (83 relocation entries and 16 already documented
  expert compiler facades) have an itemized basis, exports, owner and retirement
  condition in the [inventory](internal-forwarder-retirement.md) and its JSON.
- Only three internal files await #1084 consumers: structured frontend model,
  verifier and their import parent. They are not public compatibility promises.

The current map has 456 rows, including alias-only retirements that claim no
moved definitions. The final seven-path #1084 increment has no relocated-source
intersection, added-target collision or new import of a retired path. The sole
full-PR intersection remains the individually audited density embedding helpers;
the rest of the changed engine remains in place.

## Public behavior and source validation

Final complete Unit/API/Team result: 19,395 passed, 957 skipped (422.94 seconds).
Root modules and local training lifecycle integration: 1,591 passed, 193 skipped
(106.23 seconds). Live Kaiwu smoke and unavailable hardware/dependency fixtures
retain their explicit exclusions/skips.

The initial full validation exposed old tool-file loaders, private mock targets,
raw source-pointer readers and a synthesis admission cache lookup that depended
on a historical module alias. These were corrected without restoring retired
wrappers. An isolated process now creates a canonical synthesis Program and runs
it without loading the historical services module. Compiler wiring assertions
continue to compare the actual registered pass functions and analyses.

The pre-retirement wheel supplied real pickle bytes for MemorySnapshot,
CollapseOperator and LindbladPlan. The restored plan still executes. A separate
fixture preserves all 127 previously supported public class field-annotation
cases, qualified class modules and annotation extras. Constructor signatures and
dataclass field contracts stay unchanged. Three previously unsupported cases
are recorded as existing limits rather than claimed as fixed.

Ruff, Black, strict mypy, public API, architecture, frozen vocabulary, ownership,
documentation links/source, repository hygiene and language checks: all passed.
Changed-path ownership also passes: 882 added/modified paths use the current
policy, and 283 deletions use their baseline ownership; Integration scope covers
all 1,165 changed paths. Deleted-only ownership reservations are not reintroduced.
The structural guard rejects resurrected files/directories and operational
absolute or relative imports of retired files and members.

The normalized AST audit still covers 1,522 moved top-level definitions, with
only the same five reviewed import/path/ownership differences as the preceding
migration. No numerical algorithm has been rewritten.

## Installed artifact

Platform: macOS arm64, Python 3.12.14, Torch 2.14.1.
Wheel: `flagquantum-0.4.0-cp310-abi3-macosx_11_0_arm64.whl`.
SHA-256: `3288315bfaa16bcf263513f321d59df9f2209c8c0d1513130493fd451ecfb83c`.

Built through the declared setuptools backend from a clean build directory, then
rebuilt after final source changes. Archive inspection finds no retired members;
every packaged Python source matches the reviewed checkout byte for byte.
Isolated installation verifies CPU execution and gradients, historical pickles,
public object identity, retired-import rejection, native CPU extension loading,
metadata-only protocol/catalog imports and the benchmark console entry.

The development environment contains an editable installation of another
checkout. Validation excludes that finder and source path per process so it
cannot supply deleted modules. The shared environment itself is not modified.

Frozen contracts, public API snapshots, packaged contract copies and raw
scientific evidence are unchanged. This is local structural acceptance, not a
GPU/QPU, distributed-capacity, Rust or performance qualification. GitHub CI must
qualify the new pushed commit; results for the preceding migration do not count.

## CI manifest follow-up

The first remote CPU and distributed jobs stopped before tests: the locked-image
manifest comparison included Ruff configuration, whose per-file path changed
with the operator relocation. The checker now excludes Ruff settings, as it
already excludes release-wheel configuration. Dependency declarations, optional
dependencies, build requirements and the build backend remain compared.
All 25 offline-environment tests pass, including relocated lint paths and rejection
of dependency/build changes. The normalized checkout manifest also matches the
audited main baseline. Remote CI must validate the follow-up commit.

## Installed QEC probe follow-up

The next package job reached the real QEC wheel installation and correctly
rejected an import from the checkout. Three standalone installed-artifact probes
had acquired a source-root bootstrap during relocation; those probes must never
prepend the checkout, so the bootstrap is removed. Their original installation
assertions remain intact. A subprocess regression checks all three probes from
outside the repository in isolated mode and rejects import-path mutation.
Related package/QEC tests: 102 passed, 7 skipped. The existing installed macOS
wheel also passes the full core QEC probe outside the checkout, with reused local
dependency packages; this is not a fresh Linux installation claim. Ruff, Black
and architecture checks pass. Linux package CI must qualify the pushed fix.

## Cold source restoration budget

The latest CPU source producer succeeded on retry. GPU source restoration
continued making verified progress but was killed at the ten-minute step limit,
after requesting 1,145 objects; checkout fallback then failed with a TLS reset.
Both producers now bound the API step at 20 minutes and the job at 30 minutes,
reserving ten minutes for fallback, packing and uploads. This accommodates the
existing 2,048-object ceiling without increasing batch size, API concurrency or
retries. Integrity validation, read-only permissions, fallback guards and all
downstream test gates remain unchanged. Scheduling tests pin both budgets.

The same remote run exposed two remaining source-hash readers in executable
MPS evidence producers and one native-extension-only private mock target. The
producers now hash the relocated capacity supervisor and profiling owner; raw
results/manifests remain untouched. The native gather test patches the actual
Simulation helper, preserving its call-count, value and gradient assertions.
The native test passes against the previously built installed wheel (whose
production implementation is unchanged). Source scheduling/integrity tests: 164
passed; the critical-path source snapshot regression passes. Local recovery
tests have ten passing non-collective cases; the seven two-rank cases cannot be
claimed passed because macOS c10d rendezvous timed out, including after allowing
loopback networking. No skip or tolerance relaxation was added. Linux CI must
validate the two-rank recovery path and its new provenance assertion.

## Coverage scope follow-up

The complete 192-shard measurement passed its test selections but the policy
checker still queried retired physical package names. It reported zero coverage
for Kernel/JAX/frontend scopes and measured only the remaining public facades
for several other scopes. The existing source-relocation ledger now attributes
actual file/definition line hits to the original policy directories. Splits and
merges retain their original floors; unrelated destination functions/globals
cannot donate coverage. Retained facades and new code are still measured.
No policy minimum, exclusion, raw hit or global rate is changed.

The real artifacts from CI run 45801572959 were downloaded and each SQLite
input passed integrity_check. Their measured merge commit is
`1de8b02aec693f93b8b4cbc1b151f19d47744e0d`; its tree exactly matches retirement
head `47a78728bfe85309f404387426a03e0aa1d8d7c7`. Recombination preserves all
111,547 covered and 125,677 total measured statements and the reported global
88.76%. All frozen scope floors pass, including Operators 96.12%, Runtime
Distributed 80.54% and the historical Kernel/JAX scopes.

All 103 coverage/capacity/routing tests pass. Cases reject missing measurements,
duplicate line counting, ignored facade lines, loss of a partial source floor,
and donation from an unrelated destination. coverage-policy.toml remains byte
identical (SHA-256 `405133db0ef6619c59f041dda43928247da2a0d11aa84df784b37ff35887b5fa`).
The workflow now retains actual coverage.xml after a policy failure as well as
a success, so diagnosis does not require downloading all shard databases.
This is a local replay of real remote data; the next head still needs new CI.
