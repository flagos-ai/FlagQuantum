# Team Scope and Protected Path Coverage Inventory

Status: integration governance inventory. Sections 1, 2, and 3 of "Proposed
change" are applied by the change that carries this revision; section 4 is half
applied and the open questions remain open.

Owner: Integration

Worktree: `FlagQuantum-vNext`

Branch: `refactor/flagquantum-vnext-architecture`

Inventory date: 2026-10-06

Verification date: 2026-10-07

Scope: `team-ownership.toml` coverage and `architecture.toml` package-surface
consistency.

This inventory changes no Stable Core surface, moves no file, adds no
abstraction, and makes no distributed or performance claim. It records ownership
facts and applies a governance diff. It is admissible under the binding control
sequence in `AGENTS.md` "Current Strategic Priority" because it opens no
horizontal architecture track: it adds no contract type and no boundary, and it
removes entries that no longer match any path.

## Decision

`team-ownership.toml` left eight `flagquantum/` domains and seven root modules
reachable only through `teams.integration.owns = ["**"]`. Because the
most-specific rule in `docs/development/MULTI_TEAM_DEVELOPMENT.md` selects a
specialist team when one matches and falls back to Integration otherwise, no path
is unowned. The defect is narrower and more consequential: work in those paths
cannot be delegated to a specialist team, so every change to them is an
integration change and is serialized behind the integration owner.

This document records, with evidence, (1) which paths have no specialist owner,
(2) the assignment applied for each, (3) one verified defect in `protected_paths`,
and (4) four adjacent inconsistencies that a reviewer should decide on rather
than inherit.

## Verification

The first pass of this inventory used a cone-mode sparse checkout in which
`tools/`, `tests/`, `benchmarks/`, `examples/`, `contracts/`, and `.github/` were
absent, so its claims about the matcher and about the package allowlist were
stated as reasoning and marked unverified. That checkout has since been disabled.
Every command below was executed against the full tree of 2518 files, so the
findings now cite observed behaviour instead of inference.

- The path matcher is `fnmatch.fnmatchcase` (`tools/check_team_scope.py`). Under
  `fnmatch`, `*` already matches `/`, so `dir/**` requires a literal `dir/`
  prefix and can never match the sibling module `dir.py`.
- `tools/check_architecture.py` compares `allowed_top_level_directories` against
  the real `flagquantum/*` directories that contain `*.py`, and reports both
  directions: an unreviewed directory and a stale allowlist entry each fail.
  Finding 4's second reading therefore holds.
- `git log --all` finds no commit for `flagquantum/devices`,
  `flagquantum/encoding`, `docs/site`, `docs/roadmap/VNEXT_EXPERIENCE.md`,
  `docs/development/DOCUMENTATION_STRATEGY.md`, or `requirements*.txt`: none of
  them ever existed. `git grep` across the tree finds each name only in
  `team-ownership.toml` itself, so no document authorizes any of them.
- `flagquantum/lindblad/` was the only real domain directory without a
  `README.md`; the other nineteen have one.

## Finding 1: eight domains and seven root modules have no specialist owner

Coverage of `flagquantum/**` by the nine specialist teams in
`team-ownership.toml`, applying the most-specific rule:

| Path under `flagquantum/` | Files | Specialist owner | Note |
| --- | --- | --- | --- |
| `algorithms/` | 20 | none | Applications area |
| `benchmarking/` | 24 | none | Evaluation area |
| `circuit.py` | 1 | core | |
| `compiler/` | 36 | compiler | |
| `compute/` | 10 | compute | |
| `core/` | 16 | core | |
| `deployment/` | 6 | remote | |
| `drawer/` | 6 | compiler | |
| `dynamic.py` | 1 | none | Root module |
| `ecosystem/` | 43 | ecosystem | |
| `errors.py` | 1 | none | Root module |
| `experimental/` | 11 | none | Documented as intentionally integration-owned |
| `gradients.py` | 1 | none | Root module |
| `kernels/` | 18 | none | |
| `lindblad/` | 2 | none | |
| `models.py` | 1 | none | Root module |
| `noise/` | 5 | core | |
| `observables/` | 2 | core | |
| `operators.py` | 1 | none | Root module |
| `qec/` | 11 | none | |
| `remote/` | 30 | remote | |
| `runtime/` | 251 | runtime, plus simulation (`runtime/executors/**`) and remote (`runtime/target_execution.py`) | Most-specific rule already resolves three owners inside this subtree |
| `services/` | 6 | services | |
| `simulation/` | 98 | simulation | |
| `testing/` | 12 | none | |
| `training.py` | 1 | none | Root module |
| `twin/` | 33 | none | |
| `version.py` | 1 | none | Root module |

`flagquantum/__init__.py` and `flagquantum/_api.py` also have no specialist owner,
which is intended: both are in `protected_paths`.

## Finding 2: `flagquantum/core/ir/**` does not protect the public IR module

`protected_paths` contains the entry

```toml
"flagquantum/core/ir/**",
```

but no `flagquantum/core/ir/` directory exists. The public IR is the single module
`flagquantum/core/ir.py`, which defines `IR_VERSION = "1.0"` and the `CircuitIR`
contract.

Under a path-glob reading, `flagquantum/core/ir/**` matches paths inside a
directory named `ir` and requires the literal separator after `ir/`. Under an
fnmatch or regular-expression reading it is `flagquantum/core/ir/...` and still
requires that separator. `flagquantum/core/ir.py` is a sibling of the `ir`
directory name, not a child of it, so it does not match under either reading.

That the module is meant to be protected is not in doubt:

- `docs/generated/STABLE_API.md` lists `fq.CircuitIR` as Stable.
- `docs/architecture/ARCHITECTURE_DEPENDENCIES.md` assigns "IR and parameters" to
  `flagquantum.core` and lists `flagquantum.core.ir` as the measurement contract
  surface.
- The neighbouring entries `flagquantum/__init__.py`, `flagquantum/_api.py`, and
  `flagquantum/core/_artifacts.py` protect the public facade and the artifact
  contract by exact file path, so the intent of this list is to protect the
  serialized public surface by exact path.

The practical effect was that the defining module of the public `CircuitIR`
schema was constrained by `docs/public_api_v1.json`, the public API contracts,
the snapshot checker, and Core team ownership, but was not constrained by the
protected-path mechanism itself. The matcher is now known, and the hole is
confirmed: `scope_errors("core", ["flagquantum/core/ir.py"], policy)` returned no
error, so a Core change to the public IR module bypassed the protection check
entirely.

The class of defect matters more than this instance. A `dir/**` entry whose
directory does not exist protects nothing and fails silently, and nothing in the
repository detected that. `tools/check_team_scope.py --validate` now rejects such
an entry, so the second instance cannot be introduced quietly.

**Correction, 2026-10-08: the split this finding anticipated has landed.** The
readings above are kept as the record of 2026-10-06 and are not restated. The
public IR is now the package `flagquantum/core/ir/`, whose `__init__.py` is the
former module byte-for-byte plus one deepened relative import, so the protected
entry is `flagquantum/core/ir/**` and it matches every file of the package. The
two spellings are not interchangeable: re-measured under the checker's
`fnmatch.fnmatchcase`, `flagquantum/core/ir.py` matched the module and not the
directory, and the glob matches the directory and not a sibling module. Both
entries appeared together, which is what `--validate` requires.

## Finding 3: `teams.compiler.owns` retains a path the repository removed

`teams.compiler.owns` lists `flagquantum/compilation/**`.
`docs/development/CPU_VERTICAL_SLICE.md` records that "the former
`flagquantum/compilation` transition package has been removed" and that
"architecture checks prevent the old paths from returning". The directory does not
exist.

An `owns` entry for a removed package is not harmful to users, but it is exactly
the permanent scaffolding that engineering decision principle 1 and the human
maintainability guardrails ask reviewers to subtract. It also misleads a reader
about where Compiler code lives.

The same list contains four entries that match nothing in this checkout:

| Entry | Status |
| --- | --- |
| `flagquantum/_compiler/**` | Absent, but authorized as a future private surface by the IR Phase 2 and Phase 3 entry packets and by `MULTI_LEVEL_IR_ARCHITECTURE.md` § 19.13. Keep. |
| `flagquantum/devices/**` | Absent, unauthorized, and never in history. See Finding 6. |
| `flagquantum/encoding/**` | Absent, unauthorized, and never in history. See Finding 6. |
| `flagquantum/compilation/**` | Removed by a documented migration. Remove this entry. |

`teams.core.owns` contains `flagquantum/ops/**`, which is also absent and is not
an allowed top-level directory in `architecture.toml`. `teams.docs.owns` contains
`docs/site/**`, which is absent.

A full-tree scan over all 2518 files found eight `owns` entries and one
`protected_paths` entry that match no path at all:

| Owner | Dead entry | Resolution in this change |
| --- | --- | --- |
| core | `flagquantum/ops/**` | Removed |
| compiler | `flagquantum/compilation/**` | Removed |
| compiler | `flagquantum/_compiler/**` | Kept; authorized by a written design |
| compute | `flagquantum/devices/**` | Kept; owner decision, Finding 6 |
| ecosystem | `flagquantum/encoding/**` | Kept; owner decision, Finding 6 |
| docs | `docs/roadmap/VNEXT_EXPERIENCE.md` | Kept; owner decision, Finding 6 |
| docs | `docs/development/DOCUMENTATION_STRATEGY.md` | Kept; owner decision, Finding 6 |
| docs | `docs/site/**` | Kept; owner decision, Finding 6 |
| policy | `requirements*.txt` | Removed; see Finding 5 |

## Finding 6: `teams.docs` owns no path that exists

All three `teams.docs.owns` entries are dead, and unlike the two entries removed
above, none is authorized by a document: `git grep` finds both file names and the
site directory only in `team-ownership.toml`. In practice `teams.docs` therefore
has no ownership at all, because `docs/**` is a *shared* path that already lets
every team edit documentation.

Two facts make this an owner decision instead of a cleanup. Removing the three
entries empties `owns`, which `policy_errors` rejects, so the change forces either
a defined scope for the docs team or the removal of the team block together with
its branch and worktree. And because `docs/**` is shared, the docs team has never
been the gate for documentation, so what it owns is unclear rather than merely
unwritten.

The same scan confirms that `tools/**` is ownerless under the most-specific rule:
it appears in no team's `owns` and in neither `protected_paths` nor `shared_paths`,
so every tool change is an integration change.

## Finding 4: the package allowlist and the ownership list disagree

`architecture.toml` `[package_layout].allowed_top_level_directories` lists
services, algorithms, benchmarking, compiler, core, deployment, drawer, ecosystem,
experimental, kernels, lindblad, noise, observables, qec, compute, remote,
runtime, simulation, testing, twin.

It does not list `_compiler`, `devices`, `encoding`, or `ops`. Yet
`team-ownership.toml` assigns `_compiler/**` to Compiler and `devices/**` to
Compute and `encoding/**` to Ecosystem, and both the IR Phase 2 and Phase 3 entry
packets authorize creating `flagquantum/_compiler/passes/` and
`flagquantum/_compiler/testing/`, with no corresponding `architecture.toml`
change proposed in those packets.

Two readings were possible and the difference matters:

1. The allowlist check ignores private and not-yet-created directories, in which
   case there was no conflict and only Finding 3's dead entries remained.
2. The allowlist is enforced literally, in which case creating
   `flagquantum/_compiler/` trips `tools/check_architecture.py`, and the Phase 4
   internal level work planned for that path needs an `architecture.toml` change
   in the same integration PR.

Reading 2 holds. `tools/check_architecture.py` walks the real `flagquantum/*`
directories that contain a `*.py` file and reports each one that is missing from
`allowed_top_level_directories`, and it reports each allowlist entry with no such
directory as well. The check is therefore bidirectional and literal.

The immediate consequence is that no `architecture.toml` change is needed today,
because `_compiler`, `devices`, `encoding`, and `ops` do not exist and so cannot
be "unreviewed". It also means Finding 6's `devices` and `encoding` entries can
never be satisfied by creating the directory alone: each needs an
`architecture.toml` edit in the same change, which is a second protected file.

## Finding 5: dependency and CI configuration is not in `protected_paths`

`docs/development/MULTI_TEAM_DEVELOPMENT.md` places "Dependency manifests, CI, and
release configuration" under Integration coordination. `protected_paths` covers
`pyproject.toml`, `requirements*.txt`, and `.github/**`, but not:

| Path | Why it is a dependency, CI, or release surface |
| --- | --- |
| `dependency-policy.toml` | Declares `schema = "flagquantum_dependency_policy_v2"`, the Python support matrix, the torch range, and every optional extra. `docs/development/DEPENDENCY_POLICY.md` calls it "the executable support matrix". |
| `setup.py`, `MANIFEST.in`, `environment.yml` | Packaging and release inputs. |
| `pytest.ini`, `.coveragerc`, `.pre-commit-config.yaml` | Test and quality configuration that governs whether a change can be verified. |

Editing `dependency-policy.toml` currently requires no integration coordination
under the machine check even though it can change the supported Python versions
and the core dependency range. This is a policy-versus-mechanism mismatch, not a
code defect, and it is the repository owner's call whether to close it.

The `protected_paths` entry `requirements*.txt` is dead: no such file exists and
`git log --all` finds none in any commit. It is also narrower than it looks, since
the one requirements-shaped file the repository produces is
`audit-requirements.txt` in the supply-chain job, which does not match the
pattern. Because the new liveness check has no exception list, the entry is
removed. The tradeoff is recorded rather than hidden: a `requirements.txt` added
later would not be protected until the pattern is added with it, and this check
does not detect an unprotected manifest, only a protection rule that protects
nothing. `pyproject.toml`, the actual dependency manifest, stays protected.

## Proposed assignment

### Assignments supported by written responsibility

| Path | Proposed team | Evidence |
| --- | --- | --- |
| `flagquantum/kernels/**` | simulation | `MULTI_TEAM_DEVELOPMENT.md` states `simulation/**` "belongs to Simulation and owns numerical algorithms/kernels". `architecture.toml` gives the layer name `optional_kernels`, and `[boundaries].kernel_forbidden` prevents kernels from importing simulation, runtime, compute, remote, and deployment, placing it as a leaf numeric layer in the Simulation family. |
| `flagquantum/lindblad/**` | simulation | The package is the plan and run facade over `flagquantum/simulation/lindblad.py`, which is already simulation-owned, and it imports `..simulation`. |
| `flagquantum/dynamic.py` | runtime | Three lines that re-export `DynamicCircuit` from `runtime.dynamic.circuit`, a path Runtime already owns. |
| `flagquantum/training.py` | runtime | Re-exports the training lifecycle types from `runtime.training_state`. |
| `flagquantum/operators.py` | core | Re-exports `GateInfo` and `gate_info` from `core.operator_schema`. |
| `flagquantum/errors.py` | core | Backend-neutral exception vocabulary with no imports of its own; `docs/architecture/ARCHITECTURE_DEPENDENCIES.md` makes Core the owner of backend-neutral contracts. |
| `flagquantum/gradients.py` | runtime | Parameter-shift gradient utilities for trainable circuits, importing only `core.ir` and torch. Runtime owns the training lifecycle and `runtime/training*` is the consumer family. The alternative reading is Simulation, because parameter shift is a numerical differentiation algorithm; the entry is Medium confidence. One observation for the decision: this module is used by `docs/guides/JIUDING.md` but appears in neither `flagquantum/__init__.py` nor `docs/public_api_v1.json`, so its stability class is currently unrecorded. |
| `flagquantum/testing/**` | integration | Cross-domain verification helpers importing core, compiler, simulation, runtime, remote, and noise. `tests/**` is already a shared path, so the same reasoning applies. |

### Assignments that require the repository owner to decide

These four domains map to no existing team, and no document assigns them.
`docs/architecture/ARCHITECTURE_DEPENDENCIES.md` explicitly limits
`flagquantum.core` to "backend-neutral IR, operator schemas, parameters,
configuration, and versioned contracts only", which rules out Core as a home for
application-level workflow code.

| Path | Files | Options | Consideration |
| --- | --- | --- | --- |
| `flagquantum/algorithms/**` | 20 | new specialist team; or leave integration-owned | Applications area. `docs/development/CODE_ORGANIZATION.md` lists it as "Reusable quantum and hybrid algorithms". Runtime currently imports it, so it is not a leaf domain. |
| `flagquantum/qec/**` | 11 | new specialist team; or simulation; or leave integration-owned | Composes Compiler control flow, Runtime feedback, Simulation kernels, Noise, and Remote, and imports `..runtime`. No single existing team owns that composition. |
| `flagquantum/twin/**` | 33 | new specialist team; or simulation; or leave integration-owned | Calibration-conditioned modeling; imports `..remote` and `..core`. Largest unowned domain. |
| `flagquantum/benchmarking/**` | 24 | runtime; or integration | `docs/development/CODE_ORGANIZATION.md` gives Runtime "planning, execution lifecycle, training, evidence", and benchmarking owns measurement methodology and claim ceilings. Handing it to Runtime puts measurement of Runtime inside Runtime. |

### Paths left integration-owned on purpose

These need no `owns` entry, because Integration already matches them. Recording
them prevents a future reader from treating them as oversights:

- `flagquantum/__init__.py`, `flagquantum/_api.py` — protected paths.
- `flagquantum/experimental/**` — its own README already states that it is not
  assigned to a team and that changes are integration changes.
- `flagquantum/version.py` — release metadata.
- `tools/**`, `contracts/**`, `.github/**`, `benchmarks/**`, `examples/**`,
  `docs/**` — governance, contracts, CI, and shared paths.

## Proposed change

Sections 1, 2, and 3 are applied by the change that carries this revision.
Section 4 is half applied: the protected-path checks are in place, the dead-`owns`
report is not.

### 1. Correct the protected-path entry (applied)

```diff
-    "flagquantum/core/ir/**",
+    "flagquantum/core/ir.py",
```

If the public IR is later split into a package, the `/**` form is added in that
change, which is the point at which it starts protecting something. Keeping a
dead entry in anticipation is the speculative scaffolding that engineering
decision principle 2 asks to avoid. `requirements*.txt` was removed under the same
rule, as recorded in Finding 5.

### 2. Add specialist ownership for the assignments supported by written responsibility (applied)

The diff below covers only the rows in "Assignments supported by written
responsibility" plus the two dead entries from Finding 3. It deliberately omits
the four undecided domains, so it can land without answering Open Questions 1
and 2. If the owner chooses new teams for `algorithms`, `qec`, and `twin`, those
become `[teams.*]` blocks with their own branches and worktrees, which is a larger
change than this inventory proposes. `flagquantum/testing/**` needs no entry
because Integration already matches it through `**`, and `flagquantum/gradients.py`
stays unassigned because its confidence is Medium and its stability class is not
recorded.

```diff
 [teams.core]
 owns = [
     "flagquantum/core/**",
     "flagquantum/circuit.py",
     "flagquantum/noise/**",
-    "flagquantum/ops/**",
     "flagquantum/observables/**",
+    "flagquantum/operators.py",
+    "flagquantum/errors.py",
 ]

 [teams.compiler]
 owns = [
     "flagquantum/compiler/**",
     "flagquantum/_compiler/**",
-    "flagquantum/compilation/**",
     "flagquantum/drawer/**",
 ]

 [teams.runtime]
 owns = [
     "flagquantum/runtime/**",
+    "flagquantum/dynamic.py",
+    "flagquantum/training.py",
 ]

 [teams.simulation]
 owns = [
     "flagquantum/simulation/**",
     "flagquantum/runtime/executors/**",
+    "flagquantum/kernels/**",
+    "flagquantum/lindblad/**",
 ]
```

`flagquantum/ops/**` is removed from Core because the directory does not exist and
`ops` is not an allowed top-level directory. `flagquantum/compilation/**` is
removed per Finding 3. `flagquantum/devices/**`, `flagquantum/encoding/**`, and
`docs/site/**` are left untouched: they are forward-looking assignments for
domains that do not exist yet, and the same review that resolves Finding 4 should
decide whether they stay. The `flagquantum/benchmarking/**` assignment is omitted
because it is Open Question 2, and the application-domain assignment is omitted
because it is Open Question 1.

### 3. Add the missing domain README

`flagquantum/lindblad/` is the only domain directory without a `README.md`. The
other nineteen have one, and `AGENTS.md` human maintainability guardrail 7
requires each target domain directory to describe what it owns, what it must not
own, its allowed dependencies, its public entry points, and the shortest path for
making and testing a typical change. Guardrail 8 makes that README the acceptance
condition for the boundary.

Assigning `flagquantum/lindblad/**` to Simulation without its README would
transfer ownership of a domain that a new contributor cannot enter. The README
should be part of the same change, and it should record the three walls already
measured in `flagquantum/simulation/lindblad.py`: the CPU-only hard rejection, the
dense `dim × dim` Hamiltonian with a fixed-step integrator, and the static
Hamiltonian with no batching.

### 4. Make the class of defect detectable

Two checks are applied. Both live in `tools/` and `tests/unit/`, so both require a
full checkout.

**Fixture test for protected paths.** `PROTECTED_FIXTURES` in
`tests/unit/test_team_scope_policy.py` asserts that the repository's own
protection predicate classifies each path as protected:

```python
PROTECTED_FIXTURES = (
    "flagquantum/__init__.py",
    "flagquantum/_api.py",
    "flagquantum/core/_artifacts.py",
    "flagquantum/core/ir.py",
    "team-ownership.toml",
    "architecture.toml",
    "contracts/cudaq-export-contract.toml",
    "docs/architecture/decisions/ARCH_009_NATIVE_CPU_OPERATOR_BOUNDARY.md",
)
```

This is the check with teeth. Reverting `flagquantum/core/ir.py` to
`flagquantum/core/ir/**` fails two tests, so the fixture converts Finding 2 from
reasoning into evidence and keeps the answer correct if a later change adds or
removes a pattern.

**Dead-pattern report.** `tools/check_team_scope.py --validate` now rejects a
`protected_paths` entry that matches no file and reports each `owns` entry that
matches no file. Protected entries are rejected because protecting nothing is
never intentional; `owns` entries are reported because two of the remaining six
name a path that a written design authorizes but no change has created yet, and
the other four are Open Question 5.

The split is deliberate. An exception list inside `team-ownership.toml` would
have added a concept to the policy file, and engineering decision principle 2
asks for the smallest mechanism that satisfies the requirement. A hard failure on
`owns` would instead have required resolving Finding 6 in this change. The report
keeps every dead entry visible in CI output, and
`test_dead_ownership_entries_are_exactly_the_reviewed_set` pins the exact set, so
a newly dead entry fails immediately. That is the property that would have
surfaced `flagquantum/compilation/**` when the package was removed.

## Open questions for the repository owner

1. Which team owns `flagquantum/algorithms/**`, `flagquantum/qec/**`, and
   `flagquantum/twin/**`? One application-domain team, three specialist teams, or
   continued integration ownership?
2. Does `flagquantum/benchmarking/**` belong to Runtime or to Integration?
3. ~~Is the `architecture.toml` allowlist enforced literally for private
   directories?~~ Answered by the full-checkout verification: yes, and both
   directions are checked. Creating `flagquantum/_compiler/` therefore needs an
   `architecture.toml` change in the same integration PR. No action is required
   today, because the directory does not exist.
4. Should `dependency-policy.toml` and the packaging and CI configuration files
   join `protected_paths` (Finding 5)?
5. What scope does `teams.docs` have (Finding 6)? Its three `owns` entries match
   no file and no document authorizes them, but removing them empties `owns`,
   which `policy_errors` rejects. Either the team gets a real scope, or the team
   block and its branch and worktree are removed. Until then the entries are kept
   and reported by the dead-pattern check.
6. Does `flagquantum/gradients.py` belong to Runtime? The written-responsibility
   evidence is Medium confidence, and its stability class is not recorded, so it
   was left unassigned rather than guessed.
7. Does `flagquantum/testing/**` stay integration-owned? It is reachable through
   `teams.integration.owns = ["**"]`, and no document proposes a specialist owner.

## Consequences

- The change is a governance change only. No module moves, no import changes, no
  public API change, no capability maturity change, and no `IR_VERSION` change.
- Team branches and worktrees for any new team are new operational resources. If
  the owner chooses new teams, they must be created with their branches recorded
  in `team-ownership.toml` before a team session starts, because
  `MULTI_TEAM_DEVELOPMENT.md` requires a session to verify its branch against that
  file.
- The `flagquantum/core/ir.py` correction, the specialist assignments, the
  `lindblad` README, and both detection checks landed together; none of them
  depended on an open question. Open Questions 1, 2, 5, 6, and 7 remain open, and
  resolution of Open Question 5 shrinks `REVIEWED_DEAD_OWNS` in
  `tests/unit/test_team_scope_policy.py`.
- Until the remaining assignments land, every change to those domains is an
  integration change. That is the current de facto policy, so leaving it in place
  is not a regression; it only means the parallel lines of work described in the
  parity programme cannot each have their own team.
