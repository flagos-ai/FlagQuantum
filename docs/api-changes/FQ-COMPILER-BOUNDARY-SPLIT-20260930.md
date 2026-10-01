# Compiler boundary responsibility split

## Decision and authorization

Status: **proposed, not approved.** This document records the boundary defects,
the evidence, and the intended end state so that the integration owner, the API
owner, and the compiler domain owner can approve or reject it. No code change is
included, and no approval is claimed. Implementation must not land before that
approval.

The change is proposed under `docs/development/PUBLIC_API_PROTECTION.md` and
`docs/development/MULTI_TEAM_DEVELOPMENT.md`, because the intended end state
touches a cross-domain contract type and a protected architecture policy. It is
the precursor to the ADR named in the parity backlog as W6-01; the ADR should
record the approved outcome of this document rather than re-derive it.

Two decision sets currently disagree about Compiler, and no machine gate notices.
`contracts/long-horizon-architecture-v1.json` declares `compiler` may depend only
on `core`, while `architecture.toml` permits Compiler to import anything not on a
seven-entry deny list — and `noise` is not on that list. The same pair of files
declares `runtime` may depend only on `core`, while `architecture.toml` carries a
self-cleaning eight-file allowance for Runtime importing Compiler. This document
resolves both in one direction and states what may not change.

## Problem and affected user journey

### The package owns six responsibilities, and three of them are other domains'

`flagquantum/compiler/` is 31 Python files and 10,380 lines. Its stable contract is
six names (`flagquantum/compiler/__init__.py:15-22`): `CouplingMap`, `compile`,
`lower_noise_model`, `optimize`, `route_to_topology`, `schedule_layers`. Everything
else is reached by deep module path or not at all.

Measured by that boundary, the package contains six distinct responsibilities:

| Group | Modules | Lines |
| --- | --- | --- |
| Canonical optimization and layer scheduling | `pipeline.py` | 242 |
| Routing, coupling maps, directed topology | `routing.py` 808, `directed_topology.py` 466, `topology_legalization.py` 274 | 1,548 |
| Target legalization | `target_legalization.py` 318, `native_gate_legalization.py` 302, `schedule_legalization.py` 246, `direction_legalization.py` 204 | 1,070 |
| External text emission and re-parsing | `target_conformance.py` 351, `target_emission.py` 234, `qcis.py` 232, `openqasm.py` 214 | 1,031 |
| Physical plan, evidence, artifact compilation | `physical_plan.py` **970**, `compilation_evidence.py` 340, `artifact_compilation.py` 302, `target_artifact.py` 123 | 1,735 |
| Noise lowering, operator-lowering registry, private hybrid slice | `_hybrid/` **4,361**, `operator_lowering.py` 242, `noise.py` 129 | 4,732 |

`physical_plan.py` is 970 lines against a `default_module_line_ceiling` of 1250
(`architecture.toml`), a 78% occupancy that makes it the largest module in the
package and the one a contributor is least able to change safely. `_hybrid/` alone
is 42% of the package.

The repository's own definition of the domain is narrow: "Compiler owns program
transformations over Core semantics ... Compiler does not run the program"
(`flagquantum/compiler/README.md`), and "Compiler transforms programs; Runtime
organizes execution" (AGENTS.md, Engineering Decision Principle 4). Three of the
six groups above do not clearly satisfy that definition, and the disagreement is
resolved today by nobody: there is no ADR that assigns them.

### Three other domains depend on a Compiler type that is not a public API

`CouplingMap` is defined at `flagquantum/compiler/routing.py:15`, exported by
`flagquantum/compiler/__init__.py`, and absent from `docs/public_api_v1.json` — it
is not Stable Core, not a documented contract, and not covered by
`tests/api_contract/`. It is nonetheless the interface between Compiler and three
other domains, at five import sites:

| Importer | Import |
| --- | --- |
| `flagquantum/runtime/dynamic/routing.py:5` | `CouplingMap`, `route_to_topology` |
| `flagquantum/deployment/cloud.py:22-25` | `CouplingMap`, `compile`, and `compiler.openqasm.emit_openqasm`, `compiler.qcis.emit_qcis` |
| `flagquantum/deployment/routing_evidence.py:10` | `CouplingMap` |
| `flagquantum/remote/qpu/azure.py:11` | `CouplingMap` |
| `flagquantum/remote/qpu/braket.py:70` | `CouplingMap` |

Three further sites are inside the package (`topology_legalization.py:13`,
`artifact_compilation.py:23`, `physical_plan.py:13`).

The Runtime site is instructive. `runtime/dynamic/routing.py:6-8` takes a
`CouplingMap` parameter whose only use is to forward it to `route_to_topology`, so
it sits legitimately on the program-to-plan composition boundary the allowance
comment describes — but it still has to import a *contract type* from Compiler to
express its own signature. A type that four files outside Runtime and one
allowlisted file inside Runtime depend on cannot stay owned by the domain that
consumes it.

`contracts/long-horizon-architecture-v1.json` sets `shared_contract_owner` to
`core`, and its `core_contracts` migration track is **in progress** with the
completion evidence "portable cross-domain program, capability, audit
request/result/evidence, and provider contracts are Core-owned". A type that five
files in three other domains depend on is exactly such a cross-domain contract, and
it is owned by the domain that is supposed to be its consumer. The
`compiler_convergence` track is marked **complete** ("one pipeline can be replaced
without Runtime or user API changes"), which cannot be true of a type whose
definition lives in Compiler while Deployment and Remote construct it.

### Compiler imports Noise, and the deny list does not say so

`flagquantum/compiler/noise.py:10-15` imports `DeviceNoiseProfile`, `KrausChannel`,
`NoiseModel`, and `thermal_relaxation_channel` from `flagquantum.noise`.

`contracts/long-horizon-architecture-v1.json` states `compiler may_depend_on =
["core"]` and `noise may_depend_on = ["core"]`. Compiler importing Noise violates
the first. No check reports it, because `[boundaries] compiler_forbidden` in
`architecture.toml` is `["_compiler", "compilation", "runtime", "deployment",
"simulation", "compute", "remote"]` — `noise` is absent, and
`tools/check_architecture.py:530-537` only enforces the listed entries.

The omission is not cosmetic. It means the deny list is a record of what someone
remembered to forbid, not of what the layer order implies, and the two lists have
already diverged. A contributor reading `architecture.toml` cannot tell whether
`compiler → noise` is permitted or an oversight, and today the answer is
"permitted by accident".

### `_hybrid/` is private and migration-scoped, and QEC depends on it

`flagquantum/compiler/IMPLEMENTATION.md` describes `_hybrid/` as "the private
structured-program semantic slice used to migrate program-level control flow and
ordered quantum effects into vNext ... intentionally absent from public exports".

Two production files import it:

- `flagquantum/qec/dem.py:34` — `from ..compiler._hybrid import INDEX, capture_source, lower_dynamic_program`
- `flagquantum/qec/repetition.py:8` — the same import

A migration slice with production dependents in another domain has no removal
condition, which contradicts AGENTS.md Engineering Decision Principle 1
("Compatibility adapters must have a named owner, documented scope, removal
condition, and target version"). `_hybrid/` is also the single largest block in the
package at 4,361 lines, so its migration status materially affects where a
contributor looks for the real compiler.

### The declared layer order is enforced by nothing

`architecture.toml` has a `[layers] order` table listing nine layers. No tool
reads it. `tools/check_architecture.py:269` reads only
`package_layout.allowed_top_level_directories`; the strings `layers` and `order`
appear nowhere else in the checker.

That is the mechanism behind the three findings above. `allowed_top_level_directories`
lists 20 directories, and boundary checks exist for seven of them (`core`,
`compiler`, `runtime`, `services`, `noise`, `simulation`, `kernels`). These 13 have
no import-direction check at all:

`algorithms`, `benchmarking`, `deployment`, `drawer`, `ecosystem`, `experimental`,
`lindblad`, `observables`, `qec`, `compute`, `remote`, `testing`, `twin`.

Deployment, Remote, QEC, Testing, and Twin each import Compiler internals today
(`deployment/cloud.py`, `deployment/routing_evidence.py`, `remote/qpu/azure.py`,
`remote/qpu/braket.py`, `qec/dem.py`, `qec/repetition.py`,
`testing/correctness.py:12-14`, `twin/experiment.py:12`), and none of it is
policy-checked.

### Runtime still reaches into Compiler in eight places

`[long_horizon_boundaries] runtime_compiler_import_allowed` names eight Runtime
files, and the comment above it states the intent: "Runtime may invoke Compiler
only at explicit program-to-plan composition or maintained raw-program
compatibility boundaries ... New imports remain blocked; do not invent a duplicate
Core contract merely to make this list shorter."

The list is self-cleaning — `tools/check_architecture.py:632-639` errors on an
allowance that is no longer used — but it has no removal target. Nothing states
which of the eight should disappear, by when, or what replaces each. Combined with
a `compiler_convergence` track marked complete, the repository simultaneously says
the migration is finished and that eight call sites remain.

### Affected user journeys

This is an internal boundary decision, so the affected journeys are contributor
journeys rather than end-user paths:

1. A contributor adding a routing option reads `flagquantum/compiler/README.md`,
   changes `routing.py`, and cannot tell that `CouplingMap` is a contract four
   files in other domains construct — so a signature change breaks Deployment and
   Remote with no policy signal.
2. A reviewer asked to approve a new `flagquantum/<domain>/` module cannot check
   import direction, because 13 of 20 directories have no rule to check.
3. A contributor migrating hybrid control flow reads `_hybrid/README.md`, sees
   "private", and edits it without knowing QEC depends on two of its exports.
4. A maintainer planning W6 (DAG plus pass manager) has no approved answer to
   "which module owns the pass framework", because three of the six responsibility
   groups in the package have never been assigned.

## Evidence

| Claim | Evidence |
| --- | --- |
| Compiler depends only on Core (contract) | `contracts/long-horizon-architecture-v1.json`, `domains.compiler.may_depend_on == ["core"]` |
| Compiler imports Noise (reality) | `flagquantum/compiler/noise.py:10-15` |
| The deny list omits `noise` | `architecture.toml`, `[boundaries] compiler_forbidden` (7 entries, no `noise`) |
| The checker only enforces listed entries | `tools/check_architecture.py:530-537` |
| `CouplingMap` is not a public API | absent from `docs/public_api_v1.json`; no `tests/api_contract/` coverage |
| `CouplingMap` is cross-domain | `deployment/cloud.py:22`, `deployment/routing_evidence.py:10`, `remote/qpu/azure.py:11`, `remote/qpu/braket.py:70` |
| Core owns shared contracts | `contracts/long-horizon-architecture-v1.json`, `shared_contract_owner == "core"`, and `core_contracts` completion evidence |
| `_hybrid/` is private and migration-scoped | `flagquantum/compiler/IMPLEMENTATION.md`; `flagquantum/compiler/README.md` lists it under "Structured hybrid programs" |
| QEC imports `_hybrid` | `flagquantum/qec/dem.py:34`, `flagquantum/qec/repetition.py:8` |
| Layer order is unenforced | `tools/check_architecture.py:269` reads only `package_layout`; no `layers` reference exists in `tools/` |
| 13 of 20 directories are unchecked | `architecture.toml` `allowed_top_level_directories` (20) versus the 7 prefixes tested at `tools/check_architecture.py:524-593` |
| The Runtime allowance has 8 entries and no removal condition | `architecture.toml` `[long_horizon_boundaries] runtime_compiler_import_allowed`; `compiler_convergence` track `status == "complete"` |
| Package size | `flagquantum/compiler/*.py` 6,019 lines; `flagquantum/compiler/_hybrid/*.py` 4,361 lines; 31 files, 10,380 total |
| Module ceiling | `architecture.toml`, `default_module_line_ceiling = 1250`; `physical_plan.py` 970 |

## Alternatives considered

**A. Keep the boundary as it is and document it.** Rejected as the default
outcome, not on principle. It is the only alternative that requires no work, and
it is defensible for the three legalization groups, which plainly transform
programs. It is not defensible for `CouplingMap`: documenting an undeclared
cross-domain type does not give it an owner, a stability level, or a test, and the
`compiler_convergence` track already claims a completeness that four external
constructors contradict.

**B. Move emission (`openqasm.py`, `qcis.py`, `target_emission.py`,
`target_conformance.py`) to Ecosystem.** Attractive because Ecosystem already
translates external representations, and it would leave Compiler purely
transformational. Rejected for now: Ecosystem's stated rule is that external
library objects stop at their owning boundary, and these modules translate
FlagQuantum IR to text, not external library objects. Moving them would also
break four in-repository importers (`deployment/cloud.py`,
`testing/correctness.py`, `twin/experiment.py`, plus `compiler/__init__.py` via
`target_emission`) for no user-visible gain. The narrower fix — give emission a
declared entry point — is preferred, and the ecosystem question is left as an open
question rather than decided here.

**C. Move `CouplingMap` to Core.** Proposed below. The risk is that Core is the
protected layer, and adding a type invites imports from everywhere. Mitigated by
requiring the moved type to stay non-Stable and by adding the direction rules in
the same change, so Core cannot silently become a hub.

**D. Inline the eight Runtime call sites.** Not possible as a single change. Four
of them (`runtime/execution.py`, `runtime/planner/__init__.py`,
`runtime/planner/noise_selection.py`, `runtime/noise_registry.py`) are on the
program-to-plan composition path that the allowance comment explicitly permits.
Only the ones that exist to consume compiled IR rather than to invoke compilation
are removable, and each needs its own analysis.

**E. Add the layer-order check without changing any dependency.** Rejected as
insufficient alone — it would immediately fail on `compiler → noise`. But it is
the right second step, and the two must land together or the check will be
disabled on first contact.

## Old and proposed behavior

This document changes no runtime behavior. It proposes a target ownership map,
an enforcement gap to close, and an approval path.

| Surface | Today | Proposed |
| --- | --- | --- |
| Canonical optimization, scheduling | Compiler | Compiler, unchanged |
| Routing and coupling maps | Compiler owns `CouplingMap`; Deployment and Remote import it from Compiler | Core owns the coupling-map contract type; Compiler, Deployment, and Remote all consume it from Core |
| Target legalization | Compiler | Compiler, unchanged |
| OpenQASM/QCIS emission and re-parsing | Compiler internals, reached by deep path from Deployment, Testing, Twin | Compiler, with a declared entry point that those three import instead of `compiler.openqasm` / `compiler.qcis` |
| Physical plan, evidence, artifact compilation | Compiler | Compiler, unchanged in ownership; `physical_plan.py` split by responsibility without new public concepts |
| Noise lowering | Compiler imports the Noise domain; the deny list omits `noise` | Decision required: either `noise` is added to `compiler_forbidden` and re-homed, or `compiler may_depend_on` is amended. The two records must agree |
| `_hybrid/` | Private, migration-scoped, with two QEC dependents | Either QEC's exported need is promoted to a declared Compiler entry point, or the dependency is removed and `_hybrid`'s removal condition is recorded |
| Layer order | Declared in `architecture.toml`, read by no tool | Enforced, with the 13 unchecked directories covered by an explicit rule each |
| Runtime → Compiler | 8 allowed files, no removal condition, track marked complete | Each of the 8 classified as permanent composition boundary or removable; the removable ones get a removal condition |

## Source and behavioral compatibility impact

No behavior changes in this document. The implementation it proposes has these
impacts, in decreasing order of risk:

1. **`CouplingMap` relocation.** Moving a class between packages breaks every
   in-repository import of it. Five external sites are known
   (`runtime/dynamic/routing.py:5`, `deployment/cloud.py:22`,
   `deployment/routing_evidence.py:10`, `remote/qpu/azure.py:11`,
   `remote/qpu/braket.py:70`) plus three internal ones
   (`topology_legalization.py:13`, `artifact_compilation.py:23`,
   `physical_plan.py:13`). It is not a Stable Core name, so no protected public API
   changes, but `fq.compile(..., coupling_map=...)` accepts it and
   `flagquantum.compiler.CouplingMap` is currently importable — that import path
   must keep working for at least one release, through a deprecated alias with a
   named owner and removal version.
2. **Emission entry point.** Adding a declared entry point is additive. It does not
   break `compiler.openqasm` / `compiler.qcis`, which should stay importable until
   the four internal callers move.
3. **Layer enforcement.** Newly enforced rules can fail on existing code. The
   13-directory coverage must be added rule by rule, and each rule must be
   satisfied by the current tree or accompanied by a recorded allowance with a
   removal condition. A rule that fails on first run and is then exempted is worse
   than no rule.
4. **`compiler → noise`.** Adding `noise` to `compiler_forbidden` fails
   immediately. Resolving it means either moving noise lowering behind a Core-owned
   contract type or amending the long-horizon contract — and the second is a
   contract change needing the same approval as this document.

## Migration example

Nothing here is executable yet. The shape of the intended end state, for review:

```python
# Today: Deployment and Remote reach into Compiler for a non-public type.
from flagquantum.compiler import CouplingMap          # deployment/cloud.py:22
from ...compiler import CouplingMap                   # remote/qpu/azure.py:11

# Proposed: the cross-domain contract is owned where shared contracts live,
# and the Compiler re-export remains for one deprecation window.
from flagquantum.core import CouplingMap              # canonical

# Today: Testing and Twin reach into emission by deep module path.
from flagquantum.compiler.openqasm import emit_openqasm   # testing/correctness.py:12
from flagquantum.compiler.qcis import emit_qcis           # testing/correctness.py:14
from ..compiler.openqasm import emit_openqasm             # twin/experiment.py:12

# Proposed: one declared entry point covers both formats.
from flagquantum.compiler import emit_target_text
```

`emit_target_text` is illustrative, not proposed as a name: the compiler owner
settles the spelling under AGENTS.md rule 9, and this document should not fix a
public name that it does not also implement.

The deprecation alias has an owner (compiler maintainers), a scope
(`flagquantum.compiler.CouplingMap` only), a removal version, and a test that fails
when the alias outlives it.

## Versioning

The `CouplingMap` relocation is a **minor** change under
`docs/development/PUBLIC_API_PROTECTION.md`, because the name is not Stable Core and
no documented signature or serialized schema changes. The re-export alias is the
compatibility path and must be removed in the release after it is deprecated.

The layer-enforcement work is **patch** scope and changes no API.

Any amendment to `contracts/long-horizon-architecture-v1.json` is a **contract
version** question, not a documentation edit: `schema` and `version` fields exist
for that purpose, and a `may_depend_on` change must be recorded as a new version
with the reason, not as an in-place string edit.

## Documentation and tooling impact

- `flagquantum/compiler/README.md` and `IMPLEMENTATION.md` both need the approved
  responsibility map, because they are the contributor's first stop.
- `flagquantum/compiler/AGENTS.md` needs the same map in its own terms.
- `docs/architecture/decisions/README.md` gains the ADR this document precedes, and
  the mapping-rules table gains a row for the layer-order check with its real
  coverage.
- `architecture.toml` gains the missing boundary rules; `tools/check_architecture.py`
  gains the layer-order enforcement and the new directory coverage.
- `tests/unit/test_architecture_boundaries.py` gains the cases that fail when a
  newly covered directory imports upward.
- `contracts/long-horizon-architecture-v1.json` is amended only if the
  `compiler → noise` question is resolved toward Noise, and then as a versioned
  change.
- No capability entry changes. This work moves ownership, not capability levels,
  so `capability-maturity.toml` should not be touched.

## Owner and approvals

- Owning domain: `integration`, with `compiler` as the affected domain.
  `docs/api-changes/**` is a shared path, and
  `python tools/check_team_scope.py --team integration --files
  docs/api-changes/FQ-COMPILER-BOUNDARY-SPLIT-20260930.md` passes.
- Required approvals before implementation: **integration owner** (architecture
  policy and `architecture.toml`), **API owner** (`CouplingMap` ownership and the
  deprecation window), and **compiler domain owner** (which of the six
  responsibility groups stay).
- Sequencing: this document, then the ADR, then the enforcement change, then the
  relocations. Enforcement must not precede the relocations for the rules that
  currently fail.
- Implementation note for the owner: the `compiler → noise` question should be
  answered first, because its answer determines whether the layer-order check can
  be turned on at all, and because it is the one item here that requires amending a
  versioned contract rather than a policy file.
