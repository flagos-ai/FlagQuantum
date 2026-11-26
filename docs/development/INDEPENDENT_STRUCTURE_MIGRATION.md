# Independent structure migration

The implementation now covers the independently auditable structure migration,
including product modules, development tools and repository configuration. The
allocator move in commit `08549f62` was the first package, not the scope limit.
The implementation migration and internal-forwarder retirement have passed final local acceptance. The retirement inventory records the retained public boundary and the only #1084-dependent internal files. This is a
structure change: it introduces no Rust implementation or performance claim.

## Frozen integration inputs

| Input | Revision |
| --- | --- |
| Main | `7278efd00f6768fbbacff6a41eb4dec57a59783e` |
| PR #1084 base | `03017ddf96dd2d9604f2dbc4d2cdfe2f3837f878` |
| Audited open, draft #1084 head | `9b1a8cd196f9d8b8427b382c404f66705cc57840` |
| Migration branch | `refactor/compute-local-memory-owner` |
| Review | [Draft PR #1331](https://github.com/flagos-ai/FlagQuantum/pull/1331) |
| Audit date | 2026-11-26, Asia/Shanghai |

The actual PR diff is its base-to-head diff. A main-to-PR comparison alone also
contains main changes absent from the feature branch and would overstate overlap.
The audit additionally checks the complete PR tree for new target collisions and
new consumers, including paths whose content is unchanged from the PR base.
Git trees, not the capped GitHub changed-files API, provide the inventory.
The final incremental check from `8b1aca5b` to `9b1a8cd1` covers seven changed paths,
with no relocated-source intersection or newly added target collision.
Target emulation retains its extension SDK target-description adapter in
`compute/remote/emulation.py` and delegates execution to the existing Runtime
entry. Moving that adapter into Runtime would violate the Ecosystem boundary.

Repeat the overlap and consumer audit if either input advances before integration.

## Authoritative implementation owners

| Family | Result |
| --- | --- |
| Direct devices | Platform contracts, registry, PyTorch/FlagOS/FlagGems adapters, target observation and allocator observation live in `compute/local/`. |
| External tasks | QPU, Jiuding and Kaiwu implementations live in `compute/remote/`. Remote owns extension-backed target emulation; Runtime owns deployment planning and execution. |
| Realtime | Protocol, channels, payloads, session and service live in `compute/realtime/`, retaining their standard-library-only boundary. |
| Kernels | Catalog, Triton and existing kernel adapters live in `simulation/kernels/`. Catalog imports remain metadata-only. |
| Dynamics | Lindblad numerics, generators, problem validation, integrators and adjoint live in `simulation/dynamics/`; workflow composition lives in `runtime/evolution/`. |
| Training | Module integration, training lifecycle and compilation orchestration live in `runtime/training/`; checkpoint persistence lives in `runtime/checkpointing.py`. |
| Results and profiles | Execution, MPS, trajectory and count results live in `runtime/results/`. Training/scale/capacity definitions live in `runtime/profiles/`; measured MPS profiling lives in `runtime/observability/`. |
| Operator semantics | Schema, algebra, observable models, channels, superoperators, registry data and embedding live in `core/operators/`. Measurement requests live in `core/measurement.py`, noise composition/profile in `core/noise.py` and `core/noise_profile.py`. |
| Registration | Circuit spelling/method installation lives in `compiler/construction/operator_registration.py`; immutable registry semantics stay in Core. |
| Algorithms | Existing main implementations are grouped into primitives, chemistry, many body, search, estimation, optimization, optimizers, linear algebra, quantum ML and error mitigation. Generic Hamiltonians belong to Primitives; spin models belong to Many Body; PCA and SVD belong to Linear Algebra. |
| Frontends | Torch capture lives in `compiler/frontends/torch/`; structured Python capture lives in `compiler/frontends/_structured/`. Existing formats remain in their already authoritative owners. |
| Services | The package maintains published entry points. Preparation belongs to Compiler; plan inspection/preflight to Runtime; ground-state/research workflows to Algorithms; Qmod adaptation to Ecosystem; synthesis composition/program/readout to Synth. The northbound capabilities aggregation belongs to `_api.py`. |
| Deployment | Remote owns package models and submission validation; Compiler owns routing evidence; Runtime owns deployment planning, dynamic sealing/rematch and measured count extraction. The old package is a facade. |
| JAX numerics | Statevector, MPS and tensor-network numerical modules live below their engines' `jax/` directories. Shared JAX primitives live in `simulation/numerics/jax.py`. |
| CPU adjoint numerics | Pure CX permutation/adjoint and scalar rotation VJP/seed functions live in Simulation. Runtime retains policy, shard state, communication, deferred execution and lifecycle dispatch. |
| Tools | 138 existing root tool implementations are grouped into `tools/ci`, `validation`, `architecture`, `docs` and `release`; old command paths forward to one implementation. |
| Root configuration | Architecture/team policies live in `config/`; the Conda recipe lives in `config/environments/`, with its editable path adjusted to the same repository checkout. |
| Workshops | Executable material lives in `examples/workshops/flagos2026/`; standalone guides live in `docs/guides/workshops/flagos2026/`. The six-notebook localization exception moves with the same notebooks. |

The cumulative 560-row [path map](migration/independent-structure-path-map.md) and
[machine-readable symbol ledger](migration/independent-structure-moves.json)
identify every relocated file and every partial extraction. An implementation
has one owner even when more than one historical public module exports it.
No empty directories are added for capabilities present only in #1084.

The existing integrators remain a cohesive `dynamics/integrators.py` module.
Splitting each function into a package would not improve this implementation's
boundary. Per-trajectory stream ownership stays in `runtime/trajectories/rng.py`:
choosing a deterministic stream by global trajectory ID is execution lifecycle.
It does not justify a new Trajectory-to-Simulation dependency. Existing QEC,
Fermion, Primitives, Twin and format families already in their intended owners
are retained unless a concrete relocation is recorded in the map.

## Independence and retained families

Every wholly relocated implementation is unchanged by the audited PR. The one
partially extracted PR-modified source is `simulation/density_matrix.py`: the
five embedding helpers are individually AST-identical between the PR base and
head. The rest of that file remains in place. No target in the migration map
collides with a PR-added target. Other compatibility imports remain available to
the frozen PR snapshot while its consumers are reconciled.

| Retained family | Concrete reason and integration action |
| --- | --- |
| Core IR; static/dynamic construction; registers and lowering | #1084 directly changes IR, `_construction`, register/component and lowering families. Integrate their complete program/type/validation closure before moving these implementations. Published `fq.Circuit` and dynamic entry points remain unchanged. |
| Rotation synthesis mathematics | #1084 directly changes/adds Synth rings, lattices, rotation protocols and Clifford+T implementations. Move this connected mathematical family to Compiler synthesis after integration; the independent public workflow move is already implemented. |
| Density-matrix, classical/basis and new numerical families | #1084 changes density-matrix implementation and adds extensive classical, basis, encoding and numerical components. Migrate their complete integrated family. Do not create absent implementations or overwrite PR additions with main's smaller family. |
| Drawer | #1084 substantially adds/changes diagrams, symbols, viewers and assets. Consolidate the integrated renderer family and asset paths together. |
| Native CPU ABI and build closure | `setup.py` keeps the current native module name/source paths. #1084 changes `MANIFEST.in`, installation/source-manifest tests and distribution verification. Relocate native sources only with the integrated ABI, wheel, sdist and installed-artifact acceptance. |
| Benchmarking and testing distribution | #1084 changes benchmarking implementations and introduces testing-tool consumers. The maintained benchmark console entry lives in `pyproject.toml`, which #1084 also changes. Move into the optional developer distribution together with metadata, installed entry points and independent-wheel tests. |
| Root `ci/` data and remaining policies | #1084 consumers include `.github/workflows/local-gpu.yml`, lane-dependency tests, Compiler documentation, CI and evidence documentation. Dependency policy itself is modified. Move these JSON/TOML files with all integrated readers; no duplicate policy files are introduced. |
| Compose and container closure | The PR tree's `.github/workflows/publish-dev-container.yml` refers to `compose.dev.yaml`. Its presence matters even if that file is unchanged from the PR base. Move Compose with workflow filters, project directory/name, build context, bind mounts and environment behavior verified. |
| Standalone source bootstrap | CI downloads `tools/ci_source_archive.py` and executes it before the repository exists. Keep its standalone implementation until the #1084 CI download/filter closure moves atomically; an import facade cannot bootstrap an absent package. |
| Existing root tools directly modified by #1084 | `check_cirq_interop_contract.py`, `check_dependency_policy.py`, `check_multi_level_ir_contract.py`, and `verify_distribution_artifacts.py` retain their current implementations. The PR additionally introduces `audit_register_component_scope.py` and `tools/development/`; place those against the integrated source. |
| Rust | Implementation, dependencies and experiments wait for #1084 closeout and architecture acceptance. Existing Rust priorities, boundaries and numerical/performance acceptance remain unchanged. |

A file retaining its current owner because it implements orchestration is not an
unmigrated numerical engine. Runtime exchanges, gradient reduction, layout plans,
provider submission lifecycle and training/checkpoint policy stay in Runtime.
Pure mathematical helpers extracted from those files are individually recorded;
partial rows never redirect the entire original module.

## Compatibility after internal retirement

The [itemized retirement inventory](migration/internal-forwarder-retirement.md)
and its JSON identify every removed path, all 100 public compatibility entries (including 16 pre-existing expert Compiler facades),
and the only three internal files awaiting #1084 consumer reconciliation.
They supersede the initial blanket facade policy of commit `5f8e97c77`.

283 tracked files, including all 138 historical root tool wrappers, are removed.
Old Kernel/JAX/algebra/Torch frontend trees are absent. Internal Runtime, service,
deployment and provider helpers use direct canonical imports. CI, hooks and
wheel test commands invoke the owning tool directories. No generic module alias
helper remains and canonical imports do not recreate historical namespaces.
Published facades share the actual classes/functions through ordinary imports.

Runtime imports exactly the same three reviewed frontend symbols directly from
`compiler.frontends.torch._module_builder` and `compiler.frontends.torch._training`.
The approved exact importer/name constraints remain; the intermediate compiler
module/training facades are removed. Frozen interface records are historical
contracts; path resolution maps their source identities without editing them.

The only internal leftovers are `compiler/_hybrid/model.py`, `verifier.py` and
their import parent: the audited #1084 tests explicitly import them. The incoming
callers must be reconciled at integration and those three files removed together.
They are not public API. Root command wrappers are not retained for future merge
conflicts; the map provides the deterministic incoming command replacements.

Public compatibility follows exact API, maintained namespace and serialization
obligations, itemized per file. Actual pickle bytes written by the pre-retirement
wheel prove the MemorySnapshot, CollapseOperator and LindbladPlan module paths.
Remote credentials explicitly forbid serialization, so their private old modules
are removed while their public namespace exports remain. Removing other public
entries requires an API-owner-approved versioned public migration under release
policy; there is no automatic 0.5.0 expiry.

Source identity mapping protects frozen evidence references, not runtime imports.
The guard rejects missing owners, duplicated definitions, resurrected retired
files/directories and absolute or relative imports of retired internal modules.
Unknown evidence paths do not acquire fabricated successors.

## Validation and evidence

The environment is Python 3.12.14 and Torch 2.14.1. Use the repository as
`PYTHONPATH`; CPU fixtures and fake providers do not establish physical GPU/QPU
support, distributed capacity or a performance improvement.

```sh
python -m pytest tests/unit tests/api_contract tests/team -n 4 -q \
  -p no:cacheprovider --ignore=tests/team/remote/test_kaiwu_live_smoke.py
python -m ruff check flagquantum tests tools
python -m black --check flagquantum tests tools
python -m mypy --strict --python-version 3.12 --ignore-missing-imports flagquantum
python tools/architecture/check_architecture.py
python tools/architecture/public_api_snapshot.py
python tools/architecture/check_qubit_vocabulary.py
python tools/architecture/check_team_scope.py --validate
python tools/docs/check_docs_links.py
```

The worker compatibility scenario exports a real package source archive and runs
both historical and canonical worker modules in isolated subprocesses. Additional
fixtures cover shared provider registry state, historical pickles, class type
hints, metadata-only imports, relocated Conda checkout resolution, forbidden
Realtime/Kernel dependencies and relocation-scanner negative cases. Existing
frozen realtime pickle bytes are retained. Ports/subprocess fixtures require an
environment permitting local sockets; sandbox denial is not a behavioral failure.

The [acceptance record](migration/independent-structure-acceptance.md) records
19,269 passed / 957 skipped in Unit/API/Team and 1,591 passed / 193 skipped in
root/local integration, eleven passing quality checks and installed-wheel verification.
The expanded migration is reviewed through draft PR #1331; CI must qualify its
new pushed head independently of the earlier allocator-only result. Frozen public API files, contracts and raw
scientific evidence are not regenerated. Historical recorded failures remain
failures. Native accelerator, distributed and #1084 integrated acceptance remain
separate gates.

Rollback is a normal commit revert of the atomic migration and its path-policy
updates. No persisted public payload, checkpoint schema or provider receipt is
rewritten by rollback.

The [internal-retirement acceptance record](migration/internal-forwarder-retirement-acceptance.md)
qualifies the cleaned tree separately from the preceding facade-based migration.

## Subsequent independent batches

The [example batch](migration/example-structure-migration.md) groups repository
examples by capability, deletes old example paths and updates operational
consumers without changing installed product exports. Historical contract
references resolve through the existing source ledger. Its stacked PRs receive
the same CI through explicit architecture branch selectors.

The [matrix-free batch](migration/matrix-free-structure-migration.md) places
linear algebra primitives under Numerics. A pre-move public-plan pickle proves
the only additional retained boundary: two concrete type locators. No numerical
operation is forwarded at an old path.

The [physical dynamics batch](migration/physical-dynamics-structure-migration.md)
places existing physical numerical functions under Dynamics and deletes three
old internal modules. Runtime consumes canonical numerical owners directly;
Core physical records and public execution results retain their identities.

## Follow-up delivery boundary

The example, matrix-free and physical dynamics batches are reviewed together in
one follow-up integration PR, based on the qualified #1331 head. Each batch keeps
its own path map, compatibility basis and focused validation. One complete CI
run must qualify their combined tree, including coverage ownership, installed
artifacts and public serialization. Passing a preview or an earlier batch does
not qualify that combined commit. The directly #1084-dependent migration remains
a separate integration boundary, and Rust work remains deferred.

The [follow-up acceptance record](migration/followup-structure-acceptance.md)
records complete local regressions, installed-wheel verification and the
remaining exact-commit CI and incoming-integration gates.

The [shared-kernel batch](migration/shared-kernel-structure-migration.md) gives
catalog admission and compiled complex-contraction primitives one owner under
Simulation Kernels. Old internal files are removed; no public compatibility
entry or eager package import is added.

The [experimental ownership closure](migration/experimental-implementation-ownership.md)
removes the remaining behavioral definitions from experimental entry modules,
keeping their shipped workflows and explicitly unstable status unchanged.

The [shared gradient numerics batch](migration/shared-gradient-numerics-migration.md)
places the one finite-difference quotient and step calculation in Simulation
Numerics. Public gradient requests, results and frozen source identifiers stay
unchanged. The old internal Core file is removed without a forwarding entry.

The [Runtime trajectory kernel batch](migration/runtime-state-kernel-ownership.md)
places trajectory collapse and static state application in Simulation. Runtime
retains seeds, feedback decisions, shot records and resource/evidence assembly.
Forward/reverse local block composition shares one Simulation tensor product.

The [density/instrument numerical batch](migration/density-instrument-numerical-ownership.md)
places the dense readout adapter, checked instrument weights and the shared
caller-generator draw in Simulation. Runtime keeps request admission, private
generator creation, result metadata and evidence assembly.

The same batch places raw dense probability extraction, joint reduction and
parity reconstruction in Simulation Numerics. Backend/catalog selection and
resource admission remain Runtime; the existing normalization tolerance moves
with its numerical kernel, unchanged.

The [Dynamics normalizer retirement](migration/dynamics-normalizer-retirement.md)
removes unused internal reexports from the evolution engine. Problem normalization
keeps its existing owner, and its test consumers import that owner directly.
Seven declared public evolution exports and their historical identities remain.

The [Simulation private-binding closure](migration/simulation-private-binding-retirement.md)
also retires unused Statevector and Stabilizer aliases. Production, tests and
benchmark producers import the defining modules directly; recorded benchmark
results, contracts and published type/function identities are unchanged.

The [QEC private-alias closure](migration/qec-private-binding-retirement.md)
retires nine unused imports while itemizing thirty-three existing frozen
namespace, replacement-owner and serialization obligations.

The [independent helper-binding closure](migration/internal-private-binding-closure.md)
combines QEC alias retirement with unused Compiler Topology and internal
Runtime Dynamic aliases; frozen public and historical QEC boundaries remain.
