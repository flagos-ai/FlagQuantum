# Remaining root-tool ownership

Integration declaration was committed before implementation. Owner: Integration.

Move all six remaining root-tool implementations to their real owners in one
complete closure. Remove every predecessor; do not add an import/executable
facade, copied implementation, symlink, sys.modules alias or old-path fallback.

| Current source | Unique target | Current source SHA-256 |
| --- | --- | --- |
| `tools/ci_source_archive.py` | `tools/ci/ci_source_archive.py` | `6fe212e8f394ff197489fff3ed273d2ad6ff903fc95c8661e045a73df2c533a5` |
| `tools/check_dependency_policy.py` | `tools/architecture/check_dependency_policy.py` | `e184c94c657edcf3284884cac9345b31f2aedbd2817a934b97408d07edd9c803` |
| `tools/check_cirq_interop_contract.py` | `tools/validation/check_cirq_interop_contract.py` | `9199bfffd22ca85f108cedca44ea118ebed8b5064c876cbe66e0a1f26927445e` |
| `tools/check_multi_level_ir_contract.py` | `tools/validation/check_multi_level_ir_contract.py` | `a58084ad485d121b8343324515e92b8bb408c7d7fc8a6b79b837b1caf57d2d6c` |
| `tools/verify_distribution_artifacts.py` | `tools/release/verify_distribution_artifacts.py` | `d704b56feaa99994a556d1639aa026f6032de4ba8bd522891ea6c80d1e0f84f5` |
| `tools/setup_flagquantum_dev_env.sh` | `tools/development/setup_flagquantum_dev_env.sh` | `24b007f4080a7db742e92e9a8443c98e31df72f7724781128df0d165df085a33` |

The prerequisite is qualified PR #1359, head
`0fdbe6db967e72be2fdb9e9b128ef221f332bd2b`, tree
`3684f04966d81df2d3970868e0710b97c632e501`. Its exact tree is carried on main
`29afa414a6362d381cb4c2992cd19fd024761f83`; its green checks do not qualify this
new head. Integration scope checks cover the complete proposed source/caller,
fixture, documentation and mapping paths before this implementation.

## Source and incoming semantics

The true #1084 fork base is `e48251b01754cecd783ddf0206e24f75583f7434`.
At reviewed pin `7d692c2e2a4572176466b7c56ba8d00de50b7548`, the source bootstrap
and environment script equal that base. Preserve main's evolved bootstrap bytes;
its physical location does not require the incoming program/type implementation.
The policy/Cirq tools and artifact verifier have evolved on both branches, while
the multi-level checker is unchanged on main versus that base. Move main's real
implementations now; reconcile incoming semantic contributions during actual
#1084 integration, without restoring old-root files or losing either side's work.

All 4,691 changed incoming blobs were verified. Current full import closure
covers 5,572 Python ASTs, including an alias-only source-network test missed by
filename-only searches; changed incoming import closure covers 3,374 ASTs.
The subsequent reviewed pin `8e51d3b52c4482589ce0425d97a9cdfeed54e7a3` adds ten
linear commits, 23 changed files and 17 Python files. All new bytes/ASTs/texts
were checked; all six inputs remain unchanged, targets absent and no new
root-tool references appear. This is scoped independence review, not merged
program/type/native or G0 qualification.

## Standalone bootstrap and Git object closure

Keep the source bootstrap and distribution verifier bodies byte-for-byte.
The bootstrap must run as a copied standard-library script before a checkout or
`tools` package exists. Update every pre-checkout GraphQL expression, git-show
repair/cleanup command, workflow caller and source fixture to its new path.
The incremental pack must explicitly contain the root, `tools` and `tools/ci`
trees and the helper blob. Including only the root/tools trees and helper fails
on a later commit whose ci subtree is unchanged versus the base. Exercise the
existing real-source pack, repair and complete fallback cases with nested
fixtures, including exact trees, binary files, symlinks, dirty-worktree handling
and a consumer with no prior objects. Do not weaken source verification or
substitute a package-import wrapper for bootstrap.

## Checker roots and recorded authorities

The three checkers move one directory deeper but retain the same repository
root and original data/scan inputs. Use the established direct-script entry
bootstrap for the two product-importing validators; module imports and aliases
must bind to canonical owners. Preserve explicit custom input arguments,
options, diagnostics and all original acceptance/refusal behavior.

Keep the complete multi-level IR 0.5 contract and its
`existing_authorities.reader = "tools/check_multi_level_ir_contract.py"` value.
The checker actually calls `_resolve_authority` to test declared physical paths;
reuse the existing `source_path` resolver with the six exact whole-file move
rows. This is recorded authority/source identity, not an executable old path.
Preserve the Core comments/docstring naming that same recorded authority,
original duration hints, approval states and raw scientific/historical records.
No new registry, contract or generalized path search is required.

## Development setup and current instructions

Keep the selected interpreter, VENV_DIR, .venv default, install commands,
options and refusal exits. Adjust only the actual canonical usage path and the
repository calculation to SCRIPT_DIR/../..; executable mode stays unchanged.
Actual Bash tracing from a foreign working directory must resolve the original
repository and default environment. Help, invalid-option and missing-interpreter
checks must not install anything. Add the local development owner README and
synchronize current workflow/hook commands, active validation instructions and
the existing architecture/validation/release/CI guides.

Existing contributor workflows after implementation, from the repository root:

```bash
python tools/architecture/check_dependency_policy.py
python tools/validation/check_multi_level_ir_contract.py
python -I tools/ci/ci_source_archive.py --help
python -I tools/release/verify_distribution_artifacts.py --help
bash tools/development/setup_flagquantum_dev_env.sh --help
```

The first two validate the same policy and IR contract; the remaining commands
show existing options without installation, source fetching or submission.
Preserve original default/help/error protocols except declared physical path
text. Keep substantive checker implementations, not forwarding entry points.

## Acceptance and remaining scope

Check committed body/AST/input/contract identities and all actual callers,
including import aliases, source fixtures, pre-checkout flows and future pack
ancestor completeness. Run all complete original consumer test files and the
real ownership, architecture, hygiene, documentation and source checks. Compare
full original collection and then the entire default node/outcome/multiplicity
census; do not cut tests or numerical inputs. Run local networking tests with
actual loopback permissions, recording failures and resource limits.

Qualify this committed head's actual CI source, complete archives/fallback,
installed/distribution impact, all original scientific call phases and admitted
resources, all applicable workflows and all 384 original CPU reports. Target
every executed job below 600 seconds including preparation/upload and excluding
queue; retain all failed or over-target observations rather than selecting a
favorable rerun. Dependency versions, scientific workloads, seeds, precision,
budgets, clocks, native ABI and public API remain outside this change.

Pytest/coverage configuration, root conftest, Benchmark results and other
validation/release artifact organization remain explicit unfinished root/G0
closures. The connected #1084 families, native relocation and independently
installed developer distributions also remain unfinished. Rust, merging, force
pushing and runner administration are not part of this work package.

Implementation is present. Local and own exact-head CI qualification remain
separate requirements; no completed qualification is claimed by this declaration.
