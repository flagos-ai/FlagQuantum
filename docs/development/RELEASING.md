# Preparing a FlagQuantum release

This runbook supports the existing package installation workflow: a user installs
the published `flagquantum` wheel and imports `flagquantum as fq`. It describes
maintainer preparation, not permission to publish the current development branch.
[Release policy](RELEASE_POLICY.md) is authoritative; this guide changes no
version, API, branch-protection or publication contract.

## Select and review the release revision

Prepare an intentional version-change PR against `main`. Edit only the existing
version source, `flagquantum/version.py`, for the package version; do not maintain
separate metadata or runtime versions. Choose the version and any PEP 440
prerelease suffix under the release policy. Review release notes, supported
Python and dependency bounds, migration notices, licenses and capability claims.

After merge, identify the exact candidate SHA on `main`. Every required check
in `.github/required-checks.json` must have reported success for that revision.
Missing, cancelled or failing checks block release. Do not infer success from a
different revision or create a patch branch to evade a red `main`. Coordinate
maintainer activity around the selected revision rather than relying on a moving
branch name. Hardware claims require their own scheduled hardware and benchmark
evidence; CPU tests alone do not establish scalability.

## Verify artifacts before publication

Use an isolated clean checkout of the candidate and the supported build
environment. Inspect `git status --short` and `git diff`; preserve unrelated work.
Do not clean a shared development worktree to prepare a release.

The package jobs in [CI](../../.github/workflows/ci.yml) own reproducibility,
artifact manifests and clean-install verification. Their existing local commands
include the following, run in a prepared build environment with empty output
directories. They create artifacts and are not publication commands:

```bash
SOURCE_DATE_EPOCH=1767225600 python -m build --no-isolation --outdir dist-a
SOURCE_DATE_EPOCH=1767225600 python -m build --no-isolation --outdir dist-b
python tools/verify_reproducible_builds.py dist-a dist-b
python tools/verify_distribution_artifacts.py dist-a
python tools/artifact_manifest.py --output dist-a/manifest.json dist-a/*.whl dist-a/*.tar.gz
```

Keep the same source revision, dependency environment and timestamp across both
builds. These commands do not replace CI's supported platform matrix, clean
installation checks, installed QEC profiles or license checks. Record the
revision, environment, checksums and actual results. A local build is verification
only; no fresh wheel or hardware validation is implied by this documentation.

## Tag and publish the reviewed revision

An authorized maintainer creates the reviewed release tag and GitHub release
against the verified candidate SHA. Review the target before publishing: the
existing [CD workflow](../../.github/workflows/cd.yml) starts on a published
GitHub release. It can also be explicitly dispatched for an existing release
tag. Neither trigger is a dry run.

CD validates the selected tag with the default, runtime and distributed CPU
tiers, builds one source distribution and native wheels for its declared Linux,
macOS and Windows matrix, then downloads those exact artifacts. The publisher
runs `verify_distribution_artifacts.py --release-matrix dist` before the PyPI
action. A failed job blocks its dependents. Keep the pre-publication policy gate
separate from these workflow checks: CD does not replace verification that the
candidate revision of `main` has every required check. Do not upload a separately
built local archive or bypass the existing workflow with a packaging tutorial.

## Resume development and communicate

Review a follow-up PR for the next development version in the same single
version source, using the project's chosen release sequence. Do not mechanically
copy another project's next-minor formula. Confirm the published version and
artifacts, retain the release notes and evidence, and communicate through the
project's authorized channels. This runbook defines no mailing list and grants
no authority to send messages.

The [source-role mapping](RELEASE_SOURCE_MAPPING.md) records the complete locked
upstream guide and the deliberate FlagQuantum policy adaptations.
