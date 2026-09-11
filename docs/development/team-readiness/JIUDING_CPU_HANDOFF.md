# Jiuding CPU adapter handoff — 2026-09-09

> Commit references below have been mapped to the publication history.
> Recorded outcomes and approval status are unchanged.

## Identity and delivery state

- Team: remote.
- Worktree: `FlagQuantum-vNext-execution`.
- Branch: `codex/vnext-team-execution-providers`.
- Original live-run baseline: `9c1b7e9f0ad3983d634aab6ecfbce0ccd9f009fe`.
- Updated integration baseline: `a1d5e6a15c09ed96232884f42b647e31c1fe3fb1` (includes the observable API changes).
- Implementation commit: `f78b0bad3674853556e7d57c1f0ea0f85c031d6e`.
- Integration authorized by the user after the other API edits were committed.
- The initial hook PATH issue is resolved by using the existing
  `FlagQAI/backend/.venv/bin` tool environment and the bundled Git fallback.
  Every commit hook passed, including architecture, API baseline, Black and both
  mypy checks. No hooks were bypassed or disabled.
- Unrelated untracked planning documents in the main worktree are excluded.

## Scope and interface

The user explicitly requested implementation of the already proven Jiuding
connection and a real FlagQuantum circuit task. The new concrete experimental
adapter is `flagquantum.remote.compute.jiuding.JiudingClient`, with workspace
discovery, observed-image listing, Python task submission, receipt-bound status,
shared JSON results and stop requests. It uses direct HTTP for both experiment
creation and Job launch; the temporary CLI dependency was removed after observing
its request shape against a loopback fake gateway with a dummy token.

No root exports, Stable Core signatures, QPU contracts, generic provider
protocols, runtime/simulation internals or dependency manifests changed. No
cross-team contract or new infrastructure dependency was introduced. The
adapter's receipt dictionary is experimental and is not a promoted public schema.
All changes belong to Remote or shared examples/tests/docs paths; team scope
prechecks passed. Existing fq.run is used inside the submitted Python process.

## Concrete live result

- Job: `33295f22-cce9-4c66-8782-c5fd23cea908`.
- Experiment: `2ce1d352-56e8-4de9-ab87-50140d3839ce`.
- Platform terminal state: `Succeed`.
- Allocation: one replica, 2 CPU cores, 2 GiB, 0 GPUs.
- Two-qubit complex64 Bell state probabilities:
  `[0.4999999701976776, 0, 0, 0.4999999701976776]`.
- Maximum absolute state error against expected: 0.
- Shared result JSON was retrieved and matched to its submitted run_id.
- No numerical fallback, precision downgrade, GPU or distributed claim.
- Evidence: `docs/development/evidence/jiuding_bell_cpu_20260909.json`.

An initial reduced experiment request returned HTTP 400. Its experiment name
was checked against the complete accessible experiment listing and found absent
before a corrected submission was made. The accepted request retains the known
platform fields and accelerator model even for zero-GPU tasks. No duplicate
successful job was created. Receipts from both attempts remain on the remote
shared snapshot at `/share/project/liuwei/fq-jiuding-vnext.c0yW36`.

## Verification

Latest integration verification supersedes the older environment failures below:

- On baseline `a1d5e6a15c09ed96232884f42b647e31c1fe3fb1` plus this adapter, Python 3.12.14 / PyTorch 2.13.0:
  **1227 passed, 14 skipped**, 1270 deselected, 24.72 s for the default smoke/unit
  suite; **30 passed** for the focused local/remote regression set.
- Bell example also ran locally on the updated API and returned the same
  probabilities with maximum absolute error 0.
- Local validation used Python `-S` and explicitly added the existing venv's
  site-packages without executing unrelated editable-install `.pth` hooks.
  This prevented imports from an old sibling worktree. Obsolete, untracked
  pyc-only backend directories were preserved outside the repository in /tmp.
- Git was placed on PATH for repository-hygiene tests. These were environment
  corrections, not changes to test expectations or product code.

Original remote validation before integration:

- Final focused regression: **30 passed** in 2.88 s, covering
  `tests/team/remote/test_jiuding.py`, `tests/test_local_fast_path.py`,
  `tests/test_cloud_providers.py`, `tests/test_amazon_braket_provider.py`.
- Default `pytest -m "smoke or unit" -q`: **1216 passed, 19 skipped,
  3 failed**, 1295 deselected, 112.51 s. This preceded two additional auth/worker
  tests included in the final focused run and the creation-field correction.
- All three failures reproduced against a separate archive of the unchanged
  integration baseline, in the same Python 3.12 / PyTorch environment:
  - `test_historical_api_aggregators_are_not_shipped`: baseline ships
    `flagquantum.api`, contrary to the existing test.
  - `test_exact_shape_cache_can_compile_more_than_dynamo_default_recompile_limit`:
    PyTorch Dynamo cache_size_limit reached.
  - `test_dynamo_specialization_limit_accounts_for_microbatch_variants`:
    this PyTorch lacks `config.recompile_limit`.
- `tools/check_architecture.py`: passed.
- `tools/check_capability_maturity.py`: passed; no catalog promotion.
- `git diff --check`: passed before final staging.
- Live cancellation was not exercised; stop request shape was observed with the
  platform CLI against a fake gateway and covered by an offline behavior test.

## Remaining boundaries

This is an existing-workspace/shared-storage CPU adapter, not a general remote
project upload service. A compatible image and shared code must already exist.
User script main() returns JSON; large tensors should be stored separately.
No GPU/multinode scheduling, training validation, image build, log streaming,
automatic partial-launch recovery, release certification or public API freeze.
The first journey intentionally avoids manual project/queue IDs and credentials
in subprocess arguments. Read `docs/guides/JIUDING.md` for use and failure behavior.

The scoped implementation and follow-up verification record are delivered on
the Remote branch for a normal non-fast-forward integration merge. The merge
does not publish a package or promote the adapter to a stable API.
