# PR #4 Merge-Readiness Audit

> Audit date: 2026-08-23
>
> Pull request: [#4 — integrate MPS runtime and FlagOS accelerator boundary](https://github.com/FlagQuantum/FlagQuantum/pull/4)
>
> Base: `origin/main@e65adba619eaac25b18c7c71e3e8ac43634e8336`
>
> Audited implementation: `fe736ebb112d97b4ba4d8796670cca2a9023e23a`
>
> Machine-readable companion: [`pr4_merge_readiness.json`](pr4_merge_readiness.json)

## Decision

**Conditionally ready to merge; do not merge until the repository-required checks and the review-lane approvals below are satisfied.**

The implementation is internally coherent and the accelerator boundary is suitable for the FlagOS ecosystem. Core installation remains PyTorch-only, Torch-FL activation is lazy and isolated, native CUDA remains an independent first-class path, and the public API is snapshot-governed. The audited implementation passed the local CPU, single-GPU, and two-GPU validation described below.

The main risk is not an identified runtime defect. It is reviewability: this is a historical integration change containing runtime code, tests, research evidence, benchmark results, papers, noisy simulation, Quafu integration, MPS/TN work, and FlagOS integration. It must be reviewed as an integration release, not as a normal feature PR.

GitHub-hosted checks are currently unable to start because of an account-level Actions billing/spending restriction. That state is classified as `external_blocked`, not as a successful or failed code check. It must not be represented as green CI.

## Scope and size

Relative to the audited base, the implementation contains:

| Metric | Value |
| --- | ---: |
| Commits | 21 |
| Changed files | 1,111 |
| Additions | 1,098,767 |
| Deletions | 1,557 |
| Size of changed HEAD blobs | about 51.94 MiB |
| Production package files changed | 117 |
| Production package additions/deletions | 22,475 / 964 lines |
| Test files changed | 104 |
| Benchmark files changed | 638 |
| Artifact files changed | 91 |
| Paper files changed | 102 |

Most lines are evidence rather than installed code: benchmarks contribute about 788k lines, while artifacts contribute about 254k lines. The changed `flagquantum/` tree is about 1.58 MiB. Package discovery includes only `flagquantum*`; benchmark, artifact, paper, test, and documentation trees are not installed into the wheel.

## Architecture findings

### 1. Dependency ownership is correct

`flagquantum` core still declares only `torch>=2.5,<2.14`. Torch-FL is deliberately not a core dependency and is not imported by `import flagquantum`, backend status queries, CPU execution, or CUDA execution.

The only production module allowed to import `torch_fl` is `flagquantum/runtime/platforms/flagos.py`. `architecture.toml` and `tools/check_architecture.py` enforce that boundary. Missing or incompatible Torch-FL fails only when the user explicitly activates `flagos`.

This preserves the intended responsibility split:

- FlagQuantum owns circuit/IR semantics, planning, numerical contracts, quantum algorithms, and evidence.
- Torch-FL owns device registration, vendor routing, general operator support, stream/event/memory integration, and vendor runtime identity.
- A domestic vendor's CUDA-compatible implementation remains Torch-FL metadata; FlagQuantum does not branch on vendor names or link a vendor SDK.

### 2. CUDA behavior remains independent

CPU, CUDA, and FlagOS are separate `PlatformRuntime` implementations. Automatic backend selection does not promote a visible FlagOS device without workload evidence. Native CUDA discovery and execution remain based on `torch.cuda`; selecting or installing Torch-FL is not required for them.

The architecture gate also prevents new direct `torch.cuda` calls outside explicitly budgeted modules. This gives the existing CUDA fast path a controlled migration route without forcing it through Torch-FL.

### 3. Public API is governed, but has no remaining top-level growth budget

The v1 snapshot contains 60 stable exports, each mapped to a verification test. The root namespace remains below the emergency ceiling of 64 symbols.

However, two architectural budgets are effectively exhausted:

| File | Current lines | Budget | Assessment |
| --- | ---: | ---: | --- |
| `flagquantum/__init__.py` | 140 | 140 | no further direct growth |
| `flagquantum/api.py` | 827 | 830 legacy exception | compatibility debt; must shrink before expansion |
| `runtime/backends/mps/training_engine.py` | 1,753 | 1,753 legacy exception | split by lifecycle/responsibility before feature growth |
| `simulation/tensor_contraction.py` | 2,206 | 2,206 legacy exception | split planner/executor/evidence responsibilities before feature growth |

These are not merge blockers because the budgets are explicit and enforced. They are mandatory constraints for all work after this integration:

- no new convenience exports merely to shorten imports;
- new external frameworks live under `flagquantum.interop.<framework>`;
- new accelerators live behind platform/provider protocols;
- experimental distributed objects remain under `flagquantum.experimental`;
- new runtime behavior must be added to focused modules, not the four exhausted files.

### 4. Numerical and accelerator claims fail closed

The FlagOS path requires explicit selection, performs operator-profile preflight, records provider identity, and does not convert CUDA-backed reference evidence into domestic-hardware certification. CPU distributed tests establish semantics only; they are not used as scalability claims. Existing capability and benchmark audits preserve this distinction.

### 5. Optional framework policy is structurally safe but needs one follow-up

Qiskit, JAX, Braket, Quafu, Triton, visualization, and example dependencies are extras rather than core requirements. This is the correct installation design for domestic accelerators.

`dependency-policy.toml` does not yet enumerate all provider extras already present in `pyproject.toml` (including the newly added Quafu extra). This does not change installation behavior, but it weakens the policy file as a complete source of truth. Create a small follow-up PR that derives or validates provider extras against `pyproject.toml`; do not expand PR #4 further for this cleanup.

## Validation evidence

The following validation was completed on the audited implementation commit `fe736eb` before this documentation-only audit commit:

| Gate | Result |
| --- | --- |
| Architecture, operator manifest, runtime schema, documentation source of truth, correctness certification, capability maturity, required-check contract, repository hygiene, import budget | passed |
| `pr-default` | 557 passed, 11 skipped |
| `pr-runtime` | 137 passed, 6 skipped |
| Distributed CPU | 336 passed, 16 skipped |
| Benchmark/release contracts | 113 passed, 45 skipped |
| A800 one-GPU correctness, CUDA initialization, intrinsic performance gate | passed |
| A800 two-GPU NCCL forward, backward, training, hardware correctness | passed |
| A800 two-GPU semantic tests | 3 passed on each rank |
| A800 MPS backward / forward | 3 passed / 1 passed |
| A800 distributed intrinsic performance gate | passed |
| GitHub-hosted required checks | external blocked before runner startup |

The GPU validation used disposable containers and did not modify the shared `tovx/flagquantum:0.2.0-dev` image.

The audit commit changes documentation only. Any subsequent executable, dependency, workflow, profile, benchmark-evidence, or generated-source-of-truth change invalidates inheritance of the implementation validation and requires the appropriate gates to run again.

## Human review lanes

Review by directory and contract, not by the 21 historical commits. A reviewer may cover multiple lanes, but every lane must have a named approval in the PR conversation.

| Lane | Primary scope | Required reviewer question |
| --- | --- | --- |
| A — product/API | `flagquantum/__init__.py`, `api.py`, IR, docs snapshots | Is the stable surface coherent and migration-compatible? |
| B — accelerator boundary | `runtime/platforms`, backend registry, FlagOS reference evidence | Is Torch-FL lazy, optional, vendor-neutral, and fail-closed? |
| C — numerical/runtime | statevector, density/noise, profiles, target execution | Are dtype, convergence, fallback, and capability claims honest? |
| D — distributed MPS/TN | MPS, tensor-network, transport, reverse/training paths | Is one logical workload genuinely sharded through forward and backward? |
| E — providers/noise | Quafu, noise models, trajectories, deployment | Are provider dependencies optional and provider data reproducible? |
| F — evidence/release | benchmarks, artifacts, paper, audit tools, CI | Is evidence bounded, attributable, non-promotional, and reproducible? |

## Merge strategy

Do **not** rewrite or mechanically split the current branch now. The exact integrated tree has already received CPU and GPU validation, while the historical commits contain cross-cutting code/evidence dependencies. Reconstructing multiple PRs would create new trees, discard the strongest validation provenance, and increase conflict risk.

Use this procedure instead:

1. Freeze executable changes on PR #4.
2. Record named approvals for lanes A–F.
3. Restore GitHub Actions service and require every check in `.github/required-checks.json` to complete successfully. Do not treat a billing-blocked check as waived evidence.
4. If `main` moves, update the branch and rerun all affected CPU/GPU gates; do not rely on the current mergeable calculation.
5. Prefer a **squash merge** with a release-grade message summarizing the six lanes. The resulting tree is reviewable as one integration baseline, while historical development commits remain available on the source branch/PR.
6. After merge, require narrow PRs with one capability owner, one evidence scope, and no unrelated paper/artifact payloads.
7. Move future large reproducibility corpora to versioned release assets or a dedicated evidence repository once tests can consume immutable checksummed references without network access.

## Merge checklist

- [ ] PR head contains no executable changes after the audited implementation, except changes that were revalidated.
- [ ] Lane A approval recorded.
- [ ] Lane B approval recorded.
- [ ] Lane C approval recorded.
- [ ] Lane D approval recorded.
- [ ] Lane E approval recorded.
- [ ] Lane F approval recorded.
- [ ] All checks in `.github/required-checks.json` are successful on the final head.
- [ ] No required check is skipped, manually relabeled, or inferred from a billing-blocked run.
- [ ] `origin/main` still matches the reviewed base, or the updated tree has been revalidated.
- [ ] Squash message preserves runtime, accelerator, numerical, provider, evidence, and known-limitations scope.
- [ ] Follow-up issue exists for dependency-policy synchronization.
- [ ] Follow-up issue exists for decomposition of exhausted legacy modules.

## Final assessment

PR #4 is a valid integration baseline, but it is not a template for future PR size. Its architecture is sufficiently clean for FlagQuantum to work with Torch-FL and later external frameworks without making those frameworks core dependencies. Its merge should be controlled by explicit review lanes and required checks; after merge, maintainability depends on enforcing namespace, dependency, module-size, and evidence-budget boundaries already present in the repository.
