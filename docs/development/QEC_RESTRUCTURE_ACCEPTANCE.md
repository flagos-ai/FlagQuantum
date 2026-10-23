# QEC restructuring: mainline delivery and acceptance

Status: delivered to `main` by [PR #749](https://github.com/flagos-ai/FlagQuantum/pull/749)
on 2026-10-23. Delivery merge commit:
`679e43de74e89779a16b9c0e687e5a3198a0a717`. Post-merge validation head:
`4bf7a154efe4d40bb2216981efe503d1fb0e3793`, after the subsequent runtime PR #752.
QEC sources and frozen fixtures are identical at these two main commits.
Core owns this record. This is structural/compatibility acceptance, with no new
numerical, accelerator or scalability claim.

## Delivered scope

The 001 audit and approved 002–007 implementation are present on main. The
intermediate 003–007 PRs used temporary bases; their merge status alone did not
prove mainline delivery. #749 is the integration into main. Earlier batch records
retain their historical implementation/review context and immutable measurements.

The delivered responsibility packages are `codes` (including `codes/surface`),
`circuits`, `dem`, `decoders`, `execution` and `adapters`. See the
[layout guide](../guides/QEC_PACKAGE_LAYOUT.md) for physical source owners,
allowed dependencies, old imports, serialization and plugin constraints.

## Remaining top-level Python files

At the accepted main commit there are **29** top-level QEC Python files:
23 historical import facades, five implementation leaves and the root API facade.
The tree is not claimed to contain only directories. The 23 historical facades
have no own class/function definitions; they directly re-export relocated objects.

| Historical top-level modules | Actual responsibility owner |
| --- | --- |
| `bicycle`, `css_code`, `floquet`, `qldpc`, `reed_muller`, `subsystem`, `surface` | `codes/`, including `codes/surface/` |
| `circuit` | `circuits/memory.py` |
| `chunks`, `decoding_graph`, `dem_alternatives`, `dem_circuit`, `dem_construction`, `dem_matrices`, `dem_text` | `dem/` |
| `belief_propagation`, `bposd`, `matching`, `registry`, `sliding_window` | `decoders/` |
| `context`, `repetition`, `sampling` | `execution/` |

The five files still containing implementations are deliberately retained leaves:

| File | Retained implementation |
| --- | --- |
| `gf2.py` | GF(2) row reduction, rank, span and nullspace. |
| `pauli.py` | Existing Pauli representation and validation. |
| `logical.py` | Logical-product certification and derivation. |
| `noise.py` | Repetition/phenomenological noise records and rate validation. |
| `types.py` | Existing event, syndrome, correction, frame and memory/sweep records. |

The root `__init__.py` retains the 108 ordered public exports. Keeping facade files
is required by this delivery's unchanged historical-import/pickle contract; it is
not evidence of duplicate algorithms. Compatibility entry points and all frozen
001–006 fixtures are retained. Their removal is a separate API/serialization
change; this acceptance does not approve a removal version or new migration.

## Post-merge CI

- Main smoke passed on both the
  [delivery merge](https://github.com/flagos-ai/FlagQuantum/actions/runs/39944611161)
  and the [post-merge head](https://github.com/flagos-ai/FlagQuantum/actions/runs/39947356273),
  building/installing a wheel and executing the maintained checks outside the checkout.
- [Development container publication](https://github.com/flagos-ai/FlagQuantum/actions/runs/39947356277):
  all four variants passed on the post-merge validation head. The previous delivery-head publication was superseded
  by the new main push: three variants succeeded, and its fourth was cancelled
  by the existing concurrency policy. The newer workflow verifies all four;
  the obsolete run is not rerun to overwrite newer image tags.
- [Integration full CI](https://github.com/flagos-ai/FlagQuantum/actions/runs/39939571800):
  38/38 passed before merge; required
  [single/two-GPU checks](https://github.com/flagos-ai/FlagQuantum/actions/runs/39939571738)
  also passed. These prerequisite results are distinct from the post-merge checks.

## Installed artifact acceptance on the post-merge head

Fresh isolated sdist and wheel-from-sdist builds pass. Both contain **74** QEC
Python modules whose content matches the post-merge validation source byte-for-byte.
QEC sources and frozen fixtures also match the validated integration head.

| Artifact | SHA256 |
| --- | --- |
| macOS arm64 CPython 3.10 ABI3 wheel | `e65c378c38786f3cace43e9a81abff0600a069b96d155e7425b134550fa0d941` |
| sdist | `89ae4fb2056fec2946708da1bb7ed524029b8bfbfa3658d6684c6a8dc35e827b` |

The wheel is installed into twelve independent environments outside the checkout,
with `PYTHONPATH` unset. Actual module locations must be inside each environment's
`site-packages`. Dependency checks, native extension loading, native Bell-state
shape and all six frozen compatibility suites pass.

| SDK profile | Each Python 3.10 / 3.11 / 3.12 result |
| --- | --- |
| Neither | 45 passed; original Stim-only sampling case inapplicable. |
| Stim only | 46 passed. |
| PyMatching only | 45 passed; original Stim-only sampling case inapplicable. |
| Both | 46 passed. |

Missing-SDK refusal behavior is actively checked. The Stim case runs where its
extra is installed; no check is disabled to hide a failure. The full source QEC
suite passes **1,820 tests**, with the one original no-applicable-gate-fault skip.
All **201** contract/maturity/documentation checks pass, as do the public API
snapshot, architecture, complete vocabulary ledgers, parity rendering and all
**499** QEC alignment checks. No threshold or assertion is weakened.
All **33** installed executions of the six documented user examples match exact
stdout. The sampling example runs with its declared Stim extra. Local artifact
coverage is macOS arm64, Python 3.10.21/3.11.16/3.12.14, PyTorch 2.10.0,
Stim 1.16.0 and PyMatching 2.4.0. Python 3.10 uses SciPy 1.14.1; 3.11/3.12 use
1.17.1/1.18.1. No broader local platform or accelerator claim is made.

Run the [installed-import example](../../examples/qec/installed_imports.py):

```bash
PYTHONPATH=. python examples/qec/installed_imports.py
```

Expected output:

```text
Public exports: 108
Independent rotated classes: True
Code pickle path: flagquantum.qec.codes
Frame pickle path: flagquantum.qec.surface
Correction: (0,)
Logical error rate: 0.0
```

For installed verification, invoke the script by absolute path from outside the
checkout, using the installed environment's Python with `PYTHONPATH` unset.

## Temporary branch/worktree cleanup

Removed the eight old `FlagQuantum-qec-restructure-*` worktrees, eleven local
`refactor/qec-restructure-*` branches and the three remote 003/004/005 baseline
branches. Remote deletion used exact-tip leases in one atomic push. Three local
backup snapshot refs and eight stale remote-tracking refs were also removed
after bundle verification. The active acceptance-record branch/worktree remains
for review; main and unrelated worktrees/branches are unchanged.

All eight old worktrees were clean; ignored files contained only reproducible
caches/egg-info, with no private results to preserve. A verified recovery bundle
preserves commits outside main; a 14-ref manifest records all old local and remote
snapshot tips. Other saved commits remain reachable from main history. The bundle
SHA256 is `4e208af3f29a35b566deccd3a3cd67e9edac8b92de56ad9797fd2c636f1ebaf2`.
Recovery artifacts and installation evidence are retained outside the removed
worktrees. Active acceptance work is retained until its PR is reviewed.
No unrelated worktree, branch, compatibility source or frozen fixture is removed.
