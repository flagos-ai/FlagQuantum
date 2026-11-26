# Independent structure migration acceptance

Verified on 2026-11-26 (Asia/Shanghai), on branch
`refactor/compute-local-memory-owner`, for [draft PR #1331](https://github.com/flagos-ai/FlagQuantum/pull/1331).
The [execution record](../INDEPENDENT_STRUCTURE_MIGRATION.md) identifies retained
#1084 families and compatibility retirement gates. This is acceptance of the
independently audited migration, not of the unfinished integrated architecture.

## Inputs and scope

- Main: `7278efd00f6768fbbacff6a41eb4dec57a59783e`.
- #1084 base: `03017ddf96dd2d9604f2dbc4d2cdfe2f3837f878`.
- Audited #1084 head: `8b1aca5be01cfd15ace0cb831221c8f03a48dbba` (open, draft).
- 442 explicit mapping rows, covering 432 source paths and 439 target paths.
- Latest twelve-path PR increment: no relocated-source intersection or new target collision.
- Full PR diff: only the partial extraction from `simulation/density_matrix.py`
  intersects; all five extracted helpers are AST-identical across its base/head.
  The modified engine remains in place.

An additional normalized AST audit covered 1,522 relocated top-level definitions.
The five inspected differences are import splitting in Remote package preparation
and deployment preflight, the worker archive root's extra directory depth,
canonical observable ownership resolution, and the entropy helper's Core owner.
These do not change numerical algorithms. This audit complements behavioral tests;
it does not replace them.

## Behavioral verification

Python 3.12.14, Torch 2.14.1, macOS arm64. CPU thread counts are fixed to one for
regression workers. Live Kaiwu smoke is excluded; optional hardware/dependency
fixtures report their own skips.

- `tests/unit tests/api_contract tests/team`: **19,269 passed, 957 skipped**
  (394.12 seconds; four workers; live Kaiwu smoke excluded).
- Root test modules and local training lifecycle integration: **1,591 passed, 193 skipped (119.32 seconds)**.
- Fresh-process compatibility cases include canonical-first ordinary imports,
  shared provider state, historical pickles and type identity, class annotations,
  lazy metadata imports, and offline source-archive workers.
- Ruff, Black, strict mypy, public API, architecture, vocabulary, ownership,
  documentation links/source, repository language and hygiene all passed.
  Frozen contracts, public API snapshots and raw scientific evidence are unchanged.

The expanded root regression initially found stale deployment mock and PCA example
imports and missing workshop script entries in the example index. These were
corrected before the final rerun, preserving the assertions and numerical examples.

## Installed artifact verification

The declared setuptools PEP 517 backend built the native CPU wheel offline:

`flagquantum-0.4.0-cp310-abi3-macosx_11_0_arm64.whl`

SHA-256: `f18b20df5d3d94c3f6a3d6a5572c7f8019b876fce5a4dcae0949bbba9bd87672`.

A temporary isolated virtual environment installed that wheel without changing the
shared development environment. Existing dependency installations were reused.
From outside the checkout, verification passed for CPU circuit execution, Module
gradients, canonical/historical leaf identity, memory snapshot pickle identity,
native CPU extension loading, the benchmark console entry and metadata-only
Realtime/kernel catalog imports without Torch/Triton initialization.

This artifact check qualifies this local platform only. No accelerator/QPU,
multi-node capacity or performance improvement is claimed. Rust remains deferred.
GitHub CI results must be checked for the pushed migration head; older green
results for the allocator-only commit do not qualify this expanded change.
