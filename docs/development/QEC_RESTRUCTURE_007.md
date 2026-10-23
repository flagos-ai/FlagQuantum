# QEC-RESTRUCTURE-007: documentation and installation closure

Status: prepared for review under the approved final batch.
Owner: Core. Classification: `single_device_fast_path` compatibility evidence;
no new numerical, accelerator or scalability claim.

## Stack and scope

This branch targets `refactor/qec-restructure-006`, PR #746, at `3cd2b5506a7af167418132c2658cf8ce82ba90b3`.
Both stack layers require all 38 full-CI jobs and the required single/two-GPU
checks; their exact-head results are recorded in their PRs. PR #744 supplies the DEM layer; #746 supplies
decoders/adapters. No PR is merged by this work. #731 is not edited or merged.

This batch changes documentation and a runnable example only. Library Python
sources, API exports, schemas, pickle identifiers, registries, numerical
implementations, dependency manifests, workflows and thresholds match the
validated parent. Current-head CI still runs the complete existing graph.

The [package guide](../guides/QEC_PACKAGE_LAYOUT.md) records the final ownership
map, historical imports, independent rotated classes, pickle/introspection
implications, plugin behavior, optional SDK boundaries and installation procedure.
The QEC README links each domain's ten-minute contribution path.

## Completed responsibility split

| Batch | Result |
| --- | --- |
| 001 | Immutable audit, proposal, 108-export public/behavior/pickle baseline and user example. |
| 002 | Code-owned CSS matrix record/extraction and executing noise sweep ownership. |
| 003 | Code families and `codes/surface`, retaining both independent rotated classes. |
| 004 | Declarative memory circuits and execution/sampling/signature composition. |
| 005 | DEM model, grammar, mechanisms, IR realization, chunks and graph. |
| 006 | Existing decoder algorithms/protocols, sole registry and optional PyMatching boundary. |
| 007 | Current layout/compatibility guidance, installed artifact verification and explicit compatibility freeze. |

The original 001 inventory remains a historical measurement; it is not rewritten
to describe the new tree. Current source contains 74 QEC Python modules, including
required compatibility aliases and shared leaves. No new public abstraction was
introduced to increase that count.

## Cleanup decision and compatibility freeze

All six responsibility directories and `codes/surface` have local ownership and
dependency guidance. The remaining flat aliases are still referenced by the root
facade, documented public imports, old pickles and plugins. `codes/_compat.py` is
used by relocated definitions to retain their old identifiers. These files are
active compatibility mechanisms, not unreferenced scaffolding eligible for
deletion. They are explicitly frozen, owned by Core, with no approved removal
release. Removal requires an API/deprecation and artifact migration proposal
before a target release can be assigned.

Normal Python package filesystem metadata changes where a file became a package;
retained class/function/method identifiers do not. The guide identifies actual
source owners for documentation and private instrumentation. Public registration
continues to use one code table and one decoder table; no shadow table, wrapper,
new reducer or library-managed safe-global list is introduced.

## Artifact and installation verification

Standard isolated sdist and wheel-from-sdist builds use the declared PyTorch 2.10
build dependency. Artifact membership is compared with every committed QEC Python
source byte. Both must contain all 74 modules, including every new initializer
and canonical owner, without stale `dem.py`, `decoders.py` or `adapters.py` files.

The same CPython 3.10 ABI3 wheel is installed outside the checkout in twelve clean
environments. `PYTHONPATH` is unset; import locations must resolve into each
environment's `site-packages`. Dependency consistency, native extension loading,
a native Bell-state shape check and all six frozen compatibility suites run.

| Installed SDKs | Compatibility result for each Python 3.10 / 3.11 / 3.12 |
| --- | --- |
| Neither | 45 pass; the original Stim-only sampling case is inapplicable. |
| Stim only | 46 pass. |
| PyMatching only | 45 pass; the original Stim-only sampling case is inapplicable. |
| Both | 46 pass. |

Missing-PyMatching behavior is actively asserted in absence profiles. The
Stim-only case executes in installed-Stim profiles; no check is disabled to
bypass a failure. Trusted old globals/instances/checkpoints, mid-stream resume,
exact results, class identities, TypeVar/decorator metadata and registration
behavior remain covered by unchanged 001–006 fixtures.

Local scope is macOS arm64, Python 3.10.21 / 3.11.16 / 3.12.14, PyTorch 2.10.0,
Stim 1.16.0 and PyMatching 2.4.0. Python 3.10 uses SciPy 1.14.1, a valid supported
dependency wheel; Python 3.11/3.12 use SciPy 1.17.1/1.18.1. Required CI separately
covers its declared Python/PyTorch jobs, optional integrations and platform-wheel
checks. Exact artifact hashes, build head and current-head CI links are recorded
in the PR rather than claiming every platform/version combination locally.

## Runnable installed-import example

Run [the example](../../examples/qec/installed_imports.py) from a source checkout:

```bash
PYTHONPATH=. python examples/qec/installed_imports.py
```

For installed verification, invoke that script by absolute path from outside the
checkout using the installed environment's Python, with `PYTHONPATH` unset.

Expected output:

```text
Public exports: 108
Independent rotated classes: True
Code pickle path: flagquantum.qec.codes
Frame pickle path: flagquantum.qec.surface
Correction: (0,)
Logical error rate: 0.0
```

All five prior user examples and this closure example are checked against their
exact documented stdout in core/both-SDK installed profiles where applicable:
11 executions per supported Python minor, 33 total. The Stim sampling example
runs with its declared extra. The PR embeds the complete closure code and output.

## Acceptance

The parent full QEC suite passes with 1,820 tests and one existing
no-applicable-gate-fault skip. Frozen fixtures and all library source bytes must
remain unchanged in this batch. Documentation/source links, example registration,
Stable Core snapshot, architecture, complete vocabulary ledgers, parity rendering
and all 499 QEC alignment checks remain required. No threshold is lowered.

This closes the approved structural plan for review. Merging and any future
public API evolution remain separate maintainer decisions.
