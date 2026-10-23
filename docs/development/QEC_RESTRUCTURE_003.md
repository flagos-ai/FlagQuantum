# QEC-RESTRUCTURE-003: code families and surface ownership

Status: implementation for review under the user's approval of batch 003.
Owner: Core; physical source-evidence maintenance has its own Integration commit.
Classification: `single_device_fast_path`. No numerical algorithm, public schema,
dependency, decoder registration or supported execution mode changes.

## Branch and review boundary

The branch `refactor/qec-restructure-003` starts at freshly fetched `origin/main`
`19c8570f9b0124ae78bbf50500771ddbd802548b`. Approved 001/002 commits are replayed
on that main before 003. PRs #735 and #737 remain unmerged, so the separate 003
PR targets `refactor/qec-restructure-003-base`, the frozen prerequisite tip
`1b920d65c`. This makes its diff contain only 003, excluding both prerequisite
changes and unrelated mainline changes. The base is a review dependency, not a
request to merge a separate integration PR. After #735/#737 land, retarget 003
to current main, reconcile prerequisites and run required CI before merge.

Maintainers had already merged #731 into the fetched main as `3baccfe8e`.
This work does not modify that PR, its branch or DynamicCircuit implementation,
and performs no PR merges. Including a previously merged mainline commit is
not an additional merge of #731 by this work.

## Ownership and compatibility

`codes/__init__.py` now imports checks, protocol and families from their owners.
It retains the existing registry implementation and `_CODES` table, including
its globals namespace, validation, replacement semantics and default registration
order. Moving that table into a forwarding module would change the behavior of
old plugins/tests replacing `codes._CODES`; avoiding that change is deliberate.
No second registry or new registration API is introduced.

The package owns:

- `checks.py`, `repetition.py`, `steane.py`;
- `css.py`, `_css_helpers.py`, existing `matrices.py`;
- `colour.py`, `toric.py`, `qldpc.py`, `bicycle.py`, `reed_muller.py`;
- `subsystem.py`, `floquet.py`;
- `surface/rotated.py`, `surface/_rotated_frame.py`, `surface/zxxz.py`.

The registered/root `RotatedSurfaceCode` is defined in `surface/rotated.py` and
retains `flagquantum.qec.codes` metadata. The independent historical
`flagquantum.qec.surface.RotatedSurfaceCode` is defined in `_rotated_frame.py`.
They remain unequal objects/classes with equal checks and logical observables;
only the former is the registered `rotated_surface` factory. `ZxxzSurfaceCode`
and its checkerboard helpers belong to `zxxz.py` and retain their historical
`flagquantum.qec.surface` identifiers. The two existing lattice helper definitions
remain independent; there is no numerical/helper unification in this batch.

The old flat family modules explicitly re-export the same definitions, including
previous helper imports. Existing root exports and ordered `__all__` tuples are
unchanged. `_compat.py` performs only narrow historical module assignment on
moved definitions and methods/accessors owned by the moved class. It leaves
Python/typing-owned protocol methods unchanged. No wrapper class, reducer,
serialization schema or library-managed safe-global allowlist is added.
The 002 matrix aliases and lazy noise sweep remain unchanged.

Core owns these compatibility paths. No removal release is approved; removing
one requires a separate approved deprecation and serialized-artifact migration
proposal. Public plugins continue importing/using the same records and registry.
Physical source locations change: source tools must follow the new files or a
method's code filename rather than treating a class's retained `__module__` as
its physical source location. This is especially relevant to source inspection
and documentation links. Docstrings and executable examples remain attached to
the same public objects. The doctest gate explicitly verifies the mapping from
existing tested entries to physical modules, without skipping their examples.

The [code README](../../flagquantum/qec/codes/README.md) and
[surface README](../../flagquantum/qec/codes/surface/README.md) state dependency
rules and the shortest contributor workflow. The root QEC facade remains eager;
this PR makes no cold-import isolation claim.

## Retained evidence

The 001 API/signature/field/identity fixture and 002 tensor/method-pickle and
seeded-sweep fixtures remain byte-for-byte unchanged. New trusted repository
fixtures were generated from the unchanged pre-003 implementation before moving
code: `restructure_003_codes.pkl` retains 92 class/function/method globals plus
13 existing record instances, and `restructure_003_metadata.json` retains 246
module/qualified-name entries, including property accessors and protocol methods.
They are never generated from the moved implementation to bless changed paths.

Tests load old pickles, compare canonical object identities, round-trip the same
historical globals, verify both surface classes at distances 3/5/7 and verify
registration through old/root imports. Aggregate pickle byte layout may vary
with string memoization; existing record byte fixtures and identifiers remain
protected. Old paths, not an incidental aggregate memo table, define the contract.

A source AST comparison verifies 63 original class/function bodies unchanged,
excluding only local import paths that now resolve the owning CSS module.
The existing CSS, algebra, matrix, code-family, DEM, memory and seeded-feedback
tests exercise numerical behavior through the original public entries.

Physical paths in the vocabulary/alignment/parity ledgers and maturity artifacts
follow their new owners. No API identifier, vocabulary retirement count,
capability status, tolerance, schema or enforcement rule changes. Generated
capability/limitations/parity documentation is regenerated from its authorities.
Historical audit inventories and public API snapshots are not rewritten.

## Runnable user example

Run without optional SDKs, credentials or hardware:

```bash
PYTHONPATH=. python examples/qec/code_family_imports.py
```

The script uses both existing root/flat imports and the surface package aliases,
loads an old-style frame through pickle, reads Steane matrices and executes the
existing zero-noise memory workflow. Expected output:

```text
Registered family: True
Independent frame: True
Equivalent checks: True
Pickle paths: flagquantum.qec.codes flagquantum.qec.surface
Pickle identity: True
Steane matrices: 7 6 2
Logical error rate: 0.0
```

The PR description includes the complete runnable Python code as well.

## Validation

- The 26 combined 001/002/003 compatibility cases pass on Python 3.10.21 and
  Python 3.12.14 with PyTorch 2.13.0.
- All 79 cases in the complete compatibility, public-docstring,
  capability-maturity, docs-source and repository-hygiene files pass.
- QEC alignment: 493 checks pass. Parity rendering, docs source-of-truth and
  the complete qubit vocabulary check pass with unchanged retirement counts.
- The Stable Core public API checker passes on this latest-main base; no API
  snapshot is modified.

The full QEC/reference run with Stim 1.16.0 and PyMatching 2.4.0 completes with
**1,827 passed, 1 skipped** in 290.46 s. The skip is the existing gate-noise case
with no applicable `cx` fault, not a migration exclusion.

An isolated standard sdist and wheel-from-sdist build uses the declared PyTorch
2.10 build dependency. Both artifacts contain all 19 Python modules under
`qec/codes`. A fresh Python 3.12.14 environment installs the wheel with PyTorch
2.10.0; all 26 compatibility cases and the complete user example pass outside the
source checkout with `--import-mode=importlib --noconftest`. The imported package
is verified to come from that environment's site-packages, without source-path
insertion. Black, Ruff and ownership checks pass for changed files.

The full default smoke/unit run completes with **14,653 passed, 22 failed,
6 errors, 304 skipped, 3,141 deselected** in 741.46 s. Every failure/error node
matches the separately reproduced pre-003 base at `1b920d65c`: the same 22 failures
and six errors reproduce there in 10.22 s. There are **no new failure/error nodes**
in this batch. Those 28 nodes also match the retained 001 inventory (native CPU
source-checkout expectations, evidence/history assertions, local-network checks
and Gloo setup). No assertions, stable snapshots or unrelated implementations
are changed to conceal them; the repository-wide default gate is not green.

CI remains a review dependency: after #735/#737 merge, replay only the 003
commits onto current main, retarget this PR to main and run required CI before
merge. Python 3.11 and other operating systems are not certified by this local
run. No implementation from the next restructuring batch is included.
