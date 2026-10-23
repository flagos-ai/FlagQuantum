# QEC-RESTRUCTURE-004: circuit declarations and execution composition

Status: implemented for review under the user's explicit batch-004 approval.
Owner: Core, with physical evidence-path changes in a separate Integration commit.
Classification: `single_device_fast_path`. No numerical algorithm, public schema,
code/decoder registration, dependency or DynamicCircuit implementation changes.

## Branch and review boundary

`refactor/qec-restructure-004` starts at freshly fetched `origin/main`
`d2f689a4146cdb494784e78a8683972a2caa5dad`. The approved 001/002/003 commits are
replayed on that main. Their PRs #735, #737 and #739 remain open, so the separate
004 PR targets `refactor/qec-restructure-004-base`, the frozen prerequisite tip
`344a2fb83`. Its review diff therefore contains only 004.

After those prerequisites merge, replay only the 004 commits onto current main,
retarget this PR to main, and run required mainline CI before merge. This work
performs no PR merges. Maintainers had already merged #731 into the mainline
ancestry; this batch does not modify its PR, branch or core/runtime/compiler
implementation. The new main change since 003 is the separately merged #733
DynamicCircuit contract-test baseline, retained without modification.

## Implemented responsibilities

- `circuits/memory.py` owns the existing source builder, measurement handles,
  detector/observable layouts, and `MemoryCircuit`. It only consumes code records
  and algebra. It neither lowers programs nor imports execution, compiler,
  runtime, DEM construction or simulation.
- `execution/repetition.py` owns the original feedback controller, source and
  repetition experiment. `execution/noise_sweep.py` imports that same runner
  directly, without changing its finite-shot loop.
- `execution/sampling.py` owns lowering plans, noise placement, measurement
  records and measurement-to-detection parity. Draw ordering, source anchors,
  instruction/measurement order and seed consumption are unchanged.
- `execution/context.py` composes circuit declarations, DEMs and per-basis maps.
  It retains eager model construction and performs no shot execution.
- `execution/signatures.py` owns the existing forced-fault injectors, lowering
  and reference execution, together with the original shared fault vocabulary.
  DEM construction consumes those exact functions/constants; the signature module
  does not import DEM construction, avoiding a reverse dependency/cycle.
- Flat `circuit.py`, `context.py`, `sampling.py` and `repetition.py` explicitly
  re-export their former objects, helper imports and constants. `dem_construction`
  retains its analytic mechanism/model logic and re-exports the moved execution
  functions and constants. DEM package conversion remains batch 005.

The [circuit README](../../flagquantum/qec/circuits/README.md) and
[execution README](../../flagquantum/qec/execution/README.md) state ownership,
allowed dependencies and a short edit/test workflow. The execution initializer
remains empty of experiment imports. The existing QEC root facade remains eager;
there is no claim that a cold root import avoids infrastructure.

## Identity, serialization and extensions

The 108 ordered QEC root exports, signatures, dataclass fields, protocols,
exception messages and registry behavior remain protected by the unchanged 001
baseline. Both rotated surface classes remain independent and retain the 003
identities. Neither code nor decoder registry implementation changes in 004.

Moved classes, functions and their own methods/accessors retain the historical
module and qualified names. Existing imports and the new implementation paths
refer to the same objects. The existing narrow `codes._compat` metadata helper is
reused; there is no new wrapper class, reducer, serialization schema or
library-managed safe-global allowlist. Historical private helper imports remain
reachable, including forced-signature entry points at `dem_construction`.

Core owns these compatibility paths. No removal release is approved; removal
requires a separately approved public-API/deprecation and serialized-artifact
migration proposal. Plugins keep using existing public imports and the unchanged
single registries. Physical source filenames change: source tools and documentation
must follow the new owner or a method's code filename rather than assuming a
retained class `__module__` is its physical source location. The existing builder's
docstring example is still executed; its coverage gate verifies the actual owner
instead of changing the entry or skipping its example.

Trusted pre-004 fixtures are captured from unchanged `344a2fb83` before moving
implementation. They retain 190 class/function/method globals, 197 module/name
entries, and five representative records: memory declaration, decoder context,
decoder inputs, measurement map and tensor-backed measurement samples. Tests
load old pickles, round-trip the globals/records and load a PyTorch checkpoint
with an explicit caller-owned allowlist of those records' nested dataclasses.
This is targeted compatibility evidence, not a new general checkpoint contract.

The separate JSON fixture retains four complete declaration/source/layout/map,
forced-source/signature and matrix-vs-circuit DEM scenarios, plus 15 sampler and
feedback scenarios. Sampling covers zero, finite and certain fault rates,
measurement records and detector/observable bits; feedback covers all three
existing modes. Raw seeded sample equality is scoped to the recorded Stim 1.16.0
and arm64 machine architecture, matching the existing sampler's reproducibility
caveat. Other supported engine/machine combinations retain the unchanged
independent statistical/reference tests rather than a false byte-equality claim.
The 001/002/003 fixtures are unchanged.

## Runnable user workflow

The example constructs a memory declaration, reads decoder maps, samples actual
measurement/detection records and runs all three existing feedback modes:

```bash
pip install -e '.[stim]'
PYTHONPATH=. python examples/qec/memory_layout_and_sampling.py
```

Expected output:

```text
Same circuit class: True
Context class: True
Measurements, detectors, observables: 9 8 1
Detector map shape: (8, 9)
Historical pickle class: flagquantum.qec.circuit
Noiseless measurements: (4, 9)
Detection events, observable flips: 0 0
compiled_lookup 0.0 (0, 0, 0)
offline_pauli_frame 0.0 (0, 0, 0)
runtime_decoder 0.0 (0, 0, 0)
```

The PR description includes the complete runnable Python code. The existing
`examples/qec/restructure_compatibility.py` remains a core-only workflow with
`Correction observables: (0,)`, `Logical error rate: 0.0` and
`Decoded data: (0, 0, 0)`.

## Validation

- All 33 combined compatibility scenarios pass on Python 3.10.21 / PyTorch
  2.13.0, Python 3.11.16 / PyTorch 2.10.0, and Python 3.12.14 / PyTorch 2.13.0.
  Final checkpoint and declaration-isolation assertions also pass in targeted runs.
- The full QEC/reference suite with Stim 1.16.0 and PyMatching 2.4.0 completes
  with **1,834 passed, 1 skipped** in 234.79 s. The skip is the original
  gate-noise case with no applicable `cx` fault.
- All 53 complete public-docstring, maturity, docs-source and repository-hygiene
  cases pass. No example entry, snapshot or failing assertion is disabled.
- 73 original class/function ASTs, including nested bodies and local imports,
  are identical. DEM construction's original assignment ASTs also match; only
  their owning files/import paths and explicit metadata assignment change.
- QEC alignment 499 checks, complete vocabulary retirement counts, parity rendering,
  docs source-of-truth and Stable Core public API checks pass. No API snapshot,
  retirement count, tolerance, capability status or schema changes.

A standard isolated sdist and wheel-from-sdist build succeeds using the declared
PyTorch 2.10 build dependency. Both artifacts contain all eight Python modules
under `circuits/` and `execution/`. A fresh Python 3.12.14 / PyTorch 2.10.0
environment installs the final wheel, with imports verified from site-packages
and tests run outside the source checkout using `--import-mode=importlib
--noconftest`. The installed variants are checked explicitly:

| Installed optional SDKs | Result |
| --- | --- |
| Neither | 32 passed, 1 skipped (Stim sampling unavailable) |
| PyMatching only | 32 passed, 1 skipped (Stim sampling unavailable) |
| Stim only | 33 passed |
| Stim and PyMatching | 50 passed, including all 17 adapter cases |

The new complete sampler/feedback example and the existing core-only golden path
print their expected results from the installed wheel. Black, Ruff and ownership
checks pass. Other operating systems and every Python-minor/extra cross-product
are not certified by these local checks.

The full default `pytest -m "smoke or unit" -q` run completes with
**14,666 passed, 22 failed, 6 errors, 305 skipped, 3,141 deselected** in 677.62 s.
Every failure/error node exactly matches the separate pre-004 base comparison at
`344a2fb83`, where those same 28 nodes reproduce in 15.53 s. There are **no new
failure/error nodes**. They also match the retained 001 inventory: native CPU
source-checkout expectations, evidence/history assertions, local-network checks
and Gloo setup. No unrelated implementation, assertion or public-API snapshot is
changed to hide them; the repository-wide default gate is not green.

After #735/#737/#739 merge, replay only 004 commits onto current main, retarget
this PR and run required mainline CI before merge. No later restructuring batch
is implemented or authorized by this PR.
