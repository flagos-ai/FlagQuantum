# API Change Proposal 085: Retire QEC historical submodule imports

## Status and target version

The API owner explicitly authorized direct retirement on 2026-10-23, accepting
incompatibility for the old import paths and related pickle artifacts. The same
directive requires this proposal PR first, directly against `main`, with no
implementation deletion or automatic merge.

**Target release: 0.4.0**, the next breaking minor after the current 0.3.1.
Retirement must not ship in a 0.3.x patch. This PR does not bump the version,
remove modules, change metadata, or claim retirement has shipped. Core owns the
implementation; Integration reviews the API contract and release boundary.

The approval is a narrow exception to the usual deprecation window in the
[release policy](RELEASE_POLICY.md): these 23 paths may be removed directly in
0.4.0 without a warning-only release. It does not authorize other API removals,
changes to serialized record schemas, or weakening the public API policy.
Implementation follows in separately reviewed PRs after this proposal is
reviewed and merged. No change to or merge of PR #731 is part of this work.

## Decision and motivation

The responsibility packages introduced by the QEC restructuring already own the
implementations. Twenty-three root-level modules now only re-export objects and
historical module globals. Their continued presence makes the directory look
less organized and preserves import and serialization obligations that the API
owner has now elected to retire.

Delete exactly the following 23 files during implementation. Every path in the
table is relative to `flagquantum.qec`; the target version for **every row is
0.4.0**. Replacement modules exist now unless explicitly marked **planned**.
Use `from flagquantum import qec` and its existing exports for ordinary usage;
the table also identifies expert module imports.

| # | Retired module / root file | Replacement import module(s) | Target |
| --- | --- | --- | --- |
| 1 | `belief_propagation` / `belief_propagation.py` | `flagquantum.qec.decoders.belief_propagation` | 0.4.0 |
| 2 | `bicycle` / `bicycle.py` | `flagquantum.qec.codes.bicycle` | 0.4.0 |
| 3 | `bposd` / `bposd.py` | `flagquantum.qec.decoders.bposd` | 0.4.0 |
| 4 | `chunks` / `chunks.py` | `flagquantum.qec.dem.chunks` | 0.4.0 |
| 5 | `circuit` / `circuit.py` | `flagquantum.qec.circuits.memory` | 0.4.0 |
| 6 | `context` / `context.py` | `flagquantum.qec.execution.context` | 0.4.0 |
| 7 | `css_code` / `css_code.py` | `flagquantum.qec.codes.css` | 0.4.0 |
| 8 | `decoding_graph` / `decoding_graph.py` | `flagquantum.qec.dem.graph` | 0.4.0 |
| 9 | `dem_alternatives` / `dem_alternatives.py` | `flagquantum.qec.dem.alternatives` | 0.4.0 |
| 10 | `dem_circuit` / `dem_circuit.py` | `flagquantum.qec.dem.circuit` | 0.4.0 |
| 11 | `dem_construction` / `dem_construction.py` | `flagquantum.qec.dem.construction` | 0.4.0 |
| 12 | `dem_matrices` / `dem_matrices.py` | `flagquantum.qec.codes.matrices` for `CssCodeMatrices` and `css_code_matrices`; `flagquantum.qec.dem.matrices` for DEM construction internals | 0.4.0 |
| 13 | `dem_text` / `dem_text.py` | `flagquantum.qec.dem.text` | 0.4.0 |
| 14 | `floquet` / `floquet.py` | `flagquantum.qec.codes.floquet` | 0.4.0 |
| 15 | `matching` / `matching.py` | `flagquantum.qec.decoders.matching` | 0.4.0 |
| 16 | `qldpc` / `qldpc.py` | `flagquantum.qec.codes.qldpc` | 0.4.0 |
| 17 | `reed_muller` / `reed_muller.py` | `flagquantum.qec.codes.reed_muller` | 0.4.0 |
| 18 | `registry` / `registry.py` | `flagquantum.qec.decoders.registry` | 0.4.0 |
| 19 | `repetition` / `repetition.py` | `flagquantum.qec.execution.repetition` for experiments; `flagquantum.qec.decoders.repetition` for repetition decoders/protocols; `flagquantum.qec.types` for shared records | 0.4.0 |
| 20 | `sampling` / `sampling.py` | `flagquantum.qec.execution.sampling` | 0.4.0 |
| 21 | `sliding_window` / `sliding_window.py` | `flagquantum.qec.decoders.sliding_window` | 0.4.0 |
| 22 | `subsystem` / `subsystem.py` | `flagquantum.qec.codes.subsystem` | 0.4.0 |
| 23 | `surface` / `surface.py` | **Planned:** `flagquantum.qec.codes.surface.rotated_frame` for the independent frame class; existing `flagquantum.qec.codes.surface.zxxz` for `ZxxzSurfaceCode` | 0.4.0 |

Replacements describe ownership of QEC symbols, not a promise to reproduce every
incidental imported global from the historical modules. Private helpers remain
private. Third-party, compiler and runtime objects once incidentally reachable
through these files must be imported from their own supported API instead.
No wildcard namespace mirror is introduced.

## Preserved contract

- Preserve all **108 ordered names** in `flagquantum.qec.__all__`, their public
  signatures, defaults, record fields, exceptions and numerical behavior. Redirect
  root imports to the existing authoritative implementations before deleting
  aliases. No new root export is added for the independent surface frame.
- Preserve one object per existing public class/function across root and expert
  imports within a process. Do not create wrappers, duplicate definitions,
  subclasses or replacement record types. Across releases, changed module
  metadata is permitted only for the approved retired paths.
- Preserve seeded results, tensor shapes/dtypes, matrix conventions, detector
  ordering, probability merging, decoding corrections, streaming state and
  experiment outputs. Import retirement is not an algorithm change.
- Preserve the code registry and the sole DEM decoder registry, their registration
  keys, duplicate rejection, decorators, default selection and plugin discovery
  behavior. Repetition protocols and DEM decoder protocols remain distinct.
- Preserve the two independent `RotatedSurfaceCode` classes. The registered
  `qec.RotatedSurfaceCode` / `qec.codes.RotatedSurfaceCode` stays defined in
  `codes/surface/rotated.py`. The independent class currently implemented in
  `codes/surface/_rotated_frame.py` moves to the planned public
  `codes/surface/rotated_frame.py`, retaining its own definition and behavior.
  Both classes keep their names; they must never become aliases of each other.
- Keep all frozen restructuring fixtures and historical audit records unchanged.
  They remain evidence of the pre-retirement contract, not snapshots to regenerate.

`codes`, `circuits`, `dem`, `decoders`, `execution` and `adapters` remain real
packages. `pauli.py`, `gf2.py`, `logical.py`, `types.py`, `noise.py` and root
`__init__.py` remain. This proposal does not require an empty package root or
relocate those five active shared implementations.

## Import, serialization and plugin impact

In 0.4.0, importing any of the 23 retired module names must fail with
`ModuleNotFoundError`. No replacement files, `sys.modules` aliases, import hooks,
or dynamic attribute bridges may silently restore those names. Root and expert
imports must work in fresh processes and in either import order.

A pickle that refers to a removed module has **no backward-read guarantee** in
0.4.0. This includes nested object graphs, registered global functions and
checkpoint payloads containing such objects. Retain record schemas and numeric
state; removing module lookup compatibility does not authorize changing them.
Artifacts referencing surviving modules are not categorically unsupported and
must retain their existing guarantees. Native optional-library objects that were
already not pickleable do not gain a new serialization promise.

During implementation, remove historical `__module__` pins only where they name
one of these 23 retired modules, using the actual canonical definition module
for new pickles. Inspect class methods, functions, the registry decorator and
`_DecoderT` as well as classes. Keep metadata pins for surviving public modules
where still required; do not delete `codes/_compat.py` indiscriminately. New
pickles must round-trip in a fresh 0.4.0 process through canonical modules, and
surviving-path historical fixtures must continue to load. The independent frame
class must use its planned public module rather than the private implementation
path for new artifacts.

Users who need to recover their own old artifacts can retain a compatible 0.3.x
environment, load there, and export explicit application data for reconstruction
with supported 0.4.0 APIs. A generic pickle converter, compatibility unpickler,
new schema or checkpoint framework is outside scope. Release notes must clearly
warn that simply changing a source import does not rewrite stored pickle paths.

Plugins using root registration APIs keep the same registry objects and behavior.
Plugins importing any retired submodule or monkeypatching its globals must adopt
the owner paths above. Documentation, type references, examples and maintained
plugins must use surviving imports before deletion. Historical audit documents
and immutable fixtures retain their old paths as evidence, with explanatory
context in current documentation instead of rewriting history.

## Delivery sequence and verification

1. **This proposal PR:** review the exact deletion list, replacement map, 0.4.0
   release boundary and narrowly accepted break. Run the existing example below
   and the complete PR CI. No library, fixture or snapshot changes.
2. **Contract PR:** record the approved import/metadata exception in the existing
   protected contract and focused conformance tests before Core implementation.
   Preserve frozen fixture bytes and the 108-name root API baseline. Classify
   each affected assertion explicitly; do not regenerate the baseline, broadly
   skip old tests, add blanket expected failures or lower thresholds.
3. **Implementation PR:** expose the independent frame's public module, redirect
   maintained imports, remove precisely the 23 facades and their metadata pins,
   and update current migration documentation/examples together. Use the agreed
   conformance baseline. Do not change code/decoder algorithms or registries.
4. **Release acceptance:** validate wheel and sdist membership, clean installation
   on Python 3.10, 3.11 and 3.12, and the existing no-SDK/Stim/PyMatching/both
   profiles. Confirm retired files are absent from artifacts, root exports are
   identical, fresh-process old imports fail, new imports/pickles work and all
   applicable numerical, streaming, registration and plugin checks pass. Publish
   the breaking migration note with 0.4.0 through the normal release process;
   no release or merge is authorized by a successful CI run alone.

Implementation acceptance additionally requires:

- the immutable 001-006 fixture hashes stay identical; numeric and state payloads
  remain directly compared with those fixtures even when old module lookup is
  intentionally unsupported;
- old pickle failures are confined to approved removed-path references, with
  positive coverage for surviving paths and new canonical round-trips;
- root/expert object identity, both distinct surface classes, custom registration,
  duplicate rejection and optional dependency failure behavior are demonstrated
  in source and installed distributions;
- repository imports and current documentation have no executable dependency on
  the 23 names, including lazy references from `noise.py` and metadata helpers;
- complete required CI, including both GPU checks, is green at the reviewed head.
  Existing environment-conditional jobs follow their existing workflow rules;
  no threshold, test selection or workflow is weakened for this retirement.

## Runnable current-version baseline

Run this example from the repository root with the existing core installation;
no Stim or PyMatching dependency is needed:

```bash
PYTHONPATH=. python examples/qec/installed_imports.py
```

Expected output on the current 0.3.1 implementation, including this proposal PR:

```text
Public exports: 108
Independent rotated classes: True
Code pickle path: flagquantum.qec.codes
Frame pickle path: flagquantum.qec.surface
Correction: (0,)
Logical error rate: 0.0
```

The [complete example](../../examples/qec/installed_imports.py) constructs both
surface classes, round-trips them, decodes a detector event and runs a seeded
noiseless memory experiment. The frame import and its printed pickle module are
explicitly the **pre-retirement baseline**, not a claim that the retired path
will work in 0.4.0. In the implementation PR the example must import the frame
from `flagquantum.qec.codes.surface.rotated_frame` and expect that canonical
pickle module. The other five output lines remain unchanged. That planned module
is not importable from this docs-only proposal.

The general root workflow is valid now and must produce exactly the same output
after retirement:

```python
import flagquantum.qec as qec

code = qec.get_code("rotated_surface", distance=3)
assert type(code) is qec.RotatedSurfaceCode
model = qec.DetectorErrorModel.from_stim_text(
    "error(0.125) D0 L0\ndetector D0\nlogical_observable L0\n"
)
print("Public exports:", len(qec.__all__))
print("Correction:", qec.get_decoder(qec.AUTHORITY_NAME, model).decode((0,)).observables)
point = qec.run_repetition_memory_noise_sweep((0.0,), rounds=3, shots=8, seed=53)[0]
print("Logical error rate:", point.logical_error_rate)
```

Expected output:

```text
Public exports: 108
Correction: (0,)
Logical error rate: 0.0
```
