# QEC package layout and 0.4.0 migration

Use `flagquantum.qec` for codes, memory experiments, detector models and
decoders. Its 108 ordered exports, signatures/defaults, numerical behavior and
code/decoder registration policy remain unchanged. The approved **0.4.0**
retirement removes 23 historical flat modules and their pickle lookup paths.
[Proposal #757](../development/API_CHANGE_PROPOSAL_085_QEC_HISTORICAL_IMPORT_RETIREMENT.md)
and the [amended contract](../development/QEC_RETIREMENT_CONTRACT.md) record the
explicit approval, including the decision to keep one `RotatedSurfaceCode`.
This source version targets 0.4.0; it does not publish a release.

## Find the implementation

```text
flagquantum/qec/
  codes/                 code records, matrices, families and the code registry
    surface/             one registered rotated class and the ZXXZ family
  circuits/              memory source and measurement/detector declarations
  dem/                   model, text, mechanisms, IR realization, chunks and graph
  decoders/              repetition, matching, BP, BPOSD, windows and one registry
  execution/             lowering, sampling, feedback, signatures and composition
  adapters/              optional PyMatching import/translation
  pauli.py               shared Pauli representation
  gf2.py                 shared binary algebra
  logical.py             shared logical-product records
  noise.py               shared noise records and retained lazy sweep exports
  types.py               shared syndrome/event/result records
  __init__.py            unchanged ordered public API
```

The root now has six Python files, and the complete QEC tree has 50 Python
files. Shared leaves are active implementations. Core owns QEC; each
responsibility directory has a README with allowed dependencies, exclusions,
entry points and a short edit/test path.

## Replace historical imports

Root imports such as `from flagquantum.qec import CssCode` continue to work.
For module imports, replace the prefix in this table. The machine-readable
[contract](../../contracts/qec-historical-import-retirement-v0.4.json) is the
exact removal list.

| Removed `flagquantum.qec` module | Canonical replacement under `flagquantum.qec` |
| --- | --- |
| `belief_propagation` | `decoders.belief_propagation` |
| `bicycle` | `codes.bicycle` |
| `bposd` | `decoders.bposd` |
| `chunks` | `dem.chunks` |
| `circuit` | `circuits.memory` |
| `context` | `execution.context` |
| `css_code` | `codes.css` |
| `decoding_graph` | `dem.graph` |
| `dem_alternatives` | `dem.alternatives` |
| `dem_circuit` | `dem.circuit` |
| `dem_construction` | `dem.construction`; execution helpers in `execution.signatures` |
| `dem_matrices` | `codes.matrices` for `CssCodeMatrices` / `css_code_matrices`; `dem.matrices` for DEM mechanisms |
| `dem_text` | `dem.text` |
| `floquet` | `codes.floquet` |
| `matching` | `decoders.matching` |
| `qldpc` | `codes.qldpc` |
| `reed_muller` | `codes.reed_muller` |
| `registry` | `decoders.registry` |
| `repetition` | `execution.repetition`; decoders in `decoders.repetition`; records in `types` |
| `sampling` | `execution.sampling` |
| `sliding_window` | `decoders.sliding_window` |
| `subsystem` | `codes.subsystem` |
| `surface` | `codes.surface.rotated` for the registered `RotatedSurfaceCode`; `codes.surface.zxxz` for `ZxxzSurfaceCode` |

There is one `RotatedSurfaceCode`, the original registered class defined in
`codes/surface/rotated.py`. Root, `qec.codes`, `codes.surface` and
`get_code("rotated_surface", ...)` return that class. The duplicate
`_rotated_frame.py` implementation is removed; no public `rotated_frame` module
is added. ZXXZ uses the existing rotated lattice helpers. Checks, logicals,
measurement order and numerical results remain protected by the original tests.

The code registry remains in `codes/__init__.py`; the decoder registry remains
in `decoders/registry.py`. Plugins must update retired module imports, then use
the same public registration functions. Duplicate rejection, explicit plugin
replacement, lazy default registration and existing keys remain unchanged.

## Serialized objects and inspection

Pickles referring to the 23 removed modules now fail lookup. This includes the
old independent surface class, old ZXXZ surface path, matrix records and moved
execution/decoder globals. There is no alias module, custom reducer, import hook
or compatibility unpickler. Before upgrading, load affected trusted artifacts
with the pre-retirement installation and export the domain data needed to
reconstruct them through supported APIs. New canonical checkpoints round-trip,
including tensor records, caller-owned PyTorch safe-global lists and in-progress
streaming state. Arbitrary old pickle conversion is outside this change.

Surviving `qec.codes`, `qec.dem`, `qec.decoders`, `qec.adapters`, `qec.noise` and
shared-record identifiers retain metadata and compatibility. In particular,
the registered rotated class still pickles as `flagquantum.qec.codes`.
`codes/_compat.py` remains only for those surviving identities. Source inspection
and private instrumentation should target the physical owner; a retained
`__module__` need not name the defining source file.

The eager root bootstrap is unchanged. Code/circuit/model boundaries are tested
after bootstrap; this change makes no cold-root-import infrastructure claim.
PyMatching remains optional and imports only when selected. Its existing
missing-extra error and native graph pickle limitation remain unchanged.

## Run a user workflow

From an installed distribution, or with `PYTHONPATH=.` in this checkout:

```bash
python examples/qec/installed_imports.py
```

Expected output:

```text
Public exports: 108
Registered class: True
Code pickle path: flagquantum.qec.codes
Correction: (0,)
Logical error rate: 0.0
```

The [code-family example](../../examples/qec/code_family_imports.py) checks root,
package and owner class identity, equal layouts, pickle identity and Steane
counts `7 6 2`. The [DEM example](../../examples/qec/dem_model_and_chunks.py)
reports rates `[0.3125, 0.25]` and a closed chunk. The
[decoder example](../../examples/qec/decoder_registry_and_pickle.py) compares
three algorithms and round-trips a canonical matching decoder. These require
no optional SDK. The [memory sampler](../../examples/qec/memory_layout_and_sampling.py)
requires `stim`; selecting `get_decoder("pymatching", model)` requires
`pymatching`.

## Verify source and installed distributions

```bash
PYTHONPATH=. python tools/check_qec_retirement.py --phase retired
PYTHONPATH=. python -m pytest tests/api_contract/test_qec_retirement_contract.py tests/qec -q
artifact_dir="$(mktemp -d)"
uv build --sdist --out-dir "$artifact_dir"
uv build --wheel "$artifact_dir"/flagquantum-*.tar.gz --out-dir "$artifact_dir"
check_dir="$(mktemp -d)"
uv venv --python 3.12 "$check_dir/venv"
uv pip install --python "$check_dir/venv/bin/python" "$artifact_dir"/flagquantum-*.whl
uv pip check --python "$check_dir/venv/bin/python"
```

The gate must print `QEC retirement contract passed: retired, 23 modules, 108 exports`.
The former `prepare` phase now exits 1 with exact missing-module diagnostics.
All original 15 restructuring fixture files remain byte-identical. Historical
compound pickles containing retired names must fail; surviving objects extracted
using the pre-retirement installation have separate provenance in
[retirement_fixtures](../../tests/qec/retirement_fixtures/README.md). Original
JSON oracles still check numerical, source/layout, sampling and feedback results.

Run installed examples by absolute path outside the checkout with `PYTHONPATH`
unset. Verify QEC imports from the environment's `site-packages`. Check all
supported Python minors, 3.10–3.12, with core/Stim/PyMatching/both profiles,
including native extension availability, dependency consistency, exact
source/wheel/sdist QEC membership and retired import refusals. Required complete
CI and GPU checks must pass at the reviewed PR head; local CPU evidence makes
no GPU scalability claim. No change to or merge of #731 is included.

The original structural delivery records
[001](../development/QEC_RESTRUCTURE_001.md)–[007](../development/QEC_RESTRUCTURE_007.md)
and [mainline acceptance](../development/QEC_RESTRUCTURE_ACCEPTANCE.md) describe
the pre-retirement compatibility boundary and retain their historical evidence.
