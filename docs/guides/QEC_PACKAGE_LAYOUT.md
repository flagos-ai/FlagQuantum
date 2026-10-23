# QEC package layout and compatibility

Use the existing `flagquantum.qec` imports for codes, memory experiments, detector
models and decoders. The package keeps its 108 ordered public exports, original
signatures/defaults, numerical behavior and code/decoder registration policy.
The responsibility split changes where contributors edit implementations.

The API owner has authorized direct retirement of 23 historical submodules.
[Proposal 085](../development/API_CHANGE_PROPOSAL_085_QEC_HISTORICAL_IMPORT_RETIREMENT.md)
lists every replacement and targets **0.4.0**. It is a separate breaking change;
the current 0.3.1 implementation still retains all historical imports and pickle
paths described below. The proposal does not remove any implementation.

## Find the implementation

```text
flagquantum/qec/
  codes/                 code records, matrices, families and the code registry
    surface/             registered rotated, independent frame and ZXXZ records
  circuits/              memory source and measurement/detector declarations
  dem/                   model, text, mechanisms, IR realization, chunks and graph
  decoders/              repetition, matching, BP, BPOSD, windows and one registry
  execution/             lowering, sampling, feedback, signatures and composition
  adapters/              optional PyMatching import/translation
```

`pauli.py`, `gf2.py`, `logical.py`, `types.py` and the record portion of `noise.py`
remain shared leaves. They are active responsibilities, not unfinished moves.
Core owns QEC. Each responsibility directory has a README describing allowed
dependencies, exclusions, public entry points and a ten-minute edit/test path.

| Existing import or responsibility | Physical owner |
| --- | --- |
| `qec.codes` protocols/checks | `codes/checks.py` |
| Root / `qec.dem_matrices.CssCodeMatrices` matrix record | `codes/matrices.py` |
| Registered `qec.RotatedSurfaceCode` / `qec.codes.RotatedSurfaceCode` | `codes/surface/rotated.py` |
| Independent `qec.surface.RotatedSurfaceCode` | `codes/surface/_rotated_frame.py` |
| Other code families | Named modules under `codes/`; ZXXZ under `codes/surface/` |
| `qec.circuit` memory declarations | `circuits/memory.py` |
| `qec.dem` model, rates, sampling and merge rules | `dem/model.py` |
| `dem_text`, `dem_alternatives`, `dem_matrices` | `dem/text.py`, `dem/alternatives.py`, `dem/matrices.py` |
| `dem_construction` analytic mechanisms | `dem/construction.py` |
| `dem_circuit`, `chunks`, `decoding_graph` | `dem/circuit.py`, `dem/chunks.py`, `dem/graph.py` |
| Former `qec.decoders` repetition protocols/algorithms | `decoders/repetition.py` |
| `matching`, `belief_propagation`, `bposd`, `sliding_window` | Corresponding modules under `decoders/` |
| `qec.registry` DEM decoder registration | `decoders/registry.py` |
| `qec.repetition`, `qec.sampling`, `qec.context` | `execution/repetition.py`, `execution/sampling.py`, `execution/context.py` |
| Noise sweep and forced-signature execution helpers | `execution/noise_sweep.py`, `execution/signatures.py` |
| `qec.adapters` PyMatching boundary | `adapters/pymatching.py` |

The code registry remains authoritative in `codes/__init__.py`; the decoder table
is authoritative in `decoders/registry.py`. Existing root and historical imports
refer to those objects. Do not create a second registry or merge the repetition
and DEM decoder protocols into a new public concept.

## Old imports, identity and serialized objects

The two `RotatedSurfaceCode` classes remain independent. Root/code-registry imports
select the registered family; `flagquantum.qec.surface` retains the independent
frame record. Their check layouts can agree while their class identities differ.

Former flat modules and the `codes`, `dem`, `decoders` and `adapters` initializers
retain explicit aliases. These aliases are required by public imports, old pickle
identifiers, caller-owned safe-global lists and existing plugins. They contain no
replacement algorithm/class and no new reducer. The metadata helper retains
original class/function/method module and qualified names. The registry TypeVar
and returned registration decorator also retain their old modules.

The existing eager root bootstrap remains eager. Pure code/circuit/model paths
have explicit dependency boundaries after that bootstrap; this split does not
promise a cold root import without execution infrastructure. Optional PyMatching
still imports only when selected. Its existing missing-extra error and native
graph pickle limitation remain unchanged.

Source filenames change. Converting a module file to a package also changes its
normal filesystem/package metadata (`__file__`, `__path__`, `__package__`). Public
import identifiers and retained class/method pickle metadata stay stable.
Source-inspection tools should use a handwritten
method's code filename or the physical owner in the table, rather than assuming
retained `__module__` names a defining source file. Private global replacement
must target the implementation owner; public registration functions and ordinary
mutations of the shared table retain their established behavior.

Core owns all compatibility paths and `codes/_compat.py`. They are explicitly
frozen for this refactor. No removal release is approved. Removal requires a
separately approved API/deprecation proposal, plugin analysis and serialized
artifact migration plan before a target release can be assigned. Existing
references justify retaining these files; no speculative cleanup removes them.

## Run a user workflow

The core-only [installed import example](../../examples/qec/installed_imports.py)
checks public exports, both surface pickle paths, decoding and the noiseless
memory sweep. Run it in this checkout with:

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

The [DEM example](../../examples/qec/dem_model_and_chunks.py) needs no SDK and
reports exact rates `[0.3125, 0.25]` and a successfully closed chunk. The
[decoder example](../../examples/qec/decoder_registry_and_pickle.py) needs no SDK
and compares three algorithms while retaining matcher pickle identity. The
[memory sampler](../../examples/qec/memory_layout_and_sampling.py) requires the
declared `stim` extra. Selecting `get_decoder("pymatching", model)` requires the
declared `pymatching` extra.

## Verify an installed distribution

Build in a fresh artifact directory with the declared isolated build dependencies:

```bash
artifact_dir="$(mktemp -d)"
uv build --sdist --out-dir "$artifact_dir"
uv build --wheel "$artifact_dir"/flagquantum-*.tar.gz --out-dir "$artifact_dir"
check_dir="$(mktemp -d)"
uv venv --python 3.12 "$check_dir/venv"
uv pip install --python "$check_dir/venv/bin/python" "$artifact_dir"/flagquantum-*.whl
uv pip check --python "$check_dir/venv/bin/python"
```

Run the example by its absolute path while the working directory is outside the
checkout, with `PYTHONPATH` unset. Verify `flagquantum.qec.__file__` points inside
the new environment's `site-packages`; otherwise the checkout can conceal a
missing wheel module. Optional profiles install neither SDK, Stim only,
PyMatching only or both. Test each supported Python minor, 3.10 through 3.12.

The closure checks compare all QEC Python members of the sdist and wheel with
the committed source bytes, load trusted old pickle/checkpoint fixtures, resume
streaming state and run public examples. They also check dependency consistency
and native extension availability. The original Stim-only sampling case runs
when Stim is installed; missing-SDK refusals are checked in absence profiles.

The migration's implementation batches and evidence are recorded in
[001](../development/QEC_RESTRUCTURE_001.md),
[002](../development/QEC_RESTRUCTURE_002.md),
[003](../development/QEC_RESTRUCTURE_003.md),
[004](../development/QEC_RESTRUCTURE_004.md),
[005](../development/QEC_RESTRUCTURE_005.md),
[006](../development/QEC_RESTRUCTURE_006.md) and
[007](../development/QEC_RESTRUCTURE_007.md). Exact-head CI results belong to each
PR. Local CPU installation checks make no new GPU scalability claim.
