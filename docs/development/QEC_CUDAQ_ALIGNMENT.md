# FlagQuantum QEC ↔ CUDA-Q alignment checklist — the reading

Read this with [`qec-cudaq-alignment-checklist.toml`](../../contracts/qec-cudaq-alignment-checklist.toml)
and [`check_qec_cudaq_alignment.py`](../../tools/check_qec_cudaq_alignment.py)
open. The TOML is the checklist; the Python tool is the check that it still
describes a real repository; this document is the part a person reads once.

```
python tools/check_qec_cudaq_alignment.py        # 299 checks, exit 0 or 1
python -m pytest tests/unit/test_qec_cudaq_alignment_check.py -q
```

The first command is run by CI, so a checklist row that stops describing this
repository fails the build rather than aging quietly. The second damages one
section of the checklist at a time and requires the checker to fail for the
stated reason; it is the evidence that the first command is a check and not a
formality.

## 1. What is being aligned to what

The question this answers is "which framework's functionality does
`flagquantum/qec` have to line up with". The answer has one target and two
supporting standards, and they are not interchangeable:

| Layer | Aligned to | Why not the others |
| --- | --- | --- |
| Product surface — objects, entry points, code families, decoders | **CUDA-Q QEC** (`cudaq-qec`, now `NVIDIA/cudaqx` `libs/qec`) | It is the only live framework with a QEC library of this shape, and `contracts/cudaq-parity-matrix.toml` already names it as the baseline. |
| Detector error model bytes | **Stim** DEM text, stim 1.16.0 | The format is stim's. Aligning it to CUDA-Q would mean inventing a second format nobody else reads. |
| Decoder correctness | **PyMatching** as cross-check, never as authority | Correctness evidence has to be independent of the implementation under test; the self-implemented matcher is the authority. |
| Logical operations (`logical.*`) | **Qualtran**, as an adaptation | CUDA-Q's `logical.*` is a preview namespace of patch/gadget/protocol records. The semantic core is the target; the preview records are not. |

Three things this checklist explicitly does **not** align to:

1. **`qiskit-qec`.** Archived, read-only, early-stage, breaking API, and its PyPI
   project returns 404. `API_CHANGE_PROPOSAL_048` L47–51 already recorded it as
   not a viable parity target.
2. **`nv_qldpc_decoder`.** It ships as `libcudaq-qec-nv-qldpc-decoder.so` under
   the NVIDIA Software License — closed source. Align to its *interface
   position* (a named decoder behind a registry) and implement with Apache-2.0
   PyMatching/LDPC-BP. Note that CUDA-Q QEC's own chromobius and pymatching
   plugins are in-tree open source, which is direct precedent for integrating
   rather than reimplementing.
3. **CUDA-Q's GPU decoder performance.** The matrix's own reason for
   `qec_decoder_family` says integrate first, accelerate later through the
   FlagOS kernel path.

## 2. The 15 alignment rows

`floor` is the priority this repository should hold the row at; where it differs
from the matrix's `priority`, the row states why.

| Row | Status | Floor | Matrix row | The gap in one line |
| --- | --- | --- | --- | --- |
| `qec_code_record` | partial | now | `qec_code_library` | Two families declared, no matrix-level record, no Steane/colour/qLDPC. |
| `qec_detector_annotations` | partial | now | — | Layouts beside the source, not annotations in the kernel; no measurement handles. |
| `qec_syndrome_extraction_owner` | partial | now | — | `extract_syndrome` is in the CUDA-Q Logical preview, not CUDA-Q QEC. Matrix fix landed. |
| `qec_dem_construction` | partial | now | — | Construction exists and is exact; no CSS-matrix entry point, no context object. |
| `qec_dem_matrices_and_rates` | partial | now | — | Orientation matches; no error ids, no rates vector. |
| `qec_dem_merge` | absent | now | — | Merging is implicit and parity-only; upstream has two stated modes. |
| `qec_dem_chunking` | absent | later | — | No chunks, no seams, therefore no sliding-window substrate. |
| `qec_dem_text_interchange` | partial | now | `qec_stim_integration` | Both directions present and independently checked; input end is narrow. |
| `qec_stim_sampling_join` | partial | now | `qec_stim_integration` | The stabilizer engine refuses noise channels, so circuit→DEM→decoder has no sampling join. |
| `qec_decoder_family` | partial | now | `qec_decoder_family` | A DEM-consuming matching decoder landed; no registry, no BP+OSD, no sliding window. |
| `qec_decoder_configuration` | absent | later | — | Nothing to configure until more than one decoder can be selected. |
| `qec_dialect` | absent | later | `qec_dialect` | Needs an internal IR level to carry the structure. |
| `qec_logical_operations` | absent | later | `qec_logical_operations` | Depends on the decoder family; align to the semantic core via Qualtran. |
| `qec_transport_and_objectives` | absent | later | `qec_transport_and_objectives` | Hardware-shaped; out of scope until a neutral-atom target exists. |
| `qec_stim_user_migration` | absent | next | `qec_stim_user_migration` | A document, and its upstream counterpart is CUDA-Q QEC's own Stim surface rather than a page to translate. |

Everything a `supported` row would need is deliberately *not* claimed here. No
row is `aligned`, because no row has the evidence an `aligned` claim would
require — and the checker enforces that: an `aligned` row must name a present
symbol and may not sit on a matrix row that says `unsupported`.

**What `qec_decoder_family` closed, and what it did not.** The row's order was
"decoding graph with `log((1-p)/p)` edge weights and observable labels, a
self-implemented minimum-weight matcher, then the optional PyMatching adapter as
the bit-for-bit cross-check." The first two are now in the repository:
`flagquantum/qec/decoding_graph.py` turns a detector error model into a graph
whose boundary node is one past the last detector, and
`flagquantum/qec/matching.py` decodes a syndrome exactly by enumerating the ways
to pair its defective detectors and to route any number of them to that
boundary. Neither imports `stim`, `pymatching`, `networkx` or `scipy`; the graph
carries the observable labels a prediction needs, which is the part the
repetition-profile decoders never had to represent.

Two refusals keep the row honest rather than merely larger. A mechanism flipping
three or more detectors is a hyperedge and raises `CapabilityError` naming the
detectors, because projecting it onto a pair would drop a detector and the
matcher would then return a correction that does not explain the syndrome. The
pairing is enumerated, so the decoder takes a defect budget (twenty by default)
and refuses a larger syndrome rather than returning a pairing that only looks
cheapest. Both are the fail-closed convention, not stubs.

The third item in that order is still open, and it is the one the row's
correctness argument rests on: PyMatching as an independent cross-check behind an
optional extra. Until it exists, the matcher's correctness evidence is brute
force over enumerable syndromes plus the exhaustive enumeration of a model's own
mechanism distribution — both independent of the implementation, neither
independent of the detector error model it reads. The registry stays absent on
purpose: `get_decoder(name, ...)` has no second caller yet, and a name-keyed
factory is a separate decision from a decoder.

## 3. The 23 field rows — `dem.py` against `DEMResult` / `dem_from_kernel` (both CUDA-Q core)

The short version, because the full table is in the TOML. Across 23 field rows:
2 `equivalent`, 1 `renamed`, 1 `extra`, 10 `reshaped`, 9 `absent`.

**Equivalent (2).** `detector_error_matrix` and `observables_flips_matrix` are
the same matrices in the same orientation — rows are detectors or observables,
columns are error mechanisms — with a container difference only (`numpy uint8`
upstream, `torch int8` here).

**Renamed (1).** `dem_from_stim_text(text, use_decomp_suggestions=False)` is
`DetectorErrorModel.from_stim_text`. Both delegate the format definition to
stim's own parser. The refusals differ in kind, not in spirit: upstream
documents two losses on the stim path (error ids always empty, `^` separators
ignored), this repository refuses constructs it cannot represent (`repeat`,
comments, skipped indices, malformed lines).

**Extra (1).** `to_stim_text`. No writer was found in the cudaq-qec bindings or
libraries; the produced stim text in the CUDA-Q stack comes from
`cudaq.DEMResult.dem` in core and from the Logical preview CLI. This is a
negative result on the upstream side, marked `provenance_unverified`, so it is
"nothing to align to at this layer" rather than "ahead".

**Reshaped (10).** The model carrier, the per-error rates, the counts, the
sampling function, the memory-circuit entry point, the noise record, the
measurement-to-detector map, the kernel annotation surface, the decoder result
record and the decoder base protocol. The two widest of these:

- **Noise record.** Upstream `CssNoise` carries independent X, Y and Z data
  rates plus a measurement rate, each also expressible per qubit or per check.
  `PhenomenologicalNoise` here is one data-flip scalar and one measurement-flip
  scalar. Per-location rates are not expressible, and neither is an asymmetric
  X/Z channel.
- **Sampling function.** Upstream
  `dem_sampling(check_matrix, num_shots, error_probabilities, seed=None, backend="auto")`
  is a free function over a matrix plus a rate vector returning sampled check
  syndromes *and* the sampled error mechanisms. Here it is a method on the model
  returning detectors and observables. The upstream error output is absent here;
  the local observable output is extra there. Upstream also rejects PyTorch CPU
  tensors, which is a full-precision-vs-`uint8` boundary worth knowing before an
  adapter is written.

**Absent (9).** `error_ids` (mutually exclusive / correlated errors),
`canonicalize_for_rounds`, `dem_merge_duplicate_columns` with its two modes,
the chunk/seam/stitch/close family, `dem_from_css_matrices`, `dem_from_kernel`
itself, the decoder registry (`get_decoder` + `@decoder`), the sampling backend
selector, and the decoder configuration schema.

Of the absent set, `error_ids` and `canonicalize_for_rounds` are the two that
sit closest to work already planned: a matcher wants round structure and a
correlated channel is what a decomposed mechanism is, and both currently have no
way to be said.

## 4. Findings from the reading, and where the matrix now records them

These came out of reading the upstream source and are *not* criticisms of this
repository — they are places where the recorded baseline and the live upstream
have drifted apart. All five are landed in
[`contracts/cudaq-parity-matrix.toml`](../../contracts/cudaq-parity-matrix.toml);
each entry below names the construct that carries it and what the tool now
refuses.

1. **`dem_from_kernel` and `DEMResult` are CUDA-Q core, not cudaq-qec.** The
   domain's surface item "dem, dem_from_kernel, and DEMResult" mixes two owners.
   cudaq-qec's own construction entry points are `dem_from_stim_text`,
   `dem_from_memory_circuit` with `x_`/`z_` variants, and
   `dem_from_css_matrices`; `dem_from_kernel` and `DEMResult` live in CUDA-Q
   core and are consumed by the library. **Landed as**
   `[[baseline.surface_ownership.items]]` in the contract, owned by `cudaq-core`,
   whose consequence states that the item must be read as two and that the core
   record is a dependency of the QEC row rather than its baseline.

2. **`extract_syndrome`'s owner is the CUDA-Q Logical preview, not cudaq-qec.**
   `cudaq.logical.extract_syndrome` exists and that layer states of itself that
   it does not simulate, sample or decode and emits no detector error models.
   CUDA-Q QEC's own extraction route is `sample_memory_circuit` /
   `x_sample_memory_circuit` / `z_sample_memory_circuit` and
   `decoder_context_from_memory_circuit`. **Landed as** a second
   `surface_ownership` entry owned by `cudaq-logical-preview`, whose consequence
   records that aligning to that layer would not deliver a detector error model
   and that the layer is therefore a second target with a different deliverable,
   not a second baseline for this domain.

3. **No `for-stim-users` page exists for cudaq-qec.** The only such page is
   *CUDA-Q Logical for Stim users*, in the preview layer, with a companion *Stim
   emission* page. Both explicitly state that the layer does no detector
   annotations, no detector error models and no sampling. **Landed as** a
   `surface_ownership` entry for the surface item, plus a rewrite of the
   `qec_stim_user_migration` row: its `reason` now names cudaq-qec's own Stim
   surface (`dem_from_stim_text`, `dem_sampling`, the decoder framework's Stim
   DEM entry point) as the counterpart, and the row carries a `search:` evidence
   item recording that no such page and no Stim-text writer exist in cudaq-qec.
   The Stim surface item is attributed too, because both components appear in
   it and only the cudaq-qec half decodes.

4. **`qec_stim_integration` named one maturity entry for two capabilities.** The
   row pointed at `stabilizer_sampling`, which is the sampling half, while the
   interchange half is `detector_error_model`. **Landed as** `maturity_refs`, the
   plural form: the row now names both entries, `tools/parity_matrix.py`
   validates every entry, and the checker reads both forms. The alternatives
   were both worse. Splitting the row would leave the join — which is the thing
   the row is actually about — in neither half, and it would move the capability
   count. A stated override in the checklist would leave the matrix row still
   claiming to be one capability with one entry. The plural form was chosen
   because the row genuinely spans two registry entries, and the contract's
   `maturity_refs_rule` states that a row uses one form or the other and never
   both. The override reason this checklist previously carried on
   `qec_dem_text_interchange` is gone, because it recorded the drift rather than
   the fix.

5. **The version pin does not reach the upstream source.** The matrix pins CUDA-Q
   0.15.1 and 0.16.0.post1; cudaq-qec's release line runs 0.1.0 to 0.8.0 and
   `0.8.0` is what the QEC rows were read at. No cudaq-qec release maps to the
   pinned versions. **Landed as** `[baseline.version_provenance]` in the
   contract: the pinned component, the QEC release line, the version recorded,
   the release index, and a `provenance_limit` stating that the two lines are
   disjoint so a QEC row is not a statement about the pinned core versions. The
   tool fails closed in both directions — a pin that meets the component line
   requires a `version_mapping`, and a mapping across disjoint lines is refused.
   The generated scoreboard and
   [`CUDAQ_PARITY_BASELINE.md`](CUDAQ_PARITY_BASELINE.md)
   both carry it, and the baseline document lists the component release line
   moving as a refresh trigger.

The surface attributions are machine-checked: an attributed item must appear
verbatim in exactly one domain surface list, must name a declared owner, and
must state its consequence, so the table cannot drift from the inventory it
explains and an item owned by another component cannot be read as a cudaq-qec
row.

## 5. How the checker fails closed

The point of `tools/check_qec_cudaq_alignment.py` is that the checklist cannot quietly
become fiction. Resolution is a source-level AST scan, not an import, so it runs
without torch, without an installed FlagQuantum, and without the optional `stim`
distribution. Per row it checks:

- every `evidence` path exists in the repository;
- every `symbols_present` resolves to a real definition;
- every `symbols_absent` resolves to nothing — so a documented gap dies the day
  it is filled, and a claimed alignment dies the day the symbol is deleted;
- every `negative_search` path is still missing;
- `maturity_ref` exists in `capability-maturity.toml` **and is one the parity
  matrix row names** — through `maturity_ref` or through the matrix row's
  `maturity_refs` — unless the row states why it cannot;
- `domain_row` names a real capability of the `quantum_error_correction` domain,
  the checklist status does not contradict the matrix status, the floor does not
  contradict the matrix priority without a stated reason, and every
  `domain_items` string is a CUDA-Q surface item the matrix actually records;
- `partial` and `absent` rows state a `next_action`; `absent` and `out_of_scope`
  rows prove the gap; every row states a `fail_closed`; a row marked
  `provenance_unverified` says `UNVERIFIED` in its own note;
- the header's component version and release index are the ones
  `contracts/cudaq-parity-matrix.toml` records, and a contract that records no
  provenance fails the header rather than passing it, so the `provenance_limit`
  this document states is the contract's limit and not a second copy of it;
- every row id appears in this document, so a row cannot exist only in the TOML.

`tests/unit/test_qec_cudaq_alignment_check.py` also applies its mutations
structurally — by row id and key name — rather than by quoting long literals, so
editing the prose of a row cannot quietly turn a mutation into a no-op.

The AST index was cross-checked against a live import, because a source scan
that disagrees with the package it scans is worse than no scan. Importing
`flagquantum.qec` directly and exercising the rows' own claims —
`from_stim_text` on a two-line model, both parity matrices, `to_stim_text` round
tripping byte-identically, `dem_sampling(shots=5)` returning a `DemSample` of
shape `(5, 1)` on both axes — agrees with what the checklist says each symbol is.

`tests/unit/test_qec_cudaq_alignment_check.py` is what makes that credible: it
damages one row at a time and requires the checker to fail *for the stated
reason*, with three controls that catch a checker which reads the checklist
instead of the tree — a symbol planted in a throwaway copy of the repository
must be seen, a name the tree lacks must be rejected, and the planted claim must
pass again once the definition is removed.

## 6. Where these files live

Four artifacts, all four of them in the repository, each doing one job:

| Artifact | Home | Role |
| --- | --- | --- |
| Checklist | [`contracts/qec-cudaq-alignment-checklist.toml`](../../contracts/qec-cudaq-alignment-checklist.toml) | The rows, as data. A repository contract like its neighbours in `contracts/`. |
| Checker | [`tools/check_qec_cudaq_alignment.py`](../../tools/check_qec_cudaq_alignment.py) | Resolves every claim against this checkout. Run by CI's `quality` job. |
| This reading | `docs/development/QEC_CUDAQ_ALIGNMENT.md` | The prose a person reads once, including the field-row index below. |
| Mutation test | [`tests/unit/test_qec_cudaq_alignment_check.py`](../../tests/unit/test_qec_cudaq_alignment_check.py) | Proves the checker can fail, by damaging one section at a time. |

Defaults inside the checker are absolute, derived from its own location, so
`python tools/check_qec_cudaq_alignment.py` gives the same answer from any
working directory and the CI step does not have to name the repository it is
standing in. `--repo-root`, `--checklist`, and `--reading` remain overridable for
the mutation test, which runs the checker against a throwaway copy of the
repository rather than the real one.

The section 4 findings are contract changes rather than separate artifacts. They
landed in `contracts/cudaq-parity-matrix.toml`, with the regenerated
`docs/reference/CUDAQ_PARITY_MATRIX.md`, `contracts/README.md`,
`docs/development/CUDAQ_PARITY_BASELINE.md`, `tools/parity_matrix.py`, and
`tests/unit/test_cudaq_parity_matrix_contract.py` updated in the same change.

The split between the two documents is deliberate and is the thing to keep
straight when editing either. `cudaq-parity-matrix.toml` answers *which
capability does CUDA-Q have, at what priority, with what evidence*, across the
whole product and all twelve domains. This checklist answers *which upstream
surface does one QEC row line up with, symbol by symbol, and what exactly is
missing*, for one domain. The matrix owns the capability inventory and the
maturity join; this checklist owns the symbol-level diff and the upstream
attribution. Nothing in it should be promoted back into the matrix, and a row
here that disagrees with its matrix row fails the checker.

### Field-row index

Every row in the checklist, keyed by the id the TOML uses. `upstream` is the
CUDA-Q side; the last column is the difference in one line.

| Field row | Verdict | Difference |
| --- | --- | --- |
| `dem_carrier` | reshaped | Upstream is an opaque object with operations and recorded DEM-text provenance; here it is a frozen, validated, immutable record. |
| `dem_detector_matrix` | equivalent | Same orientation, same entries; `numpy uint8` against `torch int8`. |
| `dem_observable_matrix` | equivalent | Same, and upstream also documents the consumer idiom (`O @ errors % 2`). |
| `dem_error_rates` | reshaped | Per-column vector upstream, per-error record here; no vector accessor. |
| `dem_error_ids` | absent | No way to say two mechanisms are alternatives rather than independent. |
| `dem_counts` | reshaped | Methods upstream against properties here; `num_error_mechanisms` against `num_errors`. |
| `dem_sampling_function` | reshaped | Free function over matrix + rates returning syndromes *and* mechanisms, against a model method returning detectors + observables. |
| `dem_sampling_backend` | absent | No execution-target selector; upstream has `auto`/`cpu`/`gpu`. |
| `dem_from_stim_text` | renamed | Same operation, same parser authority (stim), free function against classmethod. |
| `dem_to_stim_text` | extra | No writer found upstream at this layer; the produced text comes from CUDA-Q core. |
| `dem_from_css_matrices` | absent | No matrix-level entry point, so an arbitrary parity-check matrix has no route in. |
| `dem_from_memory_circuit` | reshaped | Upstream takes code + operation + rounds + noise model and is split by basis; here the circuit carries rounds and basis. No context object. |
| `dem_code_capacity_noise` | reshaped | Two uniform scalars against X/Y/Z data rates plus a measurement rate, each also per qubit or per check. |
| `dem_canonicalize` | absent | No round-structure operation; the matcher decodes across rounds without one, but a sliding window would need it. |
| `dem_merge_operation` | absent | Merging is implicit, construction-only, parity-rule-only; upstream has two stated modes. |
| `dem_seam_and_chunk_api` | absent | Monolithic model; no chunk, no seam, nothing to slide a window over. |
| `dem_measurement_to_detector_map` | reshaped | Ordered `MeasurementRef` records on the layout against a stored sparse D matrix on the model. |
| `kernel_annotation_surface` | reshaped | Layouts beside the source against annotations in the kernel body over measurement handles. |
| `kernel_dem_from_kernel` | absent | CUDA-Q core derives the DEM from the kernel's own annotations; here it is assembled by hand. |
| `decoder_registry` | absent | Decoders are constructed directly; no name lookup, no DEM-text entry point. |
| `decoder_result_record` | reshaped | Single-shot record against `converged` + optional results, with separate batch and async records. The DEM-level matcher adds `MatchingDecodeResult`, which carries observables, the selected mechanisms and their weight, and deliberately is not this record. |
| `decoder_base_methods` | reshaped | `decode` + streaming against `decode`/`decode_batch`/`decode_async`/`get_block_size`/`get_syndrome_size`/`get_version` and an errors-vs-observables request. The DEM-level decoder does not implement the repetition-only protocol, because no correction record here can express a surface-code correction. |
| `decoder_plugin_precedent` | absent | Upstream integrates open chromobius and pymatching plugins behind a boundary while its closed decoder ships as a binary — the same split the matrix plans. |
