# Detector error models

Core owns the exact model records, their text grammar, physical mechanism
construction, chunk arithmetic and graph representation. This package does not
own decoder algorithms, code families, execution kernels or SDK policy.

| Module | Responsibility and allowed dependencies |
| --- | --- |
| `model.py` | Records, validation, rates and sampling; code matrices, declarative circuits, noise records, text and alternatives. The memory factory imports construction only when called. |
| `text.py` | Existing Stim text grammar; standard library only. |
| `alternatives.py` | Exclusive mechanism groups; standard library and PyTorch. |
| `matrices.py` | Code-capacity mechanisms; code matrices and noise records. |
| `circuit_noise.py` | Gate-bound Pauli locations and exact exclusive groups; shared execution/signature boundary and existing model records. |
| `construction.py` | Circuit/noise mechanisms; declarative layouts and the existing execution/signature reference helpers. |
| `circuit.py` | Model/IR realization; owned IR and noise channels. |
| `chunks.py` | Detector layers, seams, stitching and closing; models and circuit declarations. |
| `graph.py` | Graphlike mechanisms and weights; models. |

The model/text/matrix path must not eagerly import compiler, runtime or execution.
The legacy QEC root remains eager; package conversion does not change its import
behavior. Public users can keep importing `DetectorErrorModel`, `DemError`,
`DecodingGraph` and the chunk operations from `flagquantum.qec`.

`dem/__init__.py` retains the surviving `qec.dem` model aliases and pickle
metadata. The approved 0.4.0 retirement removes flat DEM modules and their pickle
lookups. Other definitions use their canonical module metadata; root exports
still refer to those same objects. No replacement class or registry is added.
Source inspection and private instrumentation target the physical owner.

For a ten-minute change, run `PYTHONPATH=. python
examples/qec/dem_model_and_chunks.py`, then edit the module owning the behavior and
run its existing focused tests plus `tests/qec/test_restructure_dem.py`. The example
needs no optional SDK: it reports rates `[0.3125, 0.25]`, two graph edges, successful
chunk closure and pickle class `flagquantum.qec.dem`. Preserve the frozen 001–005
fixtures; add a new behavioral test when extending supported input.

For correlated Pauli circuit noise, run `PYTHONPATH=. python
examples/qec/correlated_circuit_noise.py` and
`python -m pytest tests/qec/test_correlated_circuit_noise.py -q`. Branches at
one location share an `error_id`; separate locations remain independent.
Independent-only graph, text and decoder consumers retain their refusals.

## Reusable independent DEM sampling

`model.compile_sampler(seed=7)` returns a CPU `CompiledDemSampler` from
`flagquantum.qec.execution.dem_sampling`. `sample(shots, bit_packed=True,
return_errors=True)` returns detector, observable and error tables. Error column
`i` follows canonical `model.errors[i]`, including repeated signatures; it does
not follow the insertion order of constructor arguments. Replay accepts bool or
packed uint8 independently of output packing and ignores unused input padding.
Returned padding is zero and arrays own storage.

`sample_write` streams optional detector, observable and error outputs in `01`,
`b8`, `r8`, `ptb64`, `hits` or `dets`. Error files use M labels, detector files D
and observable files L. PTB64 requires a multiple of 64 shots. Zero shots produce
empty results without advancing RNG. Files never overwrite destinations; three
outputs publish sequentially, with owned-inode rollback on failure rather than
cross-file atomic visibility. Replay is validated on a bounded immutable snapshot.

The 64 MiB principal-data guard includes all native tables, result and replay
copies; it is not process RSS or model/parser heap. Each dimension is at most
65,536, targets at most 262,144, canonical text at most 1 MiB, aggregate final
outputs at most 1 GiB, and replay snapshots separately at most 1 GiB. Native
stripes have at most 1024 shots. Handles belong to the creating thread/process,
cannot be serialized, and invalidate after execution/publication failure.
Validation failures preserve the handle. No backend fallback is performed.

Every stated `error_id` is refused before preparation, preserving the existing
independence boundary. Use unchanged `dem_sampling()` for exclusive groups.
Seed reproducibility requires the same backend/build/SIMD and call partition;
SDK/native/platform byte equality is not promised for random draws. Exact replay
and deterministic mechanisms are the cross-implementation witnesses.

Run `python -m examples.qec.compiled_dem_sampler` with the existing PyMatching
extra and a Stim 1.16.0 SDK or embedded build. Expected output is
`shots 64 batches 4 logical_failures 0`. macOS arm64 offline wheel/sdist
builds and installed native acceptance passed, including an environment without
the Stim Python package. See the implementation acceptance record in
`docs/development/QEC_COMPILED_DEM_SAMPLING_PROPOSAL.md`. This bounded
capability makes no performance, threshold, full-parity or scalability claim.
