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
