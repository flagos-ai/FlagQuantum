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
| `construction.py` | Circuit/noise mechanisms; declarative layouts and the existing execution/signature reference helpers. |
| `circuit.py` | Model/IR realization; owned IR and noise channels. |
| `chunks.py` | Detector layers, seams, stitching and closing; models and circuit declarations. |
| `graph.py` | Graphlike mechanisms and weights; models. |

The model/text/matrix path must not eagerly import compiler, runtime or execution.
The legacy QEC root remains eager; package conversion does not change its import
behavior. Public users can keep importing `DetectorErrorModel`, `DemError`,
`DecodingGraph` and the chunk operations from `flagquantum.qec`.

`dem/__init__.py` and the former flat modules are explicit compatibility aliases.
They reference the canonical definitions, including historical helper imports;
they define no replacement classes or registries. Classes, methods and functions
retain their historical `__module__` and qualified names for pickle and existing
caller-owned safe-global lists. Core owns these paths. No removal release is
approved; removal requires an approved API/deprecation and artifact migration plan.
Source-inspection tools must use the physical owner or a method's code filename.
Private instrumentation that replaces module globals must target that owner.

For a ten-minute change, run `PYTHONPATH=. python
examples/qec/dem_model_and_chunks.py`, then edit the module owning the behavior and
run its existing focused tests plus `tests/qec/test_restructure_dem.py`. The example
needs no optional SDK: it reports rates `[0.3125, 0.25]`, two graph edges, successful
chunk closure and pickle class `flagquantum.qec.dem`. Preserve the frozen 001–005
fixtures; add a new behavioral test when extending supported input.
