# Complete surface-code package export disposition

Q-0900 covers every statement of the frozen `surface_code/__init__.py` at
Qualtran v0.7.0 revision `bcb17bb25360ed68ce3bc084645db3e929799389`, blob
`9f43f1b6ac684c0649707f3559a0c0d87dc69849`.

The file contains only a module docstring and 25 reexports. Each implementation
is already closed independently. The complete mapping is checked against the
hash-verified source AST in `test_surface_code_exports_file.py`. Every target is
imported through its actual authoritative module.

AlgorithmSummary belongs to the existing compiler-owned ModeledResources, with
rotation layers supplied to the time-step consumer. PhysicalCostsSummary belongs
to PhysicalResourceEstimate, which retains separate data/factory contributions.
DataBlock belongs to SurfaceCodeDataLayout and its four geometry implementations.
Factory, hardware, error and rotation abstractions belong to their existing QEC
resource modules. MultiFactory belongs to ParallelFactory. The deprecated
get_ccz2t_costs wrapper maps directly to PhysicalCostModel composition, with
budget and candidate-search helpers in physical_search. No duplicate summary,
deprecated wrapper or upstream-branded package facade is introduced.

The explicit module ownership follows the existing FlagQuantum API and protected
export policy. This disposition changes no root or stable exports. Users retain
`circuit.append(op)` for circuits and native domain imports for analytical
resource models. The original package's eager import order is not a semantic
algorithm and is not reproduced. Each definition's full source behavior, native
admission and upstream numerical defects remain recorded in its owning audit.
The package file adds no gates, states, measurements or physical estimates.

Complete implementations and actual consumer replacements are covered by the
concentrated QEC resource regression suite. Existing per-file original-source
tests, notebooks and numerical bounds are reused. Q-0924 strict budget rounding
remains separately open. UI and interoperability notebooks have their own parent
IDs and are not imported or marked complete by this package disposition.

All 445 concentrated resource regressions pass. The complete export-disposition
test passes strict mypy, Black and Ruff. Core scope, architecture, IR and
protected API checks pass.
