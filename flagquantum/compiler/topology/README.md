# Mapping and routing

Compiler owns device connectivity, logical-to-physical layouts, SWAP routing,
route costs and directed-edge legalization in this directory. `coupling.py`
defines the undirected CouplingMap. `directed_topology.py` owns directed target
connectivity. `routing_cost.py` owns device cost models and weighted paths.
`routing_strategy.py` owns the strategy vocabulary, cost estimates, candidate
ranking and SABRE planning used to price candidates. `routing.py` emits routed
instructions and records evidence, reusing the selected plan during automatic
compilation. `sabre.py` implements the search and `layout.py` assigns wires.
These transforms preserve logical meaning and record layout changes.

Allowed dependencies include Core IR, target and operator semantics, Compiler
lowering primitives, and the existing numerical libraries used by transforms.
This directory must not select execution resources, run numerical workloads or
submit jobs. Runtime consumes compiled circuits and owns execution planning.

The expert entry paths `flagquantum.compiler.routing`, `layout` and
`layout_planning` remain supported. Normal compilation enters through
`flagquantum.compiler.compile`. Import CouplingMap internally from `coupling.py`
and change routing behavior in its owning implementation.

`routing.py` re-exports the strategy constants, `RoutingCostEstimate`,
`RoutingStrategySelection`, `estimate_routing_cost` and `select_routing_strategy`
from `routing_strategy.py`. Existing imports through both
`flagquantum.compiler.routing` and `flagquantum.compiler.topology.routing`
therefore resolve to the same objects. The previous private estimation helpers
are also retained as direct aliases. Compiler owns those private aliases, which
can be removed after repository consumers migrate and private-import cleanup is
explicitly approved, targeting the next minor release (0.5.0).

```bash
python -m examples.weighted_routing
python -m pytest -q tests/integration/test_weighted_routing_golden_path.py tests/team/compiler/test_compiler_pipeline_replacement.py
```

The example compiles a circuit onto alternative device paths and verifies that
calibrated link costs select the lower-error path. Both routes use two SWAPs.
For a candidate-ranking change, edit `routing_strategy.py` and run
`tests/unit/test_compilation_routing_estimate.py` together with this example.
For instruction emission, edit `routing.py` and extend the scenario with the
corresponding legality, layout restoration and rejection assertions.

SABRE reuses dependency lookahead only within one unchanged frontier. See the
[lifetime invariant and development evidence](../../../docs/development/COMPILER_FRONTIER_LOOKAHEAD_REUSE.md)
before changing frontier advancement or speculative SWAP recovery.
