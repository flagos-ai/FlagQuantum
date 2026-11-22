# Mapping and routing

Compiler owns device connectivity, logical-to-physical layouts, SWAP routing,
route costs and directed-edge legalization in this directory. `coupling.py`
defines the undirected CouplingMap. `directed_topology.py` owns directed target
connectivity. `link_error_rates.py` owns calibration validation, deterministic
weighted path search and its bounded cache. `routing_cost.py` adapts device
connectivity and calibration into routing legality and cost calculations.
`routing_strategy.py` owns the strategy vocabulary, cost estimates, candidate
ranking and SABRE planning used to price candidates. `routing.py` validates
inputs, dispatches strategies and passes the selected plan to emission during
automatic compilation. `routing_emission.py` owns instruction emission for
restore-after-each-gate, persistent-layout and SABRE routes, layout restoration,
and routing evidence including post-optimization counts. `sabre.py` implements
the search and `layout.py` assigns wires.
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
are also retained as direct aliases. The moved emission helpers,
`_routing_metadata` and `record_post_routing_optimization` remain direct aliases
from `routing_emission.py`; the existing root-module exports are preserved.
Compiler owns those private aliases, which can be removed after repository
consumers migrate and private-import cleanup is explicitly approved, targeting
the next minor release (0.5.0).

```bash
python -m examples.weighted_routing
python -m pytest -q tests/integration/test_weighted_routing_golden_path.py tests/team/compiler/test_compiler_pipeline_replacement.py
```

The example compiles a circuit onto alternative device paths and verifies that
calibrated link costs select the lower-error path. Both routes use two SWAPs.
For a candidate-ranking change, edit `routing_strategy.py` and run
`tests/unit/test_compilation_routing_estimate.py` together with this example.
For entry validation or strategy dispatch, edit `routing.py`. For instruction
emission or routing evidence, edit `routing_emission.py` and extend the scenario
with the corresponding legality, layout restoration and rejection assertions.
Run the routing, persistent-layout, SABRE and directed-topology tests in `tests/unit`,
plus `tests/team/compiler/test_routing_conformance.py`. Emission uses Core IR,
operand semantics, ordering barriers, the topology cost model and existing SABRE
plans; it must not call the routing entry module or select a strategy.

SABRE reuses dependency lookahead only within one unchanged frontier. See the
[lifetime invariant and development evidence](../../../docs/development/COMPILER_FRONTIER_LOOKAHEAD_REUSE.md)
before changing frontier advancement or speculative SWAP recovery.

For calibration or weighted-path changes, edit `link_error_rates.py` and run
`tests/team/compiler/test_routing_link_error_weights.py` plus the weighted-routing
example above. This module depends on the connectivity types and the standard
library. It must not choose routing strategies, emit instructions or submit jobs.
`routing_cost.py` retains `LinkErrorRates` and `_quantized` as direct aliases;
existing exports from Compiler and routing modules resolve to the same class.
The class keeps its established `routing_cost` module identity so serialized
class references resolve through the original path. Instances continue to own
thread locks and do not support pickle serialization. Compiler owns the private
helper alias under the same removal condition and target version described above.

Final-layout restoration walks each leaf's unique tree path once. See the
[restoration traversal invariant](../../../docs/development/COMPILER_RESTORE_PATH_REUSE.md)
before changing token movement or removing an active tree leaf.
