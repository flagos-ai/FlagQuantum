# Mapping and routing

Compiler owns device connectivity, logical-to-physical layouts, SWAP routing,
route costs and directed-edge legalization in this directory. `coupling.py`
defines the undirected CouplingMap. `directed_topology.py` owns directed target
connectivity. `routing.py`, `sabre.py` and `layout.py` transform programs while
preserving their logical meaning and recording layout changes.

Allowed dependencies include Core IR, target and operator semantics, Compiler
lowering primitives, and the existing numerical libraries used by transforms.
This directory must not select execution resources, run numerical workloads or
submit jobs. Runtime consumes compiled circuits and owns execution planning.

The expert entry paths `flagquantum.compiler.routing`, `layout` and
`layout_planning` remain supported. Normal compilation enters through
`flagquantum.compiler.compile`. Import CouplingMap internally from `coupling.py`
and change routing behavior in its owning implementation.

```bash
python -m examples.weighted_routing
python -m pytest -q tests/integration/test_weighted_routing_golden_path.py tests/team/compiler/test_compiler_pipeline_replacement.py
```

The example compiles a circuit onto alternative device paths and verifies that
calibrated link costs select the lower-error path. Both routes use two SWAPs.
For a small change, modify route selection and extend this scenario with the
corresponding legality, layout restoration and rejection assertions.
