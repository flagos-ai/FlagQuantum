# Traffic-network optimization

Open [the walkthrough](walkthrough.ipynb) or run the original-sized workflow from the repository root.

```bash
python -m examples.synthesis.applications.telecom.network_traffic_optimization.overview.main --figures traffic-figures --results traffic-results.json
```

The native five-layer program retains all 24 edge/demand bits, the original normalized latency objective, flow constraints, and cubic shared-edge penalty. Each optimizer query and the final sample use 10,000 actual draws. The archived SciPy transcript expands its ten-query request to twelve for a ten-dimensional simplex. Native COBYLA uses an explicit twelve-query cap, original radius 1, and best actually evaluated parameters. Its own trajectory and budget stop are reported.

The independent integral-flow reference enumerates every feasible combination on the original acyclic graph. The original MILP coefficient vector uses the earlier edge-insertion order while its constraint matrix uses graph iteration order. Both its minimized objective and the selected route's actual latency are reported separately. The quantum objective uses the source graph's true latency order.

The final count table is read directly from quantum sampling. Only actually sampled feasible assignments receive route plots. Two complete native statevectors are checked against independent Circuit parity phases, including their global phase.
