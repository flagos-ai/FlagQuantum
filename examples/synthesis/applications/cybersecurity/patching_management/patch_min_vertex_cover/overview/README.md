# Patch selection on the original attack graph

Preserve the original eight resources, seven vulnerabilities, 21 incidence
edges, 18 resource-conflict edges and three QAOA layers. The native parent
expresses the cover cost as a Python phase computation. The complete workflow
adds the graph model, bounded native optimization, actual local observations
and graph inspection after applying an observed patch set.

Run from the repository root:

```sh
python -m examples.synthesis.applications.cybersecurity.patching_management.patch_min_vertex_cover.overview.main
```

The Notebook maps all seventeen original code cells, including the illustrative
attack chain, both graph layouts, the conflict graph, model display, quantum
resources, 60-iteration search at CVaR 0.9, cost trace, actual optimized and
uniform samples, count histogram, selected cover and remaining attack graph.
Use a minimum-size **observed feasible** cover. If none was observed, report it;
do not substitute an exhaustively computed classical answer.

Native Nelder-Mead and exact local probabilities replace unspecified hosted
optimizer details. The schedule, 1,000 draws per distribution and seeds are
explicit. Force layouts replace Kamada-Kawai without adding SciPy. The
bipartition explicitly selects resource nodes and, after patching, includes
only surviving resources. No hosted optimizer trajectory equivalence, hardware
execution or quantum advantage is claimed.

The parent quantum program remains unchanged. See [WORKFLOW.md](WORKFLOW.md)
and [provenance.json](provenance.json) for source mapping and acceptance.
