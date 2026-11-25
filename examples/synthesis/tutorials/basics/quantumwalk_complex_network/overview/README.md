# Native complex-network quantum walk

Run from the repository root:

```sh
python -m examples.synthesis.tutorials.basics.quantumwalk_complex_network.overview.main
```

Open [walkthrough.ipynb](walkthrough.ipynb) for graph generation, three walk rounds,
local position sampling, and the archived hardware comparison. It reuses the
[parent native qfunc](../main.py) and [independent Circuit](../comparisons/circuit.py).
Optional teaching dependencies are Matplotlib and NetworkX; no new production
package, source importer, hardware credential or remote job is needed.

The source's graph parameters and all historical numerical arrays are retained.
The eight-node comparison is a plot of archived source data, not a fresh eight-node
simulation or hardware run. The last source cell swapped the hardware-aware and
hardware-agnostic aliases; this port follows their original data declarations.
The source leaves sampling shots unspecified: local sampling explicitly uses
1000 shots and seed17. Graph layout also uses seed312 for reproducibility.
The four-node native program is checked against the regenerated neighbor table
before execution. Other graph models are not claimed as supported by this helper.

The source's optional credential-dependent hardware snippets are prerequisites,
not accepted local execution. Native device compilation has separate contracts;
this workflow makes no IBM routing, performance or hardware-equivalence claim.

[Workflow acceptance](WORKFLOW.md). [Source provenance](provenance.json).
[Third-party license](../../../../CLASSIQ_LICENSE.txt).
