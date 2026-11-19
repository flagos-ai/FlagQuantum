# Quantum walks with structured block encodings

Run all three graph examples from the repository root.

```sh
python -m examples.synthesis.community.paper_implementation_project.explicit_quantum_circuits_for_block_encoding.quantum_walks_via_efficient_blockencoding.overview.main
```

[The walkthrough](walkthrough.ipynb) covers the original 21 source stages.
For extended binary trees with 2, 4, 8 and 16 vertices, compare the source
structured oracle with FlagQuantum's Pauli linear combination. Both resource
rows count actual gates in the CX/U basis. The Pauli implementation reports
its own normalization and clean work bits. Each structured encoding is
checked on all input columns. Two distinct coherent inputs check the Pauli
block at each size.

The eight-site circulant example preserves its increment, inverse increment
and register swap. Two reflections implement the original Chebyshev example,
including its coherent phase. Readouts align that global phase as in the
source. Finally, reuse one native instruction template per discriminant to
advance the 16-vertex walk through each of its 20 repetitions. Both finite
experiments use 1,000 seeded local draws per point because the upstream
Notebook defers its shot count to execution defaults. The plotted exact
probabilities are independently labeled.

The 4-vertex projected blocks are checked against the mathematical
discriminants, including the source's rounded controlled rotation. Full
complex states for all 40 walk points agree with independent real rotations,
Hadamards, reflections and swaps. The (k=5) guide is the source's rounded
heuristic. The marked trajectory remains oscillatory over the full range.

[Source mapping](provenance.json) · [Third-party license](../../../../../CLASSIQ_LICENSE.txt)
