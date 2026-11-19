# Complete native ring and torus walks

```sh
python -m examples.synthesis.tutorials.advanced_tutorials.discrete_quantum_walk_2d.discrete_time_quantum_walk.overview.main
```

Open `walkthrough.ipynb` from the repository root. It maps all ten original code
stages and includes executable native qfunc definitions. The one-dimensional
ring preserves six position bits, initial position 32, the complex H/S coin and
one/twenty steps. The two-dimensional torus preserves five bits per coordinate,
initial position (16,16), coin `(1,1,-1,-1)/2`, Grover reflection and twenty steps.

All three original observation batches make 1,000 actual seeded local draws.
The Notebook plots the twenty-step ring probabilities and the torus heatmap,
with the source's `Z[y,x]` coordinate convention. Sample counts come from native
execution and are never filled from the independent reference.

The coin remains an entangled environment after `drop`; it is neither reset nor
reused. Independent Circuit comparisons include its complete complex amplitudes
without phase alignment. Only additional compiler workspace must return to zero.
The one-dimensional reference uses an explicit controlled cyclic permutation;
the two-dimensional reference reuses the retained independent coin reflection
and cyclic-permutation matrices.

Classical step counts bind during synthesis, so the ring builds separately for
one and twenty steps. The overview calls the retained two-dimensional qfunc
through a checked twenty-step binding. Its sibling convenience wrapper keeps
its original sixteen-step evidence cap; this Notebook is not downsized to fit it.
No source importer, hosted IDE, hardware execution or scalability claim is used.

[Stage mapping](WORKFLOW.md) · [Third-party license](../../../../../CLASSIQ_LICENSE.txt)
