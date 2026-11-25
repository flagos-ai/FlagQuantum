# Task scheduling with native QAOA

Run [the walkthrough](walkthrough.ipynb) or the complete workflow from the repository root.

```bash
python -m examples.synthesis.applications.logistics.task_scheduling_problem.overview.main --figures scheduling-figures --results scheduling-results.json
```

Both original task graphs, capacities, workloads, and eight-layer QAOA programs are retained. Shortest-path restrictions fix impossible assignment variables to zero. The small model keeps six active decision bits and six slack bits, and the large model keeps fifteen active decision bits and nine slack bits. The decoded output restores the original three-by-four and five-by-six assignment matrices.

The penalty is 20 for every original completion, capacity, and precedence constraint. Bounded binary slack uses a repeated final weight when necessary, such as `(1, 2, 1)` for capacity 4. Freshly generated large-model Ising coefficients match the frozen quantum example's complete coefficient multiset.

Native qfunc synthesizes the generated cost polynomial. Native COBYLA executes the source's explicit 20-query and 1-query budgets with 2,048 actual shots per query. It evaluates sampled CVaR at alpha 0.3 and 0.6. Its initialization uses a documented coupling-norm schedule, and the final sample uses the best actually evaluated parameters. This implementation has its own optimizer trajectory.

The exact classical reference enumerates the original feasible task times. The observed small and large schedules are plotted as returned, including missing or invalid assignments. Independent Circuit and scalar constraint checks cover the quantum state and the cost encoding.
