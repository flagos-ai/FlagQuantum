# Boolean oracle sketching

Run the [notebook](walkthrough.ipynb) or the local workflow from the repository root.

```sh
python -m examples.synthesis.algorithms.quantum_primitives.quantum_oracle_sketching_boolean.overview.main
```

The example retains the seeded eight-value Boolean table, the original
200-sample demonstration, the 500-sample Bell sketch and the 4,000-sample
single-realization check. It also evaluates 1,500 separately sampled sketches
at each of the twelve original sample counts. NumPy reproduces the source
random sequences.

The native qfunc accumulates the commuting sample phases by basis word and
applies those exact diagonal effects. The same path supports the
inverse-probability weights for a known distribution. Its full state matches
a Circuit that applies the per-sample weighted phases in sequence.

The unknown-distribution correction uses only the stated probability
bounds and twenty sampled words. A Hadamard-test circuit encodes the sine
of the accumulated phases. The even degree-18 threshold polynomial is
recomputed with a bounded minimax fit and its QSVT phases are fitted from
scratch. The CPU example executes the complete phase sequence as a packed
matrix operation, checking both the full state and the zero-block response.
Each Bell demonstration draws 10,000 outcomes with a recorded local seed.
The fitted threshold error, block success probability, and conditional
Bell fidelity measure different parts of the workflow.
