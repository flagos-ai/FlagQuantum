# JAX tensor-network numerics

This package owns dependency-light JAX tensor-network mathematics. It does not
select devices, organize collectives, consume Runtime plans, or emit evidence.

- Start in `kernels.py` for node construction, contraction, slicing,
  observables, and contracted-output loss.
- Import the owning module directly; this package does not re-export a facade.

Parameterized physical state/expectation graphs use an explicit admitted dtype
in `kernels.py`; local rank partial sums live in `contraction.py`. Runtime owns
precision policy, rank placement, real collectives and evidence. For a typical
change, run `python -m pytest tests/unit/test_jax_tn_graph_ownership.py -q` with
optional JAX installed. Its owner replacements exercise unchanged Runtime
consumers, JIT and gradients; CPU simulation is not hardware scaling evidence.
