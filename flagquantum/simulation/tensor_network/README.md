# Tensor-network numerics

This package owns tensor-network numerical data and algorithms. It does not
select devices, schedule distributed work, manage jobs, or produce runtime
evidence.

- Start in `models.py` for tensor nodes, contraction plans, compiled schedules,
  slicing plans, and local expectation plans.
- Start in `state.py` for the local tensor-network state, cached contractions,
  observables, amplitudes, sampling, and numerical diagnostics.
- Start in `stages.py` for pair contraction, pair pullback, compiled-stage
  execution, high-rank fallback, and compensated accumulation.
- Start in `path_search.py` for greedy, multistart, tree-reconfiguration, beam,
  and bounded-optimal contraction-order search.
- Start in `contraction.py` for slicing-plan construction, contraction profiles,
  and sliced contraction execution.
- Start in `local.py` for Circuit/IR adaptation, local plan construction, and
  local numerical execution.
- Start in `observables.py` for Pauli/Hamiltonian plans, MPO compression, and
  batched observable contraction.
- Start in `entrypoints.py` for the stable tensor-network wrappers, amplitude
  projections, and Circuit/IR input adaptation.
- Import the owning submodule directly; this package does not re-export a
  tensor-network facade.
- Run `python -m pytest tests/test_tensor_network.py -q` after a typical local
  change.

## Shape-only search and real execution

`TensorNetworkNode` contains a real PyTorch tensor. Search intermediates use
the private `_SearchNode` in `path_search.py`, which can hold either a tensor
or `_DryRunTensor` metadata. This keeps planning shapes larger than PyTorch's
storage-size limit out of the public tensor contract. Tree reconstruction and
external-path conversion use the same private representation.

Greedy, beam, and optimal search return tensors for real execution; dry runs
may return shape-only metadata. Their overloads express that distinction.
Numerical branches unwrap real tensors explicitly; shape-only output
reordering must never allocate a tensor. Do not replace the metadata object
with a meta-device tensor: meta tensors still impose PyTorch size limits.

For changes to this boundary, run `tests/unit/test_tensor_path_search.py`,
`tests/unit/test_tensor_stages.py`, and `tests/test_tensor_network.py`. Preserve
contraction steps, cache identities, output values, and gradients. The search
regressions include output reordering and 96-axis products without allocating
the represented state. These are planning checks, not simulation-capacity
evidence.

## Exporting a path cost

A contraction cost is a property of one contraction order on one network, so it
is reported for an order the caller selects rather than as a property of the
program. Three surfaces export it, all of which plan without allocating the
represented state:

- `contraction_cost(strategy)` returns `estimated_cost` and `peak_size`, and adds
  `n_slices` when the selected order is a sliced one.
- `contraction_profile(strategy)` returns a `TensorNetworkContractionProfile`
  whose `summary()` reports `strategy`, `estimated_cost`, `peak_size`, `n_steps`,
  `output_size`, `n_slices`, `sliced_labels`, `total_intermediate_size`, and
  `contraction_path_source`.
- `tensor_network_contraction_peak_bytes(...)` converts a selected order's
  `peak_size` into bytes using the widest node element size.

`contraction_path_source` names the planner that produced the order, because two
planners can return different costs for the same network and a recorded cost is
only traceable with the search that produced it. This package's own search and
its own slicers report `native`. A sliced profile inherits the source of the
slicing plan it is built on, so when a slicing plan carries an external
provenance the profile reports that instead; `cotengra_slicing_plan` is the route
that records one, as `cotengra:<version>`. `opt_einsum` is surveyed in the parity
strategy's licence table but is not adopted here, so no route reports it.

These costs are estimates of a plan, not measured runtime or achieved-memory
evidence. Every native objective is a self-built search heuristic. The `optimal`
objective is exact only up to the node ceiling `_contract_nodes_optimal` accepts
(`max_nodes`, 7 by default); above that ceiling it delegates to a bounded beam
search, so an `optimal` cost must not be presented as exact for a larger network.
No strategy here may be described as a cuTensorNet-class or cotengra-class
planner.
