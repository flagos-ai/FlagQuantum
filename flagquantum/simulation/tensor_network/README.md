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

## Explicit dense register inspection

`register_tensor.py` owns output/input word-axis layout and missing-port zero
projection over existing Core signatures. `controlled_tensor.py` inserts a child
supplier's tensor into the active control block of an identity tensor. It uses
raw stored word indices and adds no compiler graph or contraction engine. This
explicit exponential-memory inspection path can feed `ContractionGraph`; it is
not the circuit execution backend or a gate-count estimator. See
[controlled register tensors](../../../docs/development/CONTROLLED_REGISTER_TENSORS.md)
for layout, callback ownership, source failures and existing append verification.
Run the two focused test files named there when changing these owners.

`register_tensor_binding.py` attaches dense register tensors to existing
integer-labeled `TensorNetworkNode` objects and groups pure-state results in
output/input order. It reuses the current greedy contraction owner, preserves
PyTorch autograd, and allocates no graph labels. See
[register tensor binding](../../../docs/development/REGISTER_TENSOR_BINDING.md)
for word bindings, directional/empty ports, source correspondence, measurement
branches and the append workflow. Open-system grouping and automatic component
flattening remain separate work.

`register_identity.py` supplies one binary identity node per unused direct-through
bit, using existing signature bindings and caller dtype/device. Pass only the
ports the graph owner has identified as direct connections. This preserves
otherwise absent unused-register axes without a dense word identity or an
external graph representation. See
[direct-through register identities](../../../docs/development/REGISTER_IDENTITY_TENSORS.md)
for alias rules, zero-bit behavior, gradients and measured append verification.

`component_tensor_nodes.py` wires an already selected typed Core SSA entry into
existing nodes using a caller-owned leaf supplier and Compiler-provided call
order. It allocates numerical edge labels and inserts the existing identities
for untouched returned inputs; it performs no decomposition or backend choice.
See [component tensor nodes](../../../docs/development/COMPONENT_TENSOR_NODES.md)
for interface, ownership, source connectivity checks, empty/scalar boundaries,
replacement suppliers and the measured append consumer.

`component_superoperator_nodes.py` reuses that SSA wiring loop for mixed
quantum/classical channels. Suppliers request fresh hidden environment edges
from the shared numerical label allocator and explicitly yield discard labels.
Existing conjugate-pair and trace helpers produce nodes for the existing channel
contraction boundary; pure-state admission remains unchanged. See the same
[component tensor notes](../../../docs/development/COMPONENT_TENSOR_NODES.md#mixed-component-channels-graph74)
for source connectivity, full-channel, branch, gradient and resource checks.

`register_zero_state.py` emits existing binary zero-ket nodes for caller-identified
pure-output leaves using current bit labels and caller dtype/device. It connects
Compiler zero-input wrappers through SSA wiring without constructing dense word
states or performing physical resets. See
[zero-state nodes](../../../docs/development/REGISTER_ZERO_STATE.md) for source
failures, gradients, complex amplitudes and measured append verification.

`conjugate_pairs.py` doubles existing numerical nodes with independent even/odd
forward/backward labels and elementwise conjugation, preserving axis order and
PyTorch autograd. It does not build a measurement channel or implement discard.
See [conjugate tensor pairs](../../../docs/development/TENSOR_CONJUGATE_PAIRS.md)
for source correspondence, density/branch tests and separate resource counts.

`trace_indices.py` merges caller-selected exposed forward/backward edges in
existing doubled nodes. Existing contraction then performs partial trace without
projecting, resetting or normalizing. It preserves tensor data and gradients;
Program ownership and automatic discard discovery remain outside this owner.
See [conjugate index trace](../../../docs/development/TENSOR_TRACE_INDICES.md)
for source reindexing, complete channels, existing partial-trace replacement,
negative topology checks and measured branch verification.

Open-system register axes and contraction are owned by
`register_superoperator_binding.py`; see
[register superoperator binding](../../../docs/development/REGISTER_SUPEROPERATOR_BINDING.md).
It reuses the existing binder and greedy engine for quantum and classical bit
axes in rf/rb/lf/lb order, with explicit caller-owned discard merges.

`register_tensor_indices.py` describes caller-bound external edges with structured
or source-compatible friendly names. It keeps a diagnostic side table so display
name collisions cannot alter integer execution connectivity. See
[external index diagnostics](../../../docs/development/REGISTER_TENSOR_INDICES.md)
for pure/doubled naming, native refusal boundaries, and source correspondence.

`classical_action_tensor.py` enumerates an explicitly supplied classical action
through caller-owned codecs, yielding the pinned phase-free dense table. It
feeds the existing register binder and never substitutes for an arbitrary
phaseful quantum supplier. See
[classical action tensors](../../../docs/development/CLASSICAL_ACTION_TENSOR.md)
for memory, original codec failures, numerical phase limits and append checks.

`classical_action_tensor_node` attaches an explicit phase-free action and its
caller-owned codecs to the same leaf boundary consumed by
`component_tensor_nodes`. It reuses the existing binder, preserves caller
integer labels and dtype/device, and refuses classical ports before callbacks.
No automatic action lookup or quantum-phase substitution is performed.

`measurement_copy.py` owns the Z-measurement copy leaf with explicit input,
result and environment integer edges. Conjugate doubling and an explicit
trace of the environment yield the measurement channel; the leaf alone retains
coherent environment information. It emits no physical instruction or reset.
See [measurement copy](../../../docs/development/MEASUREMENT_COPY.md) for the
source mapping, channel composition, physical branch checks and separate costs.

The same owner supplies `x_measurement_node` from original MeasureX data in
input/environment/result order. Its raw entries match the pinned source exactly;
H/Z-copy factorization is independent conformance evidence. Environmental basis
changes do not add a physical gate. See the X-instrument section of the same
measurement-copy document for branch amplitudes and separate physical/model costs.
