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

Pure suppliers that yield an integer discard token are refused at the first
token, before returning a partial graph or starting contraction. Use the
explicit mixed-channel entry when the supplied numerical graph implements a
trace; this is not a projection or reset. The source-token and mixed-path
boundary is documented with its tests in the component tensor notes.

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

`classical_parity.py` factors the original CBit Parity value mapping into binary
copy/XOR chain nodes. It uses the mixed graph's explicit fresh-label allocator,
retains forwarded x, and emits no physical instruction. Locked upstream Parity
has no tensor recipe; see [parity tensors](../../../docs/development/CLASSICAL_PARITY_TENSORS.md)
for the negative source contract and separate value-derived evidence.

`component_tensor_providers.py` offers an internal immutable explicit mapping
from exact Core function names to pure or mixed numerical suppliers. Its leaf
predicate and dispatch method let callers use the same bindings across existing
Compiler expansion and Simulation contraction; it does not discover providers
or guess from Python method names. See
[component tensor providers](../../../docs/development/COMPONENT_TENSOR_PROVIDERS.md).

`controlled_register_tensor_node` in `controlled_tensor.py` binds the existing
identity/active-child tensor for explicit quantum or classical controls. It
preserves phaseful children and refuses complex-to-real phase loss. See
[feedback tensors](../../../docs/development/CONTROLLED_REGISTER_TENSORS.md#classical-feedback-nodes-graph76)
for both control polarities, raw original tensor parity, prepared instruments,
all small measurement branches and the existing append workflow.

Direct external input discards in `component_superoperator_nodes.py` now produce
shared binary trace-effect tensors while retaining declared forward/backward
input axes. Internal traces still use the existing helper. See
[input discard effects](../../../docs/development/COMPONENT_TENSOR_NODES.md#direct-input-discard-effects-graph77)
for the locked source's missing-input failure, the executable Cast-wrapped
reference, full partial-trace channels and separate physical adaptation costs.

`component_dense.contract_component_tensors` and
`component_dense.contract_component_superoperator` compose existing SSA numerical
wiring and register grouping/contraction for an explicitly selected graph.
Compiler supplies root selection and visitation positions; Simulation imports
no Compiler transform. Existing pure/mixed supplier signatures, autograd and
empty-network refusals are preserved. See
[selected component dense contraction](../../../docs/development/COMPONENT_DENSE.md)
for complete source channels, all measured branches and separate cost evidence.

`register_tensor_data.py` owns raw word shape groups, control-active slices and
source-ordered directional projection. Validated `register_tensor.py` and
`controlled_tensor.py` reuse it, preserving their native checks and caller-owned
work. The raw layer has no foreign graph objects, circuit execution or provider
selection. See [the full source audit](../../../docs/development/REGISTER_TENSOR_SOURCE_AUDIT.md)
for dtype/view behavior, original-test replacement, native paired-name support,
inherited raw failures and separate primitive/model accounts.

`register_axis_groups.py` owns declared port order and grouping of actual external
register edges. Pure and mixed contraction reuse it after selecting exposed node
edges; `register_tensor_indices.py` remains the existing diagnostic side table.
The owner creates no graph, wires or quantum operations. The explicit
[source width policy](../../../docs/development/TENSOR_WIDTH_POLICY.md) is
available through the existing register and component consumers. See the [dense axis source audit](../../../docs/development/DENSE_AXIS_SOURCE_AUDIT.md)
for the complete original-test numerical replacement and contributor check.

The [tensor protocol source audit](../../../docs/development/TENSOR_PROTOCOL_SOURCE_AUDIT.md)
maps the complete flattening/classical-action original tests and all source
namespace exports to their native owners. Numerical grouping consumers import
`register_axis_groups` directly. Explicit provider declarations stay separate
from Compiler expansion and numerical execution; namespace asset closure does
not certify open source implementation boundaries.

`register_identity.register_bit_wiring_nodes` factors explicit equal-width
rebindings into binary identities. Direct-through placeholders reuse the same
owner; directional split/join ports use its caller-declared bit correspondence.
The [factorized Add composition](../../../docs/development/ADDITION_TENSOR_PROVIDERS.md)
replaces test-only dispatch with the existing exact-name provider mapping and
native CNOT/AND/basis/wiring suppliers, without adding automatic discovery.

### Source-ordered conjunction leaves

`conjunction.conjunction_tensor_node` supplies the original five-axis And data
in both target directions. It allocates no labels and does not execute measured
erasure. The [ladder workflow](../../../docs/development/CONJUNCTION_TENSORS.md)
composes real predicate bodies through explicit providers, with complete matrices,
physical append and every small erasure history checked against locked source.
Prefix outputs stay logically live until explicit cleanup; they are not scratch
claimed zero at forward exit.

### Bell composition and explicit custom tensors

The [Bell workflow](../../../docs/development/BELL_TENSOR_WORKFLOW.md) composes
existing root expansion, explicit providers and zero-state initialization. Run
`examples/algorithms/bell_tensor_workflow.py` for direct state preparation and the
unitary-to-state path. Swap dense CNOT and COPY/XOR data without changing the
consumer or physical program. Unexpanded roots without providers fail explicitly;
owned prepared outputs remain live, and no physical scratch is claimed clean.

### Contraction information after logical instance selection

Run `examples/algorithms/conjunction_contraction.py` to retain the original
MultiAnd instance IDs and inspect the existing native contraction profile. The
[workflow and cost conventions](../../../docs/development/CONJUNCTION_CONTRACTION.md)
explain atomic tensor stopping rules, imported-path comparisons, shape-only
planning and why arithmetic estimates, physical gates and process RAM are
separate quantities. No new planner or execution representation is introduced.

### Complete tensor notebook

Run all cells in
[`tensor_simulation.ipynb`](../../../examples/notebooks/tensor_simulation.ipynb)
for CNOT, rectangular And, MultiAnd inspection/custom selection, explicit custom
matrix binding, Bell expansion/refusal, and zero initialization. All 18 workflows
reuse the owners above. The [whole-source audit](../../../docs/development/TENSOR_SIMULATION_NOTEBOOK.md)
records exact reference execution, five native figures, register/work ownership,
resource units and presentation differences. It closes the concrete notebook
asset without claiming the full external optimizer or renderer interface.

### Explicit rank simplification

[`simplify_tensor_ranks`](rank_simplification.py) implements the exact,
shape-driven R pass over existing nodes and native einsum kernels. It protects
output labels and hyperedges, collapses repeated indices, and chooses connected
contractions without increasing rank. The [bounded example](../../../examples/algorithms/tensor_rank_simplification.py)
reduces three tensors to one before the same register consumer. See the
[source audit](../../../docs/development/TENSOR_RANK_SIMPLIFICATION.md) for
version pinning, pair-order evidence, scalar ownership and gradients. This is
explicit preprocessing; the complete width-triggered squeeze/ADCR policy is
opt-in through `simplification_atol`, preserving default training gradients.
The [dense entry disposition](../../../docs/development/DENSE_ENTRY_DISPOSITION.md)
maps all four source functions and the fresh-label factorized-provider entry.

### Protected squeezing and explicit column reduction

[`squeeze_tensor_axes`](simplification_axes.py) preserves output unit extents
and scalar nodes while removing unprotected size-one axes. R reuses its shared
output validation. [`find_tensor_column`](tensor_columns.py) detects support;
[`reduce_tensor_columns`](column_reduction.py) follows source cut order through
the existing slicer, which now handles every repeated-label occurrence.
The [executable example](../../../examples/algorithms/tensor_column_reduction.py)
composes squeeze/C/R before the same register consumer. Read the
[tolerance and gradient evidence](../../../docs/development/TENSOR_COLUMN_REDUCTION.md)
before applying C: support choices can discard small terms and ambient gradients
of zero entries. No default training path is changed; A/D and full-policy
integration remain open.

The [complete flatten source disposition](../../../docs/development/TENSOR_FLATTEN_SOURCE_DISPOSITION.md)
explains why explicit provider declarations replace Python method reflection,
which source decomposition errors retain leaves, and why native materialized
graphs propagate classifier failures instead of running a foreign lazy factory.
Use existing Compiler expansion and Simulation suppliers for this workflow.
