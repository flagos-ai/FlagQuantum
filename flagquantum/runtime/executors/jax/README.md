# JAX Runtime Executors

This directory owns JAX executor selection, PyTorch/JAX bridging, execution
policy, distributed coordination, and evidence collection.

It does not own statevector, MPS, or tensor-network numerical kernels. Local
numerical operations live under `flagquantum/simulation/` and are imported
explicitly; this boundary does not proxy Simulation internals dynamically.

## Layout

- `statevector/`, `mps/`, and `tensor_network/` own representation-specific
  planning, execution, records, and evidence.
- `kernel.py` owns the shared local JAX compilation entrypoint.
- `backend_dispatch.py` and `planning_core.py` own cross-representation
  dispatch and plan vocabulary.
- `array_conversions.py`, `runtime_environment.py`, `release_policy.py`, and
  `common.py` contain only helpers used across representation subpackages.

Representation-specific modules do not belong in this directory root. Move a
helper into a subpackage when only that representation consumes it; do not add
a forwarding module at the old path.

For local execution, start with `kernel.py`. For MPS circuit lowering and the
nearest-neighbor CX fast-path decision, start with `mps/lowering.py`. The latter
may recognize and lower circuit structure, but delegates tensor initialization,
updates, contraction, and observable evaluation to Simulation.

For MPS differentiation, local parameter VJPs, boundary-gate adjoints, and
QR/SVD pullbacks live in `flagquantum/simulation/mps/jax/pullbacks.py`.
The Runtime modules `mps/backward.py`, `mps/pullbacks.py`, and
`mps/canonicalization.py` retain the constrained rank protocol, device
placement, parameter ownership, collective exchange, truncation policy,
optimizer lifecycle, and evidence records. The actual rank-local RY state,
Z response and derivative calculations used by returned values, cotangents and
optimizer updates also belong to the existing Simulation pullback module.
Runtime retains the independent analytic boundary-gate checks and RY/RXX tensor
allocations used only to record memory witnesses. These evidence-only formulas
and shapes are not a second general MPS implementation; do not extract them
into generic numerical helpers solely to empty the protocol executor.

For sharded statevectors, start in `statevector/execution.py`.
`statevector/kernels.py` is a Runtime transport
adapter, despite its historical name. It converts instructions and Runtime
plans, selects local, pair-exchange, all-to-all, `pmap`, or `shard_map`
execution, and owns collective sequencing. Initial-state, local-gate,
pair-combination, all-to-all delta, and observable/loss mathematics live in
`flagquantum/simulation/statevector/jax/kernels.py`. This is the
statevector stopping point: do not move plan-aware collective code into
Simulation or create mirror shard records just to empty the Runtime file.

For sliced parameterized tensor networks, `tensor_network/gradients.py` owns
the Runtime-facing circuit/parameter adaptation, slicing tasks, backend and
collective selection, gradient lifecycle, and result evidence. The numerical
node record and local greedy/sliced contraction live in
`flagquantum/simulation/tensor_network/jax/models.py` and `contraction.py`;
lower-level kernels and observable/loss mathematics live beside them in
`kernels.py`. Runtime retains rank assignment, `pmap`/`shard_map` selection,
collective reduction, and evidence. A numerical operation should move only when
it is independently reusable without importing Runtime plans, tasks, records,
policies, or collectives.

Run the boundary checks with:

```bash
pytest tests/unit/test_jax_simulation_boundary.py
```

## Dense result readout

Statevector and TensorNetwork result adapters retain optional Torch admission,
qubit validation and reconstruction/cache. Raw probabilities and ordered Z
values are computed by Simulation Numerics; public method and alias identities
remain on the existing result types.

## Representation conversion ownership

`array_conversions.py` owns only the gate and parameter conversions actually
shared by representations. Statevector shard records are reconstructed in
`statevector/execution.py`; MPS split evidence is assembled in `mps/execution.py`;
Torch tensor-network nodes cross the JAX boundary in `tensor_network/execution.py`.
Tensor-network gradients reuse that conversion directly. These three helpers
have no forwarding bindings in the shared conversion module.

In-process statevector shard gate mathematics lives in
`simulation/statevector/jax/kernels.py::jax_apply_gate_to_shard_arrays`. Runtime
retains optional-engine readiness, matrix-shape admission, plans, immutable
index references and result record assembly. The numerical owner receives arrays
and index values, never Runtime records or plans. It performs no collective and
does not establish multi-device or multi-node capacity. The existing communication
count and nonproduction classifications remain authoritative.
