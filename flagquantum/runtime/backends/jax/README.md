# JAX Runtime Backend

This directory owns JAX backend selection, PyTorch/JAX bridging, execution
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
QR/SVD pullbacks live in `simulation/jax/mps/pullbacks.py`. The Runtime modules
`mps/backward.py`, `mps/pullbacks.py`, and `mps/canonicalization.py` retain the
constrained rank protocol, device placement, parameter ownership, collective
exchange, truncation policy, optimizer lifecycle, and evidence records. This
is the MPS backward stopping point: the small analytic checks and tensor shapes
inside those protocol executors are evidence construction, not a second
general MPS implementation. Do not extract them into generic helpers unless a
second production numerical path consumes the same operation independently of
Runtime policy and records.

For sharded statevectors, start in `statevector/execution.py`.
`statevector/kernels.py` is a Runtime transport
adapter, despite its historical name. It converts instructions and Runtime
plans, selects local, pair-exchange, all-to-all, `pmap`, or `shard_map`
execution, and owns collective sequencing. Initial-state, local-gate,
pair-combination, all-to-all delta, and observable/loss mathematics live in
`simulation/jax/statevector.py`. This is the statevector stopping point: do not
move plan-aware collective code into Simulation or create mirror shard records
just to empty the Runtime file.

For sliced parameterized tensor networks, `tensor_network/gradients.py` owns
the Runtime-facing circuit/parameter adaptation, `JAXTensorNetworkNode`
records, slicing tasks, backend and collective selection, gradient lifecycle,
and result evidence. Raw node construction, contraction, slicing, and
observable/loss mathematics that do not require Runtime records live in
`simulation/jax/tensor_network.py`. This is the tensor-network stopping point:
do not introduce a second node record or a node factory merely to move the
remaining record-aware assembly out of Runtime. A numerical operation should
move only when it is independently reusable without importing Runtime plans,
tasks, records, policies, or collectives.

Run the boundary checks with:

```bash
pytest tests/unit/test_jax_simulation_boundary.py
```
