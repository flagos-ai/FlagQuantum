# JAX Runtime Backend

This directory owns JAX backend selection, PyTorch/JAX bridging, execution
policy, distributed coordination, and evidence collection.

It does not own statevector, MPS, or tensor-network numerical kernels. Local
numerical operations live under `flagquantum/simulation/` and are imported
explicitly; this boundary does not proxy Simulation internals dynamically.

For local execution, start with `kernel.py`. For MPS circuit lowering and the
nearest-neighbor CX fast-path decision, start with `mps_kernel.py`. The latter
may recognize and lower circuit structure, but delegates tensor initialization,
updates, contraction, and observable evaluation to Simulation.

For sharded statevectors, `statevector_kernels.py` is a Runtime transport
adapter, despite its historical name. It converts instructions and Runtime
plans, selects local, pair-exchange, all-to-all, `pmap`, or `shard_map`
execution, and owns collective sequencing. Initial-state, local-gate,
pair-combination, all-to-all delta, and observable/loss mathematics live in
`simulation/jax_statevector.py`. This is the statevector stopping point: do not
move plan-aware collective code into Simulation or create mirror shard records
just to empty the Runtime file.

For sliced parameterized tensor networks, `tensor_network_gradients.py` owns
the Runtime-facing circuit/parameter adaptation, `JAXTensorNetworkNode`
records, slicing tasks, backend and collective selection, gradient lifecycle,
and result evidence. Raw node construction, contraction, slicing, and
observable/loss mathematics that do not require Runtime records live in
`simulation/jax_tensor_network.py`. This is the tensor-network stopping point:
do not introduce a second node record or a node factory merely to move the
remaining record-aware assembly out of Runtime. A numerical operation should
move only when it is independently reusable without importing Runtime plans,
tasks, records, policies, or collectives.

Run the boundary checks with:

```bash
pytest tests/unit/test_jax_simulation_boundary.py
```
