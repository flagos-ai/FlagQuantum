# Simulation Engine Inventory and First Replacement Slice

Status: Simulation delivery candidate; inventory and verification evidence, not
a public contract.

Integration update (2026-09-04): local statevector, density matrix, noiseless MPS,
and tensor-network plan construction now belong to their Simulation implementations.
Local TN planning and state entry live in
`flagquantum/simulation/tensor_network/local.py`; Pauli/Hamiltonian plans and MPO
construction live in `tensor_network/observables.py`.
`tensor_network/entrypoints.py` retains stable entries and amplitude execution.
`Circuit` temporarily retains initial states and lifecycle caches. Migration must
preserve `fq.Circuit`, Runtime, and result contracts.

Review update (2026-09-05): `simulation/statevector/operations.py` and
`simulation/statevector/adjoint.py` own gate-matrix/diagonal actions, basis-block
merging, and local adjoint mathematics. Local sharding debug paths retain indexing,
ownership, and result assembly. TN local forward/reverse contraction belongs to
`simulation/tensor_network/stages.py`. JAX dtype, gate matrices, state actions,
sharded local observables/losses, batched MPS updates, and pullbacks belong to the
representation subdirectories under `simulation/jax/`. Distributed MPS/TN local
primitives belong to Simulation; Runtime retains records, scheduling,
communication, checkpoints, and evidence. Real local engines and fakes pass the
same first-slice conformance suite. Path audits and full gates establish the exit
conditions; `simulation_extraction` is complete.

First inventory: 2026-09-03. Latest review: 2026-09-05.

Shared baseline: `d7c56603e363bba95d5e98b9a77a75adb3c52e0d`

Original team branch: `codex/vnext-team-simulation`.
Integration branch: `codex/flagquantum-vnext-architecture`.

## 1. Conclusion

Numerical implementations now follow replaceable Simulation Engine boundaries.
The stable local PyTorch statevector loop is in
`flagquantum/simulation/statevector/local.py`; gate action and fusion are in
`flagquantum/simulation/statevector/operations.py`. Local MPS/TN and density
matrices also belong to Simulation. `flagquantum/runtime/executors/` retains
plan-aware adaptation, resource/communication orchestration, and result conversion,
calling Simulation primitives.

The first candidate is the existing local PyTorch `single_device_fast_path`:
execute validated `CircuitIR` from a supplied or canonical zero state and return
a complete batched state retaining the PyTorch autograd graph. Initial tests did
not change `fq.Circuit`/`run`, add exports, or create private Simulation contracts.
They characterize numerical behavior and demonstrate real/fake replacement at
`run_local_statevector()` through the same `fq.run(plan)` consumer. This existing
call site provides engineering evidence, not a new public contract.

## 2. Method and Boundaries

The inventory covers `flagquantum/simulation/**/*.py` and
`flagquantum/runtime/executors/**/*.py`, following actual consumers and external
authoritative implementations.

| Classification | Definition |
| --- | --- |
| Numerical algorithm | State representation, gates, decomposition/truncation, contraction, noise evolution, forward/backward/VJP, numerical stability |
| Kernel invocation | Selected Triton/JAX/compiled PyTorch kernels, tensor layouts, autograd bridges |
| Execution adapter | IR/plan/parameter conversion, instruction loops, Runtime/compatibility connections |
| Resource/communication orchestration | Devices, process groups, topology, ownership, transport, memory, checkpoints, recovery, training lifecycle, backend policy |
| Result conversion | State-to-measurement, local-to-aggregate results, framework arrays, records/evidence, compatibility facades |

Files may have secondary responsibilities. Primary classification determines
ownership; mixed responsibilities determine extraction order. Algorithm-local
workspace and tensor layout may remain in Simulation. Cluster resources,
communication lifecycle, and user policy do not belong there merely because they
are adjacent to mathematics.

The first slice is `single_device_fast_path`. Distributed statevector/MPS use
`sharded_across_ranks`; local TN slicing uses `manual_sliced_tensor_contraction`.
Cross-rank trajectory allocation is task parallelism, not single-state capacity
scaling. These semantics are not interchangeable.

## 3. Implementation Locations

| Capability | Numerical authority | Orchestration/consumers | Status |
| --- | --- | --- | --- |
| Local statevector | `simulation/statevector/local.py` loop; `simulation/statevector/operations.py` layout, gates, matrix composition, compressed basis expansion, fusion; `simulation/triton_kernels/statevector_gates.py` local gates and cross-shard CX control-one pack/unpack CUDA kernels | `runtime/execution.py`, `Circuit.state()`/`Circuit.run()` | Production supported; Simulation owns numerics, Circuit temporarily owns initial state/cache; distributed executors reuse matrix/index math while owning transport |
| Small specialized statevector | `simulation/statevector/small.py` | Specific models/benchmarks | Specialized 2--4 qubit data-reuploading kernels, not a general engine |
| Distributed statevector | `simulation/statevector/operations.py`, `simulation/statevector/adjoint.py`, `simulation/triton_kernels/statevector_*` rank-local primitives | `runtime/executors/statevector/` planning/models/forward/reverse/forward_executor/training/checkpointing/gradient_reduction | Runtime owns amplitude/qubit-address ownership, communication, chunks, lifecycle; no duplicate local gate math |
| Split real/imag and Double-Single | `simulation/statevector/split_real_imag.py` P0/P1/P2 gates, zero-state loops, Pauli reductions; `simulation/statevector/double_single_host_gates.py` and `simulation/statevector/double_single_device_gates.py` matrix generation; `simulation/statevector/double_single.py` P3/P4 initialization, gates, normalization, Pauli reductions | Runtime bindings, platform identity, precision plans/authorization, encoding, observables, parameter-shift scheduling, P5 autograd/SGD, conformance/results | P0--P4 numerics extracted; P3/P4 adapters remain separate for ingestion/path evidence; P5 one-line SGD does not justify another kernel helper; experimental paths are not defaults or equivalent FP64 |
| Local MPS | `simulation/mps/local.py` noiseless loop; `mps/noisy.py` lowered single-trajectory loop; `mps/models.py`, `mps/state.py`, `mps/factorization.py`, `mps/static.py`, `mps/tebd.py`, `mps/dense_island.py`, `mps/brickwork.py` | `mps/entrypoints.py`; `runtime/trajectories/mps.py` streams, scheduling, recovery, aggregation | Independent single-device numerics accept lowered IR, initialized state, explicit RNG |
| Distributed MPS | `simulation/mps/rank_local.py`, `mps/site_kernels.py`, `mps/compiled_layers.py`, `mps/factorization.py`, `mps/canonicalization.py`, `mps/reverse.py`, `mps/observables.py` | `runtime/executors/mps/` forward/reverse/state/distribution/communication/*transport/planning/training_engine/checkpointing/production/profiling | Batched contractions, reverse factorization/truncation, canonical decomposition/residuals, gates, QR, VJP, observable/MPO scans extracted; Runtime owns transport, memory admission/microbatches, sweeps, rebalance, tapes/checkpoints, collectives, cross-rank observable pipelines, evidence |
| Local TN | `simulation/tensor_network/local.py` plans/state; `tensor_network/observables.py` observable plans/MPO; `tensor_network/state.py`, `tensor_network/contraction.py`, `tensor_network/stages.py`, `real_imag_kernels.py` | `tensor_network/entrypoints.py`, `tensor_network/path_search.py` | Local execution/observables separated; path search needs further consolidation; experimental |
| Distributed TN | `simulation/tensor_network/stages.py` pair contraction/pullback, high-rank fallback, compensated accumulation | `runtime/executors/tensor_network/` DAG/schedules, tapes/checkpoints, ownership, communication, evidence | Local forward/reverse math extracted; sliced/sharded reverse lifecycle remains coupled to checkpoint/communication plans; production transport uncertified |
| Density matrix | `simulation/density_matrix.py` | Runtime noise registry | Exact local evolution, Kraus action, measurement in Simulation; Runtime handles lowering orchestration/dispatch; old `simulation.noise` facade removed |
| Noise models/lowering | Markovian Kraus math in density kernels, `simulation/statevector/noisy.py`, `simulation/mps/state.py`/`mps/entrypoints.py` | `flagquantum/noise/` semantics; `flagquantum/compiler/noise.py` lowering; `runtime/planner/noise_selection.py` selection; `runtime/trajectories/` shared infrastructure | Simulation owns evolution, not NoiseModel, lowering, or selection policy |
| Trajectories | Lowered batched statevector loop/kernels in `simulation/statevector/noisy.py`; MPS in `simulation/mps/noisy.py` | `runtime/executors/statevector/noisy.py`, `runtime/trajectories/` seeds, ownership, statistics, checkpointing, aggregation; execution files still reduce/save | Sampling/normalization are numerical; IDs, streams, readout errors, cross-rank aggregation, recovery, adaptive stopping are Runtime |
| Differentiation | Local statevector PyTorch graph; MPS/TN operations, Triton autograd, extracted JAX MPS pullbacks in `simulation/`; other explicit sharded reverse/adjoint in backends | Gradient ownership/reduction, training, optimizers, checkpoints, evidence | Preserve gradient ownership, dtype, conjugation, and forward distribution semantics |

## 4. Simulation Ownership Matrix

| Path | Primary role | Simulation owns | Mixed boundary |
| --- | --- | --- | --- |
| `linalg.py` | Numerics | PyTorch linear algebra | No evident Runtime policy |
| `statevector/small.py` | Numerics | Small exact evolution and Z expectation | Device-aware constant cache is valid; not general execution contract |
| `mps/state.py`, `mps/factorization.py`, `mps/low_rank.py` | Numerics | States, gates, decomposition, truncation, errors | Environment-controlled kernel/decomposition policy should be resolved upstream and passed explicitly; dense correctness fallback stays visible |
| `mps/static.py`, `mps/brickwork.py`, `mps/tebd.py`, `mps/dense_island.py` | Numerics/kernels | Fixed-shape graphs, TEBD, dense islands, local compiled kernels | Distinguish cache/compile policy from Runtime/Compiler decisions |
| `mps/local.py` | Numerical execution | IR gates, fusion, bucket kernels on initialized states | No Runtime lifecycle |
| `mps/noisy.py` | Numerical execution | Lowered unitary/Kraus single-trajectory evolution | No Compiler/Runtime import, seed derivation, or checkpoints |
| `mps/entrypoints.py` | Compatibility/adaptation | Circuit/IR initialization, lowered internal entry, adaptive-bond reruns | `run_native` has Runtime orchestrate Compiler lowering; protected direct legacy entry preserves lowering signature |
| `runtime/trajectories/mps.py` | Runtime lifecycle | No MPS/Kraus mathematics; invokes supplied trajectory executor | Ownership, streams, convergence, retries, checkpoint/restart, rank aggregation |
| `mps/planning.py` | Mixed resource/execution policy | Algorithm shape/truncation estimates only | State objects must not select backends/kernels from environment |
| `mps/models.py` | Configuration/schedules/results | Internal config, immutable schedules, diagnostics | Runtime owns multi-trajectory results; models do not execute policy |
| `tensor_network/state.py`, `tensor_network/contraction.py`, `tensor_network/stages.py` | Numerics | Representation, local/sharded contraction, reverse, Kahan methods | Runtime/Core own execution plans and persistent records |
| `tensor_network/path_search.py` | Mixed algorithm planning | Contraction-order search | Device/compile policy and global budgets are upstream inputs |
| `tensor_network/local.py` | Numerical execution | IR-to-local-plan, state entry, template reuse | No Runtime/Provider dependency; consumes existing Circuit initial-state/cache lifecycle |
| `tensor_network/observables.py` | Numerics | Pauli/Hamiltonian plans, MPO compression, batched contraction | Compression device explicitly supplied; no rank/cluster environment reads |
| `tensor_network/entrypoints.py` | Stable entry/amplitudes | Thin public adaptation, amplitude projection/contraction | No Runtime/Provider policy; split amplitude code only for concrete benefit |
| `tensor_network/models.py` | Internal models | Nodes, contraction/slicing plans, compiled local schedules | Core proposals define stable cross-layer results/types |
| `real_imag_kernels.py`, `triton_kernels/**` | Kernels/numerics | Eager/Triton forward/backward | Platform capabilities and Runtime policy decide availability/fallback |
| `graph.py` | Protected compatibility | Root API compatibility export; no Compiler caller | Do not duplicate in Compiler; migrate only for a concrete caller and approved API change |

## 5. Runtime Executor Ownership Matrix

| Path | Primary role | Simulation owns | Runtime/Provider owns |
| --- | --- | --- | --- |
| `simulation/density_matrix.py` | Numerics | Construction, operator expansion, unitary/Kraus action, IR loop, measurement | Stable result projection remains Runtime/Core |
| Density adapter in `runtime/noise_registry.py` | Execution adapter | No numerics | Validate/dispatch lowered IR; direct compatibility calls request Compiler lowering |
| `statevector/split_real_imag*.py` | Mixed numerics/kernels | Evolution, precision extension, expectations/VJP; adjoint kernels in `simulation/triton_kernels/statevector_adjoint.py`; matrix generation moved to `simulation/double_single_*_gates.py` | Device identity, provider evidence, precision/fallback permissions, conformance |
| `statevector/forward.py`, `reverse_adjoint.py` | Distributed adaptation | Runtime-independent local gates, diagonal action, rank-pair/basis merging, derivatives, complex inner products | Groups, collectives, ranks/topology/ownership, global indices, chunks, Triton routing, switches, evidence; chunked local expectations depend on these semantics |
| `statevector/reverse.py`, `gradient_reduction.py` | Mixed kernel/adaptation | Autograd bridges, local gradient math | Groups, bucket policy, all-reduce, ownership/evidence |
| `statevector/local_execution.py` | Adapter | Matrix/diagonal actions delegated to `simulation/statevector/operations.py` | Backend policy, simulated ranks, shard indexing/ownership, real transport, reporting; no duplicate plan/shard types |
| `statevector/planning.py`, `models.py`, `environment.py`, `layout.py`, `kernel_dispatch.py` | Resource/communication | Algorithm constraints/cost inputs only | Plans, topology, policy, environment, Platform kernel capabilities, Core records |
| `statevector/forward_executor.py`, `training.py`, `checkpointing.py` | Lifecycle | No training lifecycle | Loops, failure coordination, optimizers, checkpoint/recovery, progress/timeouts |
| `statevector/noisy.py` | Mixed numerics/orchestration | Batched gates/Kraus sampling, normalization, observables | Trajectory ownership, collectives, checkpoints, failures, adaptive stopping |
| `simulation/mps/rank_local.py`, `mps/site_kernels.py`, `mps/factorization.py` | Numerics/kernels | Gates, environment transfer, QR/SVD | Memory budgets, microbatches, workspace pools, Runtime error translation; old numerical modules removed |
| `mps/forward.py`, `reverse.py`, `reverse_replay.py`, `reverse_*observables.py` | Mixed | Rank-local math, tape/VJP, observables | Ownership, transport sequence, collectives, lifecycle |
| `mps/state.py`, `records.py` | Mixed state/ownership | Algorithm-internal tensors | Topology ownership and cross-layer Runtime/Core records |
| `mps/communication.py`, `distribution.py`, `metadata_transport.py`, `reverse_transport.py` | Communication | Required communication operations only | Transport and process groups |
| `mps/training*.py`, `checkpointing.py`, `production.py`, `profiling.py`, `device_resolution.py` | Resources/lifecycle/results | Local loss/gradient kernels may move | Device selection, parameter broadcast, optimizers, persistence, production gates, observation |
| `tensor_network/sharded_kernels.py`, `sliced_reverse.py`, `reverse_dag.py` | Mixed adaptation | Pair contraction, individual/batched pullbacks, high-rank fallback, Kahan accumulation delegated | DAG/bucket schedule, tapes/cotangents, checkpoint plans, slice scheduling, reductions/transport |
| `tensor_network/distributed_execution.py`, `distributed_sliced_reverse.py`, `redistribution.py`, `partial_mesh.py` | Communication/adaptation | Local contraction calls | Groups, P2P/all-to-all, rank lifecycle, aggregation |
| `tensor_network/distributed_dag.py`, `sliced_tasks.py`, `multi_axis_sharding.py`, `joint_planning.py` | Planning | Algorithm feasibility/shape cost | Ownership/topology/memory/communication plans; Core cross-layer types |
| `tensor_network/dynamic_checkpoint.py`, `rematerialization.py`, `memory_evidence.py`, `distributed_optimizer.py` | Lifecycle/resources/results | Rematerialization costs, local update math | Durable checkpoints, budgets/evidence, optimizer ownership, execution policy |
| `simulation/jax/primitives.py`, `simulation/jax/statevector/kernels.py`, `simulation/jax/tensor_network/{models,contraction,kernels}.py` | Numerics | Dtypes, instruction/Pauli matrices, initial shards/local execution, cross-rank gate math, local observables/losses, TN nodes, greedy/sliced contractions/output losses | No Runtime/Platform dependencies; Runtime owns shards, communication permutations, pmap/shard-map, collectives, evidence |
| `simulation/jax/mps/kernels.py`, `simulation/jax/mps/batched.py`, `simulation/jax/mps/pullbacks.py` | Numerics | Initial states, single/pair/batched updates, swap routing math, statevector contraction, local observables/VJP, boundary and QR/SVD pullbacks | No Runtime/Platform dependencies; Runtime owns circuit loops, shards, parameters, communication, truncation policy, evidence |
| `jax/kernel.py`, `mps/lowering.py`, representation Runtime modules | Adapter | Calls Simulation implementations | JAX backend/device selection, rank tasks, pmap/shard-map, collectives |
| `jax/array_conversions.py` | Adapter | DLPack/array zero-copy semantics | Framework/fallback policy; external objects stop at boundary |
| `jax/*execution.py`, `backend_dispatch.py`, `statevector/training.py`, `mps/gradients.py`, `tensor_network/gradients.py` | Mixed adaptation | Local kernels | Profiles/backends, device counts, shards, training lifecycle |
| `jax/*planning.py`, `planning_core.py`, `runtime_environment.py`, `transport.py` | Resource/communication | Algorithm constraints/costs | Topology, environment, transport, device lifecycle |
| `jax/*records.py`, `*result.py`, `evidence_collector.py`, `release_policy.py` | Results/facades | Internal diagnostics | Core results/evidence, Runtime aggregation, release policy |

`__init__.py` maintains explicit backend boundaries without creating another
algorithm authority.

### JAX Runtime Boundary Review (2026-09-05)

Classify remaining JAX calls by responsibility rather than eliminating every
`jnp` call. Collectives, device placement, result shaping, and probes remain in
Runtime. One-line zero allocation or matrix composition moves only if it
constitutes duplicated algorithm authority, not to create tiny public helpers.
The substantive MPS environment transfer and Z observable calculation after
cross-rank tensor reconstruction in `mps/gradient_ownership.py` moved to
`simulation/jax/mps/kernels.py`; Runtime adapts rank records to arguments.

`statevector/kernels.py` retains instruction/plan adaptation, collective
permutations, and `pmap`/`shard_map`. Initial states, local gates, pair merging,
all-to-all deltas, observables, and losses delegate to Simulation. This path has
reached its stopping point; do not duplicate Runtime shard/plan types to move files.

`simulation/jax/tensor_network/` owns `JAXTensorNetworkNode`, label slicing,
greedy/sliced local contractions, and output observables/losses. Runtime
`tensor_network/contraction.py` owns task assignment, pmap/shard-map selection,
and collective reduction; `gradients.py` owns circuit/parameter adaptation,
gradient lifecycle, and evidence. Do not copy records or add wrappers just to
remove `jnp` calls. Extract further only for independently reusable numerics that
do not depend on Runtime policy and records.

`simulation/jax/mps/pullbacks.py` owns local parameter VJP, boundary RXX adjoints,
and QR/SVD canonicalization/truncation pullbacks. Remaining
`mps/backward.py`, `mps/pullbacks.py`, and `mps/canonicalization.py` logic is a
bounded rank protocol: placement, parameter ownership, exchanges, truncation
policy, optimizer lifecycle, and evidence. Small analytic checks and tensor shapes
verify protocol evidence; they are not another general MPS authority. No further
fragmentation is warranted without a second independent production consumer.

The unused JAX dtype `ContextVar` in `mps/lowering.py` was removed.
`simulation/jax/primitives.py` remains the sole numerical precision context.

## 6. Logic Outside Simulation Ownership

No credential, API token, or secret implementation was found in the two scanned
directories. Preserve that boundary. The following do not belong in engines:

1. Runtime selection/policy: distributed profiles, backends/modes, world-size
   inference, kernel/fallback authorization, memory/precision policy, release claims.
2. Device/platform selection: `resolve_device`, CUDA/JAX counts, provider
   identity, vendor routes, probes. Engines consume resolved handles/capabilities.
3. Cluster/communication lifecycle: groups, placement, P2P/collectives, watchdogs,
   coordination, topology, evidence aggregation. Engines may state operation needs,
   not own cluster policy or transport lifecycle.
4. Persistence/training lifecycle: paths, retention, recovery, durable jobs,
   retries, adaptive stopping, optimizer scheduling, progress. Numerical
   rematerialization/checkpoint-placement algorithms may advise; Runtime chooses
   when and where to persist.
5. Credentials/external services: absent now and restricted to QPU/Remote
   providers or external Compute Service in the future.
6. Stable cross-layer types: Core alone owns requests, results, evidence, failures,
   and serialization. Simulation may return internal diagnostics.

Simulation no longer imports Runtime. `runtime/trajectories/` owns
`MPSMonteCarloResult` and multi-trajectory lifecycle. Intertwined process-group and
kernel logic in the three distributed executors needs explicit boundaries and
replacement tests, not wholesale file movement.

## 7. First Replaceable Slice

### 7.1 Scope

Candidate: **Local PyTorch Dense Statevector Engine**.

- Input: validated `CircuitIR`, batched initial state (default `|0...0>`), bound
  parameters, and Runtime-resolved dtype/device/platform context.
- Output: complete complex state `(batch, 2**n_wires)` and diagnostics of actual
  numerical execution only.
- Semantics: `single_device_fast_path`; no distributed initialization or rank/
  cluster environment inspection.
- Gradients: preserve first-order PyTorch autograd and caller tensor ownership.
- Baseline: existing gates/custom matrices, batches, complex64/complex128 behavior.
- Exclusions: planning, compilation, measurement wrappers, noise, shots, sharding,
  checkpoints, device selection, fallback, provider identity, performance claims.

The capability matrix already marks local statevector training
`production_supported`. Its mature exact baseline, small interface, and absence
of approximation contracts make it a better first slice than density/MPS/TN.
The existing PyTorch path supplies the real implementation. Loops and kernel
dispatch are in Simulation; program, initial-state, and lifecycle caches can move
incrementally into internals or existing execution context without changing
Stable Core behavior.

### 7.2 Replacement Evidence

`tests/team/simulation/test_statevector_engine_characterization.py` checks:

- IR reconstruction versus `Circuit.state()` for batches, wire order, dtype, normalization;
- RY expectation gradients against analytic values;
- differentiable custom matrices retaining autograd.

`tests/team/simulation/test_statevector_engine_replacement.py` runs real
`run_local_statevector()` and a test-local fake through the same existing call
site and unchanged `fq.run(plan)` consumer. Both check batch, dtype, device, plan
identity, result semantics, and gradient ownership. Replacement changes no
planner, compiler, or public API. No product Protocol, registry, or export was
added. This proves the first implementation boundary, not completion of the final
Core contract.

## 8. Core Contract Proposal (Not Implemented)

Integration/Core must first approve a minimal versioned Simulation Contract.
Reuse `CircuitIR` and Core execution/accuracy/failure/evidence vocabulary rather
than inventing another dictionary schema.

| Item | Proposed constraint |
| --- | --- |
| `SimulationRequest` | Canonical executable/`CircuitIR`, bound parameter and initial-state references, `full_state` target, explicit precision/approximation/fallback decisions; first version world size 1 only |
| `SimulationContext` | Resolved tensor/device/kernel capabilities; no credentials, cluster policy, or vendor SDK objects; Core decides whether in-process nonserialized PyTorch tensors are permitted |
| `SimulationResult` | State payload, actual dtype/device, algorithm ID/version, approximation/truncation/fallback facts; Runtime projects stable `ExecutionResult` |
| Failures | Structured unsupported instruction/dtype/device, invalid initial state, numerical failure; no silent backend/CPU substitution |
| Engine behavior | `execute(request, context) -> result`; same consumer accepts real/fake; no planner, credentials, checkpoint paths, or group lifecycle |

Do not casually put `torch.Tensor` in infrastructure-neutral serialized Core
schemas. Separate stable serializable envelopes from in-process tensor handles;
Core/API owners decide lifecycle, residency, and DLPack representation. Do not
bypass approval through a Simulation-private Protocol.

`statevector/local_execution.py` already delegates matrix/diagonal actions to
`simulation/statevector/operations.py`. Remaining shard initialization, index
groups, ownership, reference-rank orchestration, and result assembly directly
consume/produce `DistributedStatevectorPlan` and `StatevectorShardState`.
Moving them wholesale would introduce a reverse dependency or duplicate types.
Keep this execution adapter, transport, dry runs, backend policy, and reporting
in Runtime.

## 9. Numerical Risks and Gates

| Risk | Acceptance |
| --- | --- |
| Wire/basis ordering | Bell, asymmetric input, nonadjacent/reversed wires, IR round trips |
| Batch/broadcast drift | Scalar/batched parameters, custom batched matrices, multiple initial states |
| Complex dtype/device conversion | Separate complex64/128 tolerances; no implicit CPU/precision reduction |
| Broken graph/conjugation | Analytic gradients, finite differences, gradcheck, custom matrices, complex-loss conventions |
| Fusion/Triton versus eager | Forward/gradient references and actual kernel/fallback records |
| Stale parameters/graphs in caches | Repeated Circuit parameter updates, refresh, training steps, results/gradients |
| In-place leaf mutation | Initial-state and parameter ownership checks |
| Lost custom initial state | `CircuitIR` does not contain `Circuit.inputs`; request must carry initial-state payload/reference; arbitrary Circuit and `run_native(IR)` are not necessarily equivalent |
| Runtime replanning/substitution | Supplied plan identity, explicit mode, fallback/fail-closed checks |
| Fake mistaken for scalability proof | Mark integration and `single_device_fast_path` only; no distributed claim |

Run real/fake implementations through the same conformance suite and keep Runtime's
statevector branch limited to request organization and result projection. Loop
extraction is complete; move helpers/lifecycle only for clear benefit. Preserve
protected signatures, defaults, failure stages, and serialization of `fq.Circuit`,
`run`, `plan`, and `ExecutionResult`.

## 10. MPS Boundary Audit (2026-09-04)

Compiled-layer execution, reverse factorization/bucket kernels, and
canonicalization numerics moved into Simulation. Runtime retains rank/site
ownership, scan/communication order, gradient collectives, checkpoints, and
evidence. A real two-process communication test established unchanged global
states across canonicalization migration.

Remaining concatenation, reshape, and stack operations package communication,
gradient buckets, and distributed results. Tensor usage alone does not make them
numerical algorithms; only MPS mathematical semantics belong in Simulation.
Unused private compatibility aliases were deleted. Large files alone do not
justify mechanical splitting.

Multi-trajectory results moved to `runtime/trajectories/result.py`; `result_factory`
injection and Simulation-to-Runtime type references were removed.
`mps/entrypoints.py` no longer calls Runtime. `simulation/noise.py` was removed
earlier. The reverse-dependency allowlist is empty. Stop extending horizontal
abstractions and prioritize the minimum vertical path and physical layout.

## 11. Exit Review (2026-09-08)

`simulation_extraction` requires:

1. Real engine/fake conformance with unchanged Runtime consumers.
2. JAX numerics/pullbacks extracted from `runtime/executors/jax/`, leaving backend,
   device, shard, and training orchestration.
3. Distributed TN reverse mathematics independent of ownership/checkpoints/groups
   moved into Simulation.
4. No Simulation-to-Runtime imports or corresponding architecture exceptions.
5. Passing full CPU numerical, replacement, architecture, and public API gates.

All five have evidence. Remaining executor tensor operations directly serve
ownership, communication, checkpoints, results, or evidence and do not form a
second numerical authority. Machine-readable status is `complete`.

## 12. Compiled TN Forward Review (2026-09-05)

`execute_compiled_tn_forward_with_tape()` and local
`execute_contraction_stages()` both call `complex_einsum_pair()` in
shape-compatible buckets. They are not duplicate execution authorities: the
former consumes distributed DAG value IDs and retains full reverse tapes; the
latter consumes local plans and releases intermediates after their last use.

Do not adapt objects merely to reuse the entire local executor or add wrappers
that only forward equations/tensors. Runtime owns DAGs, bucket order, and tapes;
Simulation owns contraction/pullback primitives. Extract another helper only
when two paths share a second independently reusable numerical behavior.

## 13. Distributed TN Reverse Review (2026-09-05)

`reverse_dag.py`, `sliced_reverse.py`, and `distributed_sliced_reverse.py` delegate
pair contractions/pullbacks, high-rank fallback, and Kahan accumulation to
`simulation/tensor_network/stages.py` or `simulation/real_imag_kernels.py`.

Remaining stacks, slice merges, cotangent-map accumulation, and finiteness
statistics express schedules, slice/shard ownership, tapes/checkpoints,
collectives, and evidence. Stop extraction here. Do not introduce batch wrappers,
mirrored records, or generic executors to reduce tensor calls. Further extraction
requires a second consumer independent of Runtime DAGs, tasks, checkpoints,
ownership, groups, and evidence types.

## 14. Transitional Code Review (2026-09-06)

The review first searched for unused items before selecting another slice. It
found no tracked implementation removable without changing protected APIs,
serialized artifacts, or execution semantics:

- Executor private definitions still have code/test consumers. Remaining tensor
  operations serve plans, ownership, transport, checkpoints, or evidence.
- OpenQASM/QCIS exports moved to `compiler/openqasm.py` and `compiler/qcis.py`;
  internal callers switched and old utility implementations/exports were removed.
- Remaining `compilation` modules carry protected plan, serialization, and
  calibration semantics requiring public contract migration first.
- `_gateways/mcp` contains no tracked production implementation to delete.

Deletion has reached its safe stopping point. Choose subsequent migrations by
real call chains. Prefer a complete independent numerical behavior in
`runtime/executors/statevector` with no plan/device/communication/checkpoint/
evidence dependency; otherwise preserve the boundary without another wrapper.

## 15. Double-Single Diagnostic Conversion (2026-09-06)

P2--P5 result objects duplicated high/low-to-CPU-float64/complex128 conversion.
`DoubleSingleTensor.to_float64()`,
`DoubleSingleComplexTensor.to_complex128()`, and existing `to()` already own that
representation. Runtime results now compose those methods and detach before
return. Three duplicate helper groups and one inline implementation were removed
without changing result types, plans, devices, or kernels. CPU conformance covers
state, expectation, gradient, and optimizer diagnostics. Do not add manual
high/low reconstruction for similar results.

P1--P5 complex128 reference conformance now reuses
`simulation.pauli.pauli_product_statevector_expectation()` instead of direct
gate-matrix/statevector actions. Runtime still constructs circuits, schedules
parameter shifts, and sets thresholds; Simulation owns dense Pauli-product math.

Observable-to-Pauli adaptation, parameter occurrence locations, per-occurrence
shifts, and profile scalar/dtype limits are execution validation/orchestration.
P1/P3/P4 parameter-key stringification and collision checks now share one private
Runtime function, while each path retains its value/device/precision validation.
No execution policy moved into Simulation or Core.

P2/P3/P4 comparisons of requested versus executable precision plans and requested
versus certified error bounds now share two private functions in existing Runtime
modules. Profiles retain plans, thresholds, unsupported fields, and error text.
No new types, registries, or cross-domain dependencies were added.

P0--P4 platform identity, operator-profile loading, preflight, and evidence-ID
projection share a private Runtime entry. Existing independent probes/profile
names remain. Only duplicate P3/P4 orchestration was removed; capability claims
remain separate and provider/preflight logic stays outside numerics.

P5 Double-Single SGD depends on P5 state, canonical parameter order, device
consistency, and evidence, with no second independent consumer. Do not add a
Numerics optimizer primitive or Algorithms facade just to relocate it. Parameter
keys reuse P1--P4 normalization; scalar pairs, learning-rate checks, updates, and
finiteness remain in the slice. Revisit only for a second real consumer or approved
optimizer contract.

Five existing P0--P4 named entries in `runtime/operator_probes.py` remain for their
callers. Shared FP32 profile loading, execution probing, and capability decisions
now use one private implementation, avoiding divergent dtype/device/evidence
semantics while preserving entries.

## 16. Statevector Adjoint Review (2026-09-06)

`runtime/executors/statevector/reverse_adjoint.py` delegates analytic rotation
derivatives, real-valued complex inner products, and chunked Z expectations/
adjoints to `simulation/statevector/adjoint.py`. Pure numerical tests moved into
Simulation tests and no longer build references through Runtime bindings or
reverse objects.

Runtime retains shard indices/chunks, forward recomputation, checkpoints,
persistent wire layouts, P2P, gradient collectives, Triton routes, and evidence.
`_local_expectation_z*` traverses tensors using Runtime plan boundaries/global
indices. `_fused_sharded_1q_vjp_adjoint` owns communication pipelines and evidence
counters. Moving these would introduce plan dependencies or duplicate transport
contracts. Stop until a complete independent numerical behavior appears.

## 17. Statevector Forward Review (2026-09-06)

`simulation/statevector/operations.py` owns local/diagonal gate actions, basis
indices/offsets, rank-pair merging, and gate-basis merging used by Runtime
`forward.py`. Pure tensor tests belong to Simulation; Runtime tests retain layout
policy, routing, device waits, communication workspace, plans, and evidence.

Remaining `_vectorized_*` entries adapt shard/plan records and arrange cross-shard
P2P/collectives, workspace reuse, pipelines, and counters. Wholesale migration
would introduce Runtime models or another shard contract. Numerical kernels can
change independently; Runtime decides when and through which plan/route to invoke them.

## 18. Local Distributed Measurement (2026-09-06)

The local distributed development path now reuses batched `expectation_z()` from
`simulation/statevector/noisy.py` instead of duplicating reshape, marginalization,
and per-wire assembly. Simulation owns full-statevector all-wire Z expectations.
Runtime decides whether to measure and when to pass local distributed results.
Existing development-profile tests cover Bell-state batch shape, wire order, and
numerical agreement.

## 19. v0.1 Distributed Device Retirement (2026-09-07)

v0.1 DTensor devices, device-oriented gates, encoding, and measurement paths were
removed before the first public alpha without compatibility layers. Modern
execution uses only `simulation.gate_matrix.parameter_tensor()` and
`gate_matrix()` to convert IR instructions into resident, batched,
precision-aware gate matrices that preserve gradients.
