# FlagQuantum Release Notes

Release notes describe user-visible behavior and support-boundary changes.
Benchmark claims require audited artifacts and are not inferred from this file.

## 0.2.0

- Established FlagQuantum IR, `fq.Circuit`, `fq.Module`, `fq.run`, runtime
  planning, explicit sharded statevector/MPS execution, distributed training,
  deployment packaging, and versioned result contracts as the maintained
  product architecture.
- Classified the v0.1 `DistributedQuantumDevice`, `GeneralEncoder`,
  `InvertibleUnitary`, DTensor interchange helpers, and device-oriented
  measurement path as an internal compatibility subsystem. They are excluded
  from the v0.2 stable API and cannot be used as evidence for v0.2 capability
  or patent claims.
- Deprecated construction of `GeneralEncoder` and `InvertibleUnitary`.
- Fixed the package release identity at `0.2.0`.

## Unreleased

- Added fail-closed single-node MPS development evidence on 8×A800:
  2/4/8-GPU rank-owned forward, 2/8-GPU packed boundary transport and fault
  cleanup, 2-GPU accelerator backward, and a 100-step 8-GPU SGD soak with
  stable memory and exact checkpoint/restart equivalence. This does not promote
  multi-node or release-certified support.
- Removed the pre-release `flagquantum.algorithms_stack` namespace. Import
  algorithm helpers from `flagquantum.algorithms` and optimization contracts
  from `flagquantum.algorithms.optimization`.
- Removed the pre-release `flagquantum.runtime_stack` compatibility namespace.
  Runtime integrations, benchmarks, plugins, and serialized references must use
  the canonical `flagquantum.runtime` packages.
- Removed the unused pre-release `flagquantum.core.circuit` compatibility shim.
  Import `Circuit` from `flagquantum`, or from `flagquantum.circuit` for an
  explicit implementation-module dependency.
- `fq.Circuit(n_qubits=...)` is now the preferred public spelling for circuit
  size, with `n_qubits` and Qiskit-compatible `num_qubits` properties.
  Positional construction and the `n_wires`/`nqubits` keyword aliases remain
  compatible, while conflicting counts now fail during construction. Internal
  IR, compiler, and runtime mappings continue to use wire terminology.
- Generated gate methods now accept optional semantic qubit keywords without
  removing concise positional calls: single-qubit gates use `qubit=`,
  controlled gates use `control=`/`target=`, and symmetric two-qubit gates use
  `qubit1=`/`qubit2=`. Duplicate, conflicting, and missing qubit arguments fail
  before instructions are added.
- Added staged hybrid optimization for named classical and quantum parameter
  groups. Local VQE convergence workflows can combine Adam/AdamW/SGD/L-BFGS,
  exact full or block QNG, and Rotosolve, with per-step gradient/update/evaluation
  diagnostics. Heisenberg helpers now provide phase-augmented bond-resolved HVA,
  dimer-singlet initialization, and exact small-system energy references.
  The convergence example preserves requested float64 parameters, supports
  identity-layer continuation, applies bounded-memory L-BFGS, and emits an
  explicit exact-energy convergence decision instead of treating loss decrease
  as convergence.
- Distributed MPS training supports an experimental `adam_lbfgs` schedule with
  owner-local limited-memory histories and collective scalar products. The N=8
  Heisenberg benchmark records aligned optimizer-step energy/error traces.
  Exact gradient policy now automatically retains factorization graphs for
  statevector-parity gradients; the recomputation pullback remains restricted
  to explicitly approximate-gradient work.
- `flagquantum.experimental.distributed.train_distributed_statevector(...)` now
  provides multi-step native
  PyTorch training with owner-sharded SGD/Adam state, checkpoint/resume,
  cancellation, memory preflight/measurement and structured lifecycle progress.
- Experimental statevector execution now preserves amplitude sharding through
  PyTorch backward for multi-layer shared-parameter circuits using rank-local
  rematerialization, explicit gate adjoints, and cross-rank VJP reduction.
- An experimental PyTorch-native distributed statevector forward executor now
  runs validated unitary IR with rank-local amplitude ownership across local and
  multi-sharded-wire gates on Gloo or NCCL. Full-state materialization is
  forbidden and forward-only training/release blockers remain explicit.
- Correctness infrastructure now generates deterministic circuit properties,
  covers every registered operator/lowering pair, versions numerical
  tolerances, retains minimal failure artifacts, fuzzes IR parsing, and
  classifies distributed no-progress with cleanup requirements.
- An experimental extension SDK now supports isolated backend, kernel,
  operator, compiler-pass, device, provider, measurement-collector, and planner
  plugins with version checks, capability negotiation, lifecycle containment,
  credential-safe configuration, and reusable conformance tests.
- The stable root API is bounded by a checked manifest. Experimental and
  compatibility imports no longer silently expand the supported surface.
- Circuit operations and backend lowering availability derive from one typed
  operator registry, with generated capability documentation.
- Runtime configuration is immutable, task-scoped, serializable, and propagated
  explicitly through compile, plan, and execution boundaries.
- Planner requests, executor observations, measurements, failures, provenance,
  and audit decisions use strict versioned contracts with deterministic JSON.
- Performance and memory baselines record uncertainty, correctness, progress,
  and measured hardware provenance without promoting development runs to
  release scalability evidence.
- Packaging enforces Python 3.10–3.12, bounded dependency families,
  reproducible source/wheel contents, lazy imports, and repository hygiene.
- Documentation derives stable API, operator capability, and current issue
  state tables from authoritative sources and fails CI when claims drift.

No release build or distributed scalability release artifact is declared by
these unreleased changes.
