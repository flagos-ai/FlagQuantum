# API Change Proposal 068: QBoson Kaiwu Ising provider boundary

## Status

**Draft; architecture and API review required before stable export.** A tested
prototype exists only in the provider-specific
`flagquantum.ecosystem.kaiwu` and `flagquantum.remote.kaiwu` submodules. It is
not re-exported by `flagquantum`, `flagquantum.ecosystem`, or
`flagquantum.remote`, does not appear in capability maturity, and carries no
hardware-support claim.

## Problem

QBoson's Kaiwu SDK accepts dense Ising matrices and returns spin samples from a
special-purpose coherent optical computer. This is not a gate-model circuit
service. FlagQuantum currently has no owned contract for preserving Ising sign,
QUBO auxiliary-spin, coefficient precision, sample-energy, or recoverable
remote-task semantics across that boundary.

Reusing the existing gate-model `DeploymentPackage`, `DeploymentResult`, and
`RemoteJob` would be incorrect. Those contracts bind a circuit artifact to
shots and bitstring counts. A Kaiwu result instead binds an Ising matrix to
ordered spin samples and independently recomputable energies; it may operate in
optimization or sampling mode and uses SDK checkpoint identity rather than a
circuit-deployment receipt.

The Kaiwu PyTorch Plugin adds another narrow requirement: its Boltzmann
machines synchronously call `sampler.solve(ising_matrix)` and expect a NumPy
array whose final column is an auxiliary spin. That compatibility surface must
not make the ecosystem adapter the owner of credentials or remote tasks.

## Decision

Keep two boundaries from the first supported slice:

- `flagquantum.ecosystem.kaiwu` owns pure matrix validation, energy convention,
  symmetric QUBO conversion, auxiliary-spin decoding, explicit precision
  preparation, and the synchronous plugin sampler surface.
- `flagquantum.remote.kaiwu` owns credentials, SDK/license loading, task
  submission, status normalization, bounded waiting, result validation,
  checkpoint recovery, receipt persistence, and provider evidence.
- The ecosystem sampler composes Remote through FlagQuantum-owned protocols.
  It never imports the vendor SDK, handles credentials, or implements a local
  fallback.
- The Remote provider accepts only already selected Ising matrices. It does not
  own QUBO conversion, precision policy selection, PyTorch tensors, or
  QDiffusion training.

No general `fq.Circuit` lowering, root `fq.run` route, second circuit IR, generic
job framework, or stable root-level optimization API is introduced.

## Proposed provider-specific API

The first review concerns the submodule APIs only:

```python
from flagquantum.ecosystem.kaiwu import (
    KaiwuSampler,
    canonicalize_ising_matrix,
    decode_qubo_spins,
    encode_qubo_as_ising,
    ising_energy,
    prepare_integer_precision,
)
from flagquantum.remote.kaiwu import (
    KaiwuCredentials,
    KaiwuSDKClient,
    restore_kaiwu_job,
    submit_kaiwu_task,
)

credentials = KaiwuCredentials(user_id="...", sdk_code="...")
client = KaiwuSDKClient(
    checkpoint_dir="/absolute/private-kaiwu-checkpoints",  # existing mode-0700 dir
    credentials=credentials,
    expected_version="1.3.1",
)
sampler = KaiwuSampler(
    client=client,
    task_name="bounded-qdiffusion",
    project_no="CPQC-project",
    requested_samples=10,
    timeout=3600,
    max_remote_calls=32,
    integer_target_range=(-127, 127),
)
samples = sampler.solve(ising_matrix)
sampler.last_job.save("/absolute/private-evidence/qboson-receipt.json")
```

The example is illustrative and does not approve these names as stable. In
particular, the SDK client cannot be promoted until a real pinned response
establishes provider task-ID, target, and terminal-state mappings.
Explicit or dedicated-environment credentials are trimmed, must form one
complete pair, and must contain only printable characters. Empty, partial, or
control/format-bearing pairs fail without echoing either value.
The prototype may persist a value-free schema of the documented
`get_task_result` dictionary—bounded safe field names, types, lengths, dtypes,
and shapes—to support that review. Unsafe or oversized names are omitted, the
serialized record is scanned for both credential values, and diagnostic
failure remains nonfatal to otherwise valid samples. It must not persist raw
provider values or infer a mapping from field names alone.

## Ising and QUBO semantics

- The provider spin domain is exactly `{-1, +1}`.
- A symmetric matrix `M` is evaluated as
  `E(s) = -s.T @ M @ s + bias`; both off-diagonal entries contribute.
- Diagonal Ising terms are retained and contribute a constant for spin states.
- Symmetric QUBO `x.T @ Q @ x + offset` uses one final auxiliary spin and is
  gauge-decoded with `x = (s * auxiliary + 1) / 2`.
- Input must be non-empty, square, finite, real, and symmetric. Unsupported
  values fail before submission.
- Lossy integer preparation is opt-in. Its scale, range, rounding, and errors
  are inspectable; silent clipping or rescaling is prohibited.

## Task and result semantics

- `submit_kaiwu_task` performs one initial submission and returns a detached
  Kaiwu-specific job. `status`, `result`, `wait`, and restore never choose a new
  task identity.
- Task and project names must be nonempty printable strings before submission.
  Restored provider task/target identities and task/project fields must also be
  canonical without surrounding whitespace; provider status rejects control
  characters. No narrower undocumented vendor alphabet is assumed.
- A timeout neither cancels nor resubmits the provider task.
- Before the first SDK operation, a mode-0600 recovery bundle records the
  credential-free task name, exact matrix, matrix digest, mode, sample count,
  project, schema, and original submission timestamp.
- The checkpoint directory must be private and non-symlinked before license
  initialization. A complete temporary recovery bundle is synced and atomically
  published without replacement before the first task operation. Receipt reads,
  temporary creation, exclusive publication, rollback, and directory sync are
  anchored to an opened non-symlink parent descriptor and reject a changed
  visible parent binding.
- The client freezes the checkpoint directory device/inode before license
  initialization and rechecks it before recovery access and around every vendor
  checkpoint context. A replacement detected after a vendor call is an
  indeterminate failed attempt, not permission to resubmit the task.
- Existing recovery content is immutable and opened without following
  symlinks. Its schema and UTC timestamp are validated, provider task/target
  identities must remain absent until an approved mapping exists, and every SDK
  solve, poll, and result operation requires the in-memory receipt to match the
  authoritative bundle exactly. A conflict, corruption, missing bundle, public
  file, or generic restore mismatch fails before SDK task access.
- Every successful result has valid spin shape and domain, matched sample and
  energy counts, independently recomputed finite energies, and explicit
  `fallback_occurred=false`.
- Persisted attempt failures contain only a fixed category and an empty message;
  arbitrary vendor, parser, plugin, and dynamic exception text is discarded.
  Malformed receipt and matrix inputs likewise do not retain their original
  exception as a public cause.
- Credentials are in-memory, redacted, and non-serializable. Vendor exception
  text is discarded at credential and SDK-operation boundaries.

## Plugin sampler semantics

- `KaiwuSampler.solve(matrix)` waits with an explicit timeout and returns an
  `int8` NumPy spin array compatible with the plugin.
- Identical matrices are deduplicated in-process. A timed-out matrix retains
  its job and a later call waits on that job.
- Each unique matrix consumes one declared remote-call budget slot; exhaustion
  fails before submission.
- Precision conversion occurs only when the caller supplies an explicit target
  range. Reports remain available for every distinct original matrix even when
  quantization allows remote-task deduplication. No classical or local solver
  is selected on error.
- Receipts, the last task, last result, precision report, and remote-call count
  remain inspectable for joined QDiffusion evidence.

## Dependency and compatibility impact

- Additive provider-specific submodules only. No existing symbol, signature,
  schema, root facade, runtime route, circuit behavior, or default changes.
- Importing `flagquantum` remains independent of Kaiwu, NumPy as a direct
  dependency, credentials, licenses, checkpoints, and network access.
- The proprietary SDK import is lazy and currently pinned to Python 3.10 and an
  exact caller-selected SDK version. The initial QDiffusion lane freezes Python
  3.10, Torch 2.7.0, NumPy 2.2.6, and Kaiwu 1.3.1 separately from the normal
  FlagQuantum environment.
- Before resolving credentials, the pinned client verifies the same imported
  module exposes `license.init`, `CheckpointManager.save_dir`, and callable
  `cim.CIMOptimizer`. That preflighted module instance is the one whose license
  is initialized and later used for task operations.
- FlagQuantum keeps provider-neutral task modes as `optimization` and
  `sampling`, while the pinned 1.3.1 adapter passes the version-documented
  `quota` and `sample` values to `CIMOptimizer`. The newer 1.4.1 vocabulary is
  not detected or guessed at runtime; it requires its own pinned adapter and
  conformance lane. The reviewed contract is the official
  [Kaiwu 1.3.1 CIM API](https://kaiwu-sdk-docs.qboson.com/zh/v1.3.1/source/modules/kaiwu.cim.html),
  not the moving `latest` documentation.
- The pinned client and every live launcher reject any SDK version other than
  1.3.1 before license initialization or credential resolution. A caller cannot
  select 1.4.1 while retaining 1.3.1 task-mode semantics.
- Provider launchers also require a strict SDK-rights approval object before
  credential resolution. It fixes the reviewed wheel source, digest, public
  service-terms identity, permitted organizational/container/host-staging/
  adapter-distribution uses, and `no-sdk-redistribution` boundary. The approval
  is independently bound to the exact Kaiwu distribution in the verified
  environment lock; recording public availability alone cannot authorize use.
- No dependency declaration or extra is proposed until wheel source, license,
  hashes, supported platform, and redistribution constraints are reviewed.

## Evidence and maturity

Development evidence is not hardware support. Promotion requires all of:

1. a pinned SDK wheel and license provenance;
2. an authenticated read-only capability probe;
3. reviewed mappings for provider task ID, target, raw state, failure, and
   result fields;
4. one real optimization and one real sampling task with no fallback;
5. independent bounded A800/QBoson system runs on `jp-a800-171` and
   `jp-a800-172`;
6. the preregistered protein experiment on one primary host and fixed guided
   replay on the second host;
7. a passing frozen two-host evidence manifest; and
8. an explicit capability-maturity review.

Until those gates pass, generated capability data must not claim QBoson,
QDiffusion, domestic accelerator, multi-node, distributed, production,
performance, quantum-advantage, or differentiable-quantum support.

## Explicit non-goals

- No gate-model QPU abstraction or execution through `fq.run`.
- No interpretation of CIM samples as circuit shots or `ExecutionResult`.
- No differentiation through remote sampling.
- No automatic precision reduction, simulator fallback, retry under a new task
  name, target-product inference, batch fan-out, or unbounded per-state calls.
- No domestic-accelerator claim; the current hosts contain NVIDIA A800 GPUs.
- No multi-node or distributed claim from two independent host runs.
- No public problem/result model in `algorithms.optimization` before a separate
  vendor-neutral proposal demonstrates demand beyond this provider.

## Open review questions

- Does the approved Kaiwu 1.3.1 wheel expose a stable provider task ID and
  provider-reported target in `get_task_result`, checkpoint content, or another
  documented API?
- Which exact raw states are terminal, retryable, and permanently failed?
- Does license initialization or SDK telemetry impose additional persistence,
  privacy, redistribution, or production constraints?
- Does the assigned project expose a named physical product or only an SPQC
  resource label?
- Is provider-supported batch submission available, or must QDiffusion remain
  under deduplication plus a hard call budget?

## Validation plan

- Pure exhaustive Ising/QUBO energy and gauge tests.
- Kaiwu Community conversion and energy conformance at a full source revision.
- Fake-client lifecycle, malformed-result, timeout, recovery, permission,
  identity-conflict, redaction, and no-resubmission tests.
- Direct Kaiwu PyTorch Plugin `BoltzmannMachine.condition_sample` replacement
  test at a full plugin revision.
- Pinned Python 3.10/Torch 2.7/Kaiwu 1.3.1 local and SDK conformance lane.
- Real provider optimization and sampling smoke tests guarded by explicit
  credentials and quota.
- Independent A800 system evidence from both declared hosts and frozen
  QDiffusion application evidence validated by the checked-in fail-closed gate.
- Full ecosystem and remote team tests, Black, Ruff, strict mypy, architecture,
  generated-contract, repository-hygiene, and capability-maturity checks.
