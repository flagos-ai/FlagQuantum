# QBoson Kaiwu integration and QDiffusion acceptance plan

## Implementation status

Status as of 2026-10-05 on branch `feat/qboson-kaiwu-integration`:

- API Change Proposal 068 documents the provider-specific Ising, remote-task,
  sampler, dependency, evidence, and maturity boundaries. Its status is draft;
  no root facade, dependency extra, or capability entry is authorized by the
  prototype.

- Phase 0 host reconnaissance is complete for the currently available compute
  surface. Both `jp-a800-171` and `jp-a800-172` expose eight NVIDIA
  A800-SXM4-80GB devices, driver 580.126.20, and system Python 3.10.12. Neither
  system Python environment currently contains Torch, Kaiwu, or the Kaiwu
  PyTorch plugin. Docker is available on both hosts.
- No QBoson credentials or quota have been provided. The authenticated
  capability probe and all live submissions therefore remain blocked by that
  explicit prerequisite. Kaiwu authentication uses `user_id` and `sdk_code` to
  initialize a local license; `sdk_code` must be handled as a secret.
- Phase 1 pure-data work is in progress under `flagquantum/ecosystem/kaiwu`.
  Ising validation, independent Kaiwu-convention energy evaluation, symmetric
  QUBO-to-Ising encoding with auxiliary-spin decoding, and an explicit integer
  precision policy have focused tests. Optional conformance tests match Kaiwu
  Community 1.0.7 at revision
  `b648b531c034bd6ae9b7a34fed994c717967cc72` for energy and QUBO conversion.
- The first Phase 2 security boundary is implemented under
  `flagquantum/remote/kaiwu`: credentials are explicit, redacted, and
  non-serializable, and environment discovery requires the complete dedicated
  `QBOSON_USER_ID` plus `QBOSON_SDK_CODE` pair. No vendor SDK import or network
  operation occurs during credential resolution.
- A Kaiwu-specific experimental task lifecycle now covers single submission,
  normalized status, fail-closed result validation, bounded waiting, private
  receipt persistence, and restore without resubmission. Its in-memory fake
  proves timeout and recovery semantics, spin and energy validation, matrix
  identity, sample-count limits, and explicit `fallback_occurred=false`. It is
  deliberately not exported from `flagquantum.remote`. A pinned adapter now
  implements only documented SDK behavior; real provider task-ID, target, and
  raw-state mappings remain unavailable until an approved SDK response can be
  inspected.
- The synchronous ecosystem sampler now matches the
  `kaiwu-pytorch-plugin` `solve(ising_matrix)` surface, delegates every unique
  matrix to the Remote lifecycle, deduplicates identical matrices, enforces a
  hard remote-call budget, and exposes receipts and the last result. Integer
  scaling remains explicit. A source-level conformance test passes through
  `BoltzmannMachine.condition_sample()` at plugin revision
  `f047bce7b1077449967bbe9e9fab5741542b48d4`; this is interface evidence only,
  not the pinned Python 3.10/Torch 2.7 compatibility or A800 acceptance lane.
- Pulling a Python 3.10 container from Docker Hub on `jp-a800-171` timed out.
  This is an environment provisioning constraint, not evidence of an SDK or
  FlagQuantum defect. A pinned Python 3.10/Kaiwu environment must be supplied
  or made reachable before proprietary SDK conformance and A800 execution.
- The existing `flagquantum/flagtree:0.7.0-validation` image on both A800 hosts
  observes GPU 0 as `NVIDIA A800-SXM4-80GB`, but contains Python 3.12.3 and
  Torch 2.13.0+cu129. A committed development probe can exercise the plugin,
  bounded sampler, explicit fake transport, backward pass, and parameter
  update in that image, but it must remain classified as preliminary evidence
  because it neither uses the declared Python 3.10/Torch 2.7 lane nor QBoson
  hardware.
- The documented `kaiwu==1.3.1` package was not available from the configured
  public package index during a download-only probe. The proprietary wheel or
  an approved platform download is still required to inspect and implement its
  provider-state mapping without guessing.
- The QDiffusion acceptance decision is now executable rather than narrative:
  a frozen experiment-config template, two-host manifest template, and
  fail-closed validator recompute system and application gates. Fake transport,
  CPU execution, reused task identities, fallback, resubmission, over-budget
  calls, changed software or precision lanes, missing seeds, and post-hoc metric
  failures are rejected. No real acceptance record has been produced yet.
- The proprietary SDK initialization boundary now fails before credential use
  unless Python is 3.10 and the installed Kaiwu version exactly matches the
  selected lane. SDK import is lazy, `license.init` receives the in-memory pair,
  and vendor exception text is discarded to prevent credential leakage. This
  is fake-module contract evidence; no license has been initialized yet.
- A pinned Kaiwu 1.3.1 client now implements the documented checkpoint model:
  submission calls `solve` once, later polls use the same task-name and matrix
  identity, restoration recreates that identity, and completed spins are
  independently scored. SDK globals are scoped and restored, and malformed or
  leaking vendor failures fail closed. Because documented APIs do not expose a
  stable provider task ID or provider-reported target, the client records those
  evidence gaps and cannot yet satisfy hardware acceptance.
- The 1.3.1 documentation declares `get_task_result(ising_matrix) -> dict` but
  does not document that dictionary's fields; the current 1.4.1 documentation
  still does not define provider task-ID or target keys. The pinned client now
  records a value-free schema of the returned dictionary (field names, value
  types, lengths, dtypes, and shapes) after completion. Raw values are never
  retained, and this diagnostic structure does not automatically map or accept
  a provider identity.
- Result-schema diagnostics now bound field counts, field-name length and
  syntax, dimensions, dtype names, and sequence-type inspection. Unsafe names
  are omitted, and inspection failure remains redacted and nonfatal to valid
  samples rather than causing unbounded or value-bearing evidence output.
- Before its first SDK operation, that client now atomically persists a
  mode-0600, credential-free recovery bundle in the Kaiwu checkpoint directory.
  This closes the ambiguous-submission window: a process restart reuses the
  original timestamp and task/matrix identity, while corrupt or conflicting
  bundles fail before any SDK operation.
- Checkpoint directories now fail before license initialization unless they are
  private, regular directories. Recovery and explicit job receipts are synced
  before atomic no-overwrite publication and reopened without following
  symlinks; public, non-regular, partial, or replaced files fail before an SDK
  task operation.
- Pinned 1.3.1 recovery also enforces an exact top-level schema, an aware UTC
  submission timestamp, and absent provider task/target identities. Local
  receipt editing therefore cannot fabricate the provider evidence that the
  documented SDK mapping does not expose.
- Every SDK solve, poll, and result operation reopens the deterministic
  recovery bundle and requires the in-memory receipt to match it exactly. The
  generic job-restore API cannot bypass checkpoint validation, and a separately
  saved receipt cannot replace missing authoritative SDK state.
- A cross-layer integration test now composes the ecosystem sampler, Remote
  lifecycle, pinned SDK client, checkpoint scoping, independent energy
  validation, recovery persistence, and deduplication. Only the final vendor
  module is replaced with a deterministic fake, so this proves the FlagQuantum
  boundary composition but is not provider or hardware evidence.
- A separately invoked Phase 2 live-smoke command is prepared for the approved
  Python 3.10 SDK lane. It requires an exact quota-cost acknowledgement before
  submitting one optimization and one sampling task, has no local fallback,
  writes a private credential-free record, and keeps hardware acceptance closed
  when provider task or target identity is unavailable.
- That smoke command now passes an explicit in-memory credential object after
  offline environment verification and scans both resolved credential values
  against the complete serialized evidence before creating its output file.
- Phase 2 smoke evidence now binds acceptance to an explicit real-SDK transport
  marker. Injected clients remain test evidence even when they return plausible
  provider identities, and the live command writes its diagnostic record but
  exits nonzero whenever hardware acceptance remains closed.
- A read-only host recheck confirmed that the SSH validation aliases differ
  from the machine-reported hostnames. The A800 development probe now records
  and verifies both identities separately, requires full source revisions, and
  writes an exclusive mode-0600 evidence file.
- The existing validation image was rechecked in ephemeral network-disabled
  containers on both hosts: each exposes exactly one A800 to Torch, but the
  image contains Python 3.12.3, Torch 2.13.0+cu129, and no Kaiwu package. A
  bounded runner now injects the observed host identity and mounts FlagQuantum
  and the public plugin source read-only for development-only execution.
- A subsequent read-only host recheck confirmed eight A800-SXM4-80GB devices,
  driver 580.126.20, Python 3.10.12, Docker 29.1.3, and no system Torch or Kaiwu
  package on both hosts. It also found that the same validation-image tag maps
  to image ID `sha256:3aea769f...` on `jp-a800-171` and
  `sha256:fd2afb63...` on `jp-a800-172`. The development runner therefore now
  requires and records the full per-host image ID and refuses mutable tags.
- Read-only filesystem reconnaissance found no transferred QBoson integration
  archive, pinned Kaiwu PyTorch Plugin checkout, Kaiwu 1.3.1 wheel, or isolated
  Kaiwu installation on either validation host. The existing checkout on
  `jp-a800-172` is revision `e07a642de9e10fc948c4c131210a243ec13b7c01`
  and does not contain this integration; the corresponding directory on
  `jp-a800-171` is not a Git checkout. No source was uploaded or remote file
  modified during this check.
- The bounded container runner now executes the pinned plugin's actual
  QDiffusion proposal, conditioned Boltzmann energy, FlagQuantum sampler,
  objective, backward, optimizer update, and one-step guided generation path.
  Its transport remains an explicit in-memory fake and its schema hard-codes
  `system_acceptance=false`; this is a rehearsal for, not evidence of, Phase 4.
- A separate quota-guarded live-system command now composes that QDiffusion
  slice with `KaiwuSDKClient`. It binds execution to the preregistered config
  hash and exact software lane, persists attempted receipts, checks repeat
  retrieval without a new sampler submission, scans output for both credential
  values, and fails the system gate when provider task or target IDs are absent.
- System-component acceptance now independently reconciles its exact SDK-client
  provenance, remote-call count, sampling receipts, matrix identities, requested
  sample counts, provider task IDs, and single provider target. Final summaries
  cannot pass by retaining plausible task/target fields after their underlying
  receipt set is removed or changed.
- Precision evidence now covers every distinct original Ising matrix rather
  than only the last plugin call. Quantization-equivalent inputs may share one
  remote task while retaining separate error reports; the acceptance validator
  rejects missing reports, invalid scale ranges, and negative or inconsistent
  aggregate errors.
- The Phase 5 runbook now covers frozen inputs, approved source transfer,
  development rehearsal, SDK-lane verification, quota-guarded smoke and system
  execution, interruption and same-identity resume, independent two-host runs,
  the primary protein experiment, final manifest validation, failure
  classification, and claim boundaries.
- Source audit of the pinned plugin confirms that its referenced human-proteome
  FASTA, DPLM weights, trained guided checkpoint, tokenizer snapshot, and ESM2
  weights are not in the Git tree. The acceptance config now freezes tokenizer
  and ESM2 content hashes plus training settings, and both host records must
  share the trained primary-host energy-checkpoint digest.
- An offline protein-artifact preflight now verifies exact file/tree identities,
  rejects symlinks and implicit path ambiguity, and emits a private record that
  cannot be mistaken for acceptance evidence.
- The frozen experiment now covers the plugin's corpus filtering, deterministic
  split, generation/resampling, and ESM2 evaluation knobs. FASTA preflight also
  proves that 640 eligible frozen records yield the declared 32-sequence test
  set before any A800 or QBoson quota is consumed.
- A QDiffusion builder binding now forces the pinned plugin's validation,
  training, baseline, and guided-generation branches to share the bounded
  FlagQuantum sampler. It rejects competing sampler injection and verifies the
  constructed energy model retained the same sampler object.
- A quota-guarded primary-host launcher now maps the frozen config into the
  pinned plugin's complete protein workflow one seed at a time, performs input
  preflight before credential resolution, forces local-only Transformers model
  loading, persists task receipts, and hashes the best trained energy
  checkpoint. Its record explicitly leaves both acceptance gates unevaluated.
- Protein-training provenance is now derived from the sampler's bound Remote
  client rather than a hard-coded transport label. Final assembly and
  independent revalidation require the exact SDK client, real-provider and
  QBoson-use flags, nonempty sampling receipts with provider identities,
  positive in-budget call counts, no fallback, and complete precision evidence
  for every frozen seed.
- System-probe and protein-training call budgets are now separate. A validator
  derives a conservative per-seed submission bound from the pinned plugin's
  actual positive/negative energy and generation loops; the illustrative full
  config requires up to 71,269 submissions per seed, so its protein budget stays
  unresolved until experiment size and QBoson quota are explicitly approved.
- Completed training records now hash the exact held-out, baseline, guided,
  history, and sequence-quality artifacts. A local-only ESM2 evaluator verifies
  that chain, loads one frozen checkpoint file without implicit download, checks
  aligned sequence identities, and computes candidate cosine/L2 evidence on the
  primary A800 without spending further QBoson quota.
- The shared training-record loader used by both ESM2 evaluation and replay-host
  execution now revalidates real SDK transport, QBoson-use and identity flags,
  sampling receipts, call budget, no-fallback state, and precision completeness.
  Injected or incomplete training evidence therefore fails before either A800
  evaluation work or another quota-consuming replay can begin.
- The ESM2 evaluator now independently reloads the primary host's private
  post-extraction preflight and requires both its record digest and shared
  transfer-manifest digest to match the training record before loading the
  evaluation workflow.
- Every per-seed evaluation component is now independently checked against the
  frozen source, plugin, Python, Torch, environment-lock and ESM2 identities,
  the primary host and observed A800 device, finite metric domains, zero invalid
  sequences, secret redaction, and zero provider-quota use before aggregation.
  A linked training digest alone can no longer legitimize foreign or fabricated
  evaluation metrics.
- Post-extraction content-set digests are now path-order normalized and
  execution-time recomputable. Protein training, evaluation, and replay bind
  the actual plugin-root name, file count, regular-file set, and content digest
  to the host preflight before importing the plugin workflow.
- Development, system, protein training, evaluation, and replay now also
  recompute the executing FlagQuantum source root against the host preflight;
  system and development paths bind their actual plugin root as well.
- The bounded system slice, protein training, and portability replay now
  resolve `kaiwu.torch_plugin` and its QDiffusion module from that reviewed root
  and fail closed if an installed or preloaded module comes from a different
  source tree.
- Training, evaluation, and replay also reject any loaded `dplm.*` module, and
  every bounded path rejects any `kaiwu.torch_plugin.*` module, whose source or
  namespace path escapes the reviewed plugin tree.
- An offline environment-lock verifier now requires the exact Python patch
  version and a complete, sorted installed-distribution inventory with exact
  versions and reviewed artifact digests. It rejects missing, extra, duplicate,
  placeholder, public, or symlinked inputs without installing anything.
- An offline lock builder now derives package names and versions from bounded
  wheel METADATA, hashes the reviewed wheel bytes, and requires an exact
  one-wheel-per-installed-distribution inventory. It never installs or executes
  an artifact and does not substitute for source, license, or terms approval.
- The lock also binds every distribution's installed RECORD file identities,
  sizes, and contents. Runtime verification recomputes those digests, so a
  modified Python or binary package file fails even when version metadata is
  unchanged.
- System, training, evaluation, and replay verify that exact runtime inventory
  and bind its lock digest to the frozen config before credentials are resolved.
  Final assembly copies and independently revalidates the lock as a closed-world
  evidence member.
- A quota-guarded replay-host runner now verifies and loads the exact selected
  primary-host checkpoint, rebuilds the DPLM model from frozen local artifacts,
  executes one preregistered held-out fixture through a fresh FlagQuantum remote
  sampler, checks repeat retrieval, and records provider and precision evidence.
  This is a portability gate, not multi-node execution or a second training run.
- Portability evidence now carries an exact SDK-client provenance flag and is
  independently reconciled against its A800 observation, remote-call budget,
  receipt and matrix identities, requested/returned samples, provider task IDs
  and target, precision coverage, no-fallback state, repeat retrieval, and token
  constraints before it can be linked into final replay-host evidence.
- A final evidence assembler now requires both passing system components, one
  passing replay component, and paired training/evaluation components for every
  frozen seed. It recomputes cross-seed means, validates all hash links, copies
  the exact components into a private bundle, and invokes the fail-closed final
  validator; selective seed reporting and component replacement are rejected.
- A credential-free local golden-path script now verifies clean checkouts at the
  pinned Kaiwu Community and Kaiwu PyTorch Plugin revisions, clears provider
  credential variables, explicitly gates source-only conformance without
  declaring an unapproved dependency extra, and runs conversion, lifecycle,
  sampler, plugin, and live-probe contract tests without network access or
  provider quota. It is explicitly local conformance evidence rather than A800
  or QBoson evidence.
- Machine-checked draft API gates now keep Kaiwu out of the stable root,
  ecosystem parent, remote parent, and capability-maturity registry; normal
  FlagQuantum imports are also proven not to resolve the optional vendor
  package. The repository boundary inventory rejects Kaiwu imports from Core,
  Runtime, and Simulation, preserving the planned `ecosystem + remote` split.
- Sampler calls now retain explicit transfer accounting for the plugin-produced
  CPU NumPy Ising matrix, canonical CPU float64 tensor, submitted host tuple,
  returned CPU int8 NumPy samples, cache use, and result shape. System evidence
  also records the A800 device that originated the matrix and received the
  reconstructed samples; the final validator rejects missing or inconsistent
  A800-to-CPU and CPU-to-A800 boundary records.
- A pre-extraction transfer verifier now binds the three reviewed source
  archives to manifest hashes and revision-derived names, rejects traversal,
  links, duplicate names, and special tar members, and emits a private
  preflight-only record per target alias. It does not authorize or perform the
  source transfer itself.
- A deterministic transfer builder now requires clean Git checkouts, enforces
  the pinned plugin and Community revisions, creates a new private directory,
  archives the exact commits, hashes them into the manifest, and self-verifies
  for both target aliases. Verification rejects any unlisted colocated tarball,
  preventing an extra archive from bypassing manifest review.
- Every documented QDiffusion Python entry point is now invoked as a module
  from the reviewed checkout with the user-site directory disabled. A
  parameterized subprocess test starts all thirteen commands from an unrelated
  working directory with only the reviewed root on `PYTHONPATH`, preventing an
  older installed FlagQuantum from silently satisfying imports.
- Transfer verification now binds each archive's internal top-level directory
  to its declared revision. A separate post-extraction verifier compares every
  regular file and the exact file/directory set with the reviewed archives,
  rejecting mutation, omission, additions, symlinks, and special entries before
  the source trees are used on either validation host.
- System, training, evaluation, portability-replay, and development records now
  retain the host-specific post-extraction preflight digest and common transfer
  manifest digest. Final assembly requires both host preflight files, copies
  them and the exact shared transfer manifest as immutable components, and
  revalidates every execution record's link to the correct host preflight and
  that self-contained manifest.
- Final assembly now rejects non-private or symlinked inputs and builds in a
  private sibling staging directory. The requested evidence directory appears
  atomically only after the independent validator passes, preventing a failed
  run from leaving a misleading partial bundle at the declared output path.
- Revalidation now rejects a symlink anywhere in a manifest member path and
  requires the manifest, config, host records, and copied components to remain
  inaccessible to group and other users.
- Final evidence validation now applies a closed-world file-tree check: member
  paths must be unique normalized relative POSIX paths, and unlisted files are
  rejected instead of being silently ignored beside declared evidence.

This section is a progress ledger, not a maturity or hardware-support claim.

## Decision and intended outcome

FlagQuantum should integrate QBoson's coherent optical quantum computers through
the Kaiwu SDK as an optional Ising and QUBO optimization and sampling provider.
The integration should not present a coherent Ising machine as a gate-model QPU
and should not route a general `fq.Circuit` through `fq.run` to Kaiwu. The first
complete product slice should run the PyTorch and DPLM portions of a workload on
the NVIDIA A800 systems `jp-a800-171` and `jp-a800-172`, use QBoson hardware for
Boltzmann energy-model sampling, and demonstrate that path with QDiffusion.

The architecture decision is to split the implementation from the beginning:

- `ecosystem/kaiwu` owns conversion between FlagQuantum-owned values and Kaiwu
  SDK objects, optional-dependency discovery, and compatibility with the sampler
  protocol expected by `kaiwu-pytorch-plugin`;
- `remote/kaiwu` owns credentials, task submission, polling, resumption, result
  retrieval, and provider evidence;
- `algorithms/optimization` may own vendor-neutral Ising and QUBO problem and
  result types after an approved API proposal establishes that these are not a
  second circuit IR;
- `services` may compose the A800 and remote-sampling steps only
  after the lower-level boundaries are proven independently.

QDiffusion is the final acceptance workload, not the source of the provider
contract. The provider must first pass small deterministic conformance tests so
that a QDiffusion result cannot hide an encoding, sign, precision, or task
lifecycle defect.

## Current technical basis

Kaiwu Community exposes `IsingSolver` and `QuboSolver` abstractions. Its solver
contract accepts an Ising matrix and returns spin solutions, while
`CIMOptimizer` submits those matrices to a QBoson special-purpose quantum
computer in optimization or sampling mode. The documented hardware path is
therefore an Ising and QUBO service, not arbitrary quantum-circuit execution.
See the [Kaiwu solver implementation](https://github.com/qboson/kaiwu_community/blob/main/src/kaiwu/core/_base_solver.py)
and the [Kaiwu CIM API](https://kaiwu-sdk-docs.qboson.com/zh/latest/source/modules/kaiwu.cim.html).

QBoson currently publishes `CQ-D-100`, `CQ-D-550`, `CQ-D-1000`, and
`Yuliang Shanhai 1000` product names. A cloud allocation may expose only an
SPQC capacity label rather than the physical product identity, so FlagQuantum
must record the provider-reported target and must not infer a product model from
the problem size. The current product specifications are published on the
[QBoson product comparison page](https://www.qboson.com/product/productSpecs).

FlagQuantum already separates external-object translation in `ecosystem` from
external task control in `remote`. Its architecture requires optional
integrations to remain outside the mandatory local PyTorch path and requires
backend changes to preserve program and result semantics. This integration must
follow those boundaries rather than adding Kaiwu imports to Core, Runtime, or the
root facade.

The QDiffusion implementation in `kaiwu-pytorch-plugin` combines a proposal
model with a conditioned energy model. Its energy model can contain a Boltzmann
machine and a sampler; hidden-state sampling converts each conditioned state to
an Ising matrix and calls `sampler.solve(...)`. Its protein example uses DPLM
backbones and compares baseline and energy-guided generation. See the
[QDiffusion source](https://github.com/qboson/kaiwu-pytorch-plugin/blob/main/src/kaiwu/torch_plugin/qdiffusion.py)
and the [QDiffusion example workflow](https://github.com/qboson/kaiwu-pytorch-plugin/blob/main/example/qdiffusion/README.md).

## Scope

The first supported slice includes:

- dense, symmetric, finite real Ising matrices with an explicit coefficient
  convention;
- Kaiwu optimization and sampling modes;
- local fake and Kaiwu simulated-annealing implementations for conformance;
- nonblocking QBoson submission, persistent task identity, polling, timeout,
  resumption, and result retrieval;
- explicit coefficient scaling and precision-reduction evidence;
- a synchronous sampler compatibility wrapper for bounded QDiffusion calls;
- A800 execution of proposal, feature-encoding, energy, loss, backward, and
  optimizer operations on `jp-a800-171` and `jp-a800-172`;
- QDiffusion system and application acceptance evidence.

The first slice does not include:

- general `fq.Circuit` lowering to a coherent Ising machine;
- treating a CIM sample set as `fq.ExecutionResult.expectation()`;
- a claim that CIM samples are differentiable through PyTorch autograd;
- automatic fallback from QBoson hardware to a classical solver;
- automatic selection of a named QBoson product from matrix width;
- multi-QPU fan-out, real-time hardware feedback, or a scalability claim;
- domestic-accelerator compatibility or performance validation;
- multi-node execution across the two A800 hosts;
- a stable root-level `fq.optimize` API before approval of its API contract.

## Validation compute environment

The first validation campaign uses the NVIDIA A800 hosts `jp-a800-171` and
`jp-a800-172`. Each host must execute the bounded hybrid system acceptance path
independently on one observed A800 device. Using both hosts establishes that the
integration can be reproduced in the two available environments; it does not
establish multi-node execution, distributed scalability, or domestic-accelerator
support.

The full protein effectiveness experiment may run on one declared primary host
to control cost and provider quota. The other host must repeat the bounded
system acceptance path and one fixed guided-inference fixture from the primary
run. Every artifact must name its execution host, CUDA device, GPU model, and
whether the run was the primary experiment or the portability replay.

## Architecture and data flow

```text
PyTorch model and DPLM backbone on an A800 validation host
                        |
                        | conditioned BM coefficients
                        v
       FlagQuantum optimization domain or bounded adapter contract
                        |
                        v
             flagquantum.ecosystem.kaiwu
       validation, encoding, scaling, SDK translation
                        |
                        v
               flagquantum.remote.kaiwu
      submit, poll, resume, retrieve, provider evidence
                        |
                        v
             Kaiwu SDK and QBoson SPQC service
                        |
                        | spins, energies, task receipt
                        v
       result decoding and QDiffusion energy computation
                        |
                        v
       loss, backward, optimizer, and guided generation
                 on the same A800 host
```

This is heterogeneous hybrid computation coordinated by FlagQuantum. It is not
one transparent device and does not imply that PyTorch autograd differentiates
through the optical quantum computer.

## Proposed package ownership

```text
flagquantum/
├── ecosystem/
│   └── kaiwu/
│       ├── __init__.py
│       ├── conversion.py
│       ├── dependency.py
│       ├── precision.py
│       └── sampler.py
├── remote/
│   └── kaiwu/
│       ├── __init__.py
│       ├── credentials.py
│       ├── provider.py
│       ├── task.py
│       └── result.py
├── algorithms/
│   └── optimization/
│       ├── problems.py
│       └── results.py
└── services/
    └── hybrid_sampling.py

tests/
├── unit/ecosystem/kaiwu/
├── unit/remote/kaiwu/
├── integration/kaiwu/
└── hardware/kaiwu/

examples/
└── qdiffusion_kaiwu/
```

Only the directories needed for the current phase should be added. In
particular, `services/hybrid_sampling.py` should not be created until two lower
layers have a complete input, validation, execution, result, failure, and
evidence path. New public problem or result types require an API change proposal
and must not be introduced as an unreviewed parallel contract.

## Boundary contracts

### Ising input

The adapter must define and test all of the following before live submission:

- whether the energy is `s^T J s`, `-s^T J s`, or another normalized form;
- whether the diagonal carries linear bias or must be represented by an
  auxiliary spin;
- whether one triangle or both symmetric entries contribute to the energy;
- the accepted spin domain, which is `{-1, +1}` at the provider boundary;
- finite-value, symmetry, width, coefficient-range, and precision requirements;
- the exact inverse mapping from returned spins to any `{0, 1}` consumer state.

Conversion must be pure and testable without Kaiwu installed. Round-trip tests
must include hand-computable one-spin, two-spin, ferromagnetic,
antiferromagnetic, and linear-bias fixtures.

### Precision policy

Precision reduction may change the optimization landscape. The submission plan
and returned result must therefore record:

- original coefficient dtype and range;
- symmetry normalization;
- scale factor, rounding policy, and target integer range;
- maximum and mean quantization error;
- whether the best known fixture solution is preserved after reduction;
- the Kaiwu and provider target limits used for validation.

Out-of-range input must fail unless the caller explicitly authorizes a named
precision policy. Silent clipping or rescaling is prohibited.

### Remote task lifecycle

The provider contract must support `prepare`, `submit`, `status`, `result`, and
`resume`. A local timeout must leave the remote task identifiable and resumable.
Every live result must retain:

- provider name and provider-reported target identifier;
- project and task identifiers, excluding secrets;
- optimization or sampling mode;
- requested and returned sample counts;
- submission, queue, execution, and retrieval timestamps when supplied;
- input and precision-policy identities;
- SDK version and adapter version;
- provider status and terminal failure details;
- a declaration of whether any fallback occurred.

The first implementation must reuse the repository's existing remote-job and
result patterns where their semantics match. It must not create a second generic
job framework solely for Kaiwu.

### QDiffusion sampler compatibility

`kaiwu-pytorch-plugin` currently expects a sampler with a synchronous
`solve(ising_matrix)` method. The ecosystem wrapper may provide that narrow
surface, but it must delegate task ownership to `remote.kaiwu` and expose the
last receipt and evidence for inspection. Synchronous waiting must be bounded by
an explicit timeout.

Conditioned Boltzmann sampling currently calls the sampler once for each visible
state. A direct implementation could therefore create an unacceptable number of
remote jobs. Before application acceptance, the integration must do one of the
following:

1. implement a provider-supported batch submission path;
2. deduplicate identical conditioned matrices and reuse completed tasks;
3. constrain the acceptance workload to a declared remote-call budget.

The selected behavior and its call count must be recorded. Hidden per-sample
submission is not acceptable for the final QDiffusion run.

## Dependency and release policy

The Kaiwu dependency must be optional and lazily imported. Importing
`flagquantum` or running local CPU and accelerator tests must not require Kaiwu,
credentials, or network access.

The initial compatibility environment should be pinned as a separate test lane
because the current Kaiwu PyTorch plugin declares Python 3.10, Torch 2.7.0,
NumPy 2.2.6, and Kaiwu 1.3.1, while current Kaiwu documentation describes SDK
1.4.1. The project must validate the chosen SDK version instead of assuming
that 1.3.1 and 1.4.1 are interchangeable. FlagQuantum's broader Torch support
must not be narrowed for users who do not install the integration.

Before production support, record the Kaiwu package source, hashes or lockfile,
license, cloud-service terms, credential requirements, and an exit path that
allows the provider client to be replaced without modifying problem consumers.

## Delivery phases and gates

### Phase 0 contract and environment reconnaissance

Deliverables:

- an approved architecture or API proposal for any new cross-domain contract;
- a verified Kaiwu SDK environment and version matrix;
- one authenticated read-only capability probe against the assigned QBoson
  account;
- recorded provider target labels, matrix limits, coefficient limits, sampling
  limits, and task-state vocabulary;
- a decision on whether the live account exposes a product identity such as
  `CQ-D-550` or only a resource class such as `SPQC-550`.

Exit gate: no credentials or vendor objects enter Core, and every unknown live
capability is recorded as unknown rather than inferred.

### Phase 1 conversion and local conformance

Deliverables:

- pure Ising validation and conversion;
- explicit precision policy;
- fake solver and Kaiwu simulated-annealing conformance tests;
- energy recomputation independent of Kaiwu;
- replacement test proving that fake and Kaiwu local implementations can be
  exchanged without changing the consumer.

Exit gate: seeded fixtures return valid spins, independently recomputed energies
agree under the declared convention, and unsupported inputs fail before any
submission.

### Phase 2 remote QBoson provider

Deliverables:

- credentials and redaction;
- nonblocking submit, status, result, timeout, and resume behavior;
- persistent task receipts;
- hardware tests guarded by explicit credentials and quota selection;
- failure fixtures for authentication, quota, validation, timeout, provider
  failure, and malformed results.

Exit gate: one optimization task and one sampling task complete on the assigned
QBoson resource, retain provider evidence, and show `fallback_occurred=false`.

### Phase 3 A800 and sampler interoperability

Deliverables:

- a bounded synchronous sampler wrapper for `kaiwu-pytorch-plugin`;
- a small PyTorch energy-model test on one A800 device on each of
  `jp-a800-171` and `jp-a800-172`;
- explicit evidence for the execution host and requested and observed CUDA
  device;
- transfer accounting at A800-to-CPU and CPU-to-A800 boundaries;
- a remote-call budget or provider batch path.

Exit gate: on each validation host, one Boltzmann objective and optimizer update
use an observed NVIDIA A800 for tensor work and QBoson hardware for samples,
with neither execution path silently replaced. The two runs are independent and
do not constitute a distributed execution claim.

### Phase 4 QDiffusion acceptance

Phase 4 has two separate decisions: system acceptance and application
effectiveness. Both are required for the full QDiffusion milestone, but a failed
effectiveness experiment must not be misreported as a provider transport defect.

#### System acceptance

Use a bounded QDiffusion workload before the full protein experiment:

- one seeded, short sequence batch;
- one proposal forward pass on an observed A800 on each validation host;
- one conditioned BM sampling request completed by QBoson hardware;
- one finite `energy_objective`;
- one successful backward pass and optimizer update for trainable energy-side
  parameters;
- one guided generation step using returned samples;
- an enforced maximum remote-call count;
- a saved receipt linking each run to its host, CUDA device observation, and
  QBoson task identity.

The gate passes only when:

- the execution host is `jp-a800-171` or `jp-a800-172` and the observed tensor
  device is an NVIDIA A800;
- the QBoson task reaches a successful terminal state;
- returned spins have the validated domain, shape, sample count, and energy;
- the trainable parameter delta is finite and nonzero;
- generated token IDs obey the declared token and special-token constraints;
- no classical, CPU, or local-simulator fallback occurs;
- rerunning result retrieval does not resubmit the remote task;
- secrets are absent from logs, receipts, and saved evidence.

#### Application effectiveness

Run the DPLM protein workflow on one declared primary A800 host with a frozen
dataset split, checkpoint, tokenizer, generation configuration, sample count,
and seed set. Compare baseline DPLM generation with QBoson-guided QDiffusion
using the workflow's existing metrics:

- ESM2 mean and median cosine distance;
- ESM2 mean and median L2 distance;
- uniqueness;
- repeat ratio;
- sequence validity and length compliance;
- identity and Jensen-Shannon divergence where their definitions are fixed by
  the workflow version.

Before running the experiment, designate one primary metric and its direction.
The recommended primary metric is mean ESM2 cosine distance to the reference
set, where lower is better. Full application acceptance requires:

- the QBoson-guided result to improve the preregistered primary metric against
  the baseline over the declared seed aggregation;
- no invalid generated sequence;
- uniqueness no lower than 95 percent of the baseline value;
- repeat ratio no more than 0.05 absolute above the baseline value;
- complete evidence for all attempted seeds, including failed or timed-out
  provider tasks;
- no claim stronger than the number of seeds, sample size, hardware identity,
  and recorded environment support.

The upstream example's published sample result may be used as a qualitative
reference, not as FlagQuantum evidence or as a copied threshold. FlagQuantum
must generate its own baseline and guided results from the same frozen
environment. The second A800 host must replay one fixed guided-inference fixture
and satisfy the system gate; it need not repeat the full protein experiment.

### Phase 5 documentation and maturity decision

Deliverables:

- a ten-minute local golden path requiring no credentials;
- a separately invoked live-hardware example with explicit cost and quota
  warnings;
- a QDiffusion runbook covering submit, interrupt, resume, and evidence
  collection;
- generated capability and known-limitation entries only after the corresponding
  evidence is accepted;
- an explicit maturity decision under the repository capability policy.

The live result initially establishes development evidence only. It does not
establish performance superiority, production support, quantum advantage, or
distributed scalability.

## Verification matrix

| Layer | Test resource | Required proof |
| --- | --- | --- |
| Conversion | No optional dependencies | Spin encoding, symmetry, sign, bias, scale, round trip, and independent energy |
| Provider contract | In-memory fake | Lifecycle, idempotent retrieval, timeout, resume, redaction, and malformed result handling |
| Kaiwu local | Kaiwu classical solver | Replaceable implementation and semantic parity on seeded fixtures |
| QBoson smoke | Assigned SPQC quota | Real task identity, returned samples, independent energy, and no fallback |
| A800 compute | `jp-a800-171` and `jp-a800-172` | Independent observed A800 device, finite forward and backward values, and parameter update on each host |
| Hybrid QDiffusion | A800 and SPQC | One bounded end-to-end training and generation slice on each host with joined evidence and no distributed claim |
| Protein evaluation | Frozen DPLM experiment on one primary A800 host | Baseline-versus-guided metrics under the preregistered acceptance rule, plus one fixed guided-inference replay on the second host |

## Evidence artifact

Produce one QDiffusion evidence record per validation-host run and one manifest
that joins the two record identities without merging their measurements. Each
host record should contain at least:

```json
{
  "schema": "flagquantum.qboson_qdiffusion_acceptance",
  "version": "1.0",
  "source_revision": "<full Git revision>",
  "flagquantum_version": "<version>",
  "kaiwu_sdk_version": "<version>",
  "kaiwu_pytorch_plugin_revision": "<full Git revision>",
  "python_version": "<version>",
  "torch_version": "<version>",
  "execution_host": "jp-a800-171|jp-a800-172",
  "run_role": "primary|portability_replay",
  "requested_cuda_device": "<device>",
  "observed_cuda_device": "<device>",
  "observed_gpu_model": "NVIDIA A800",
  "qboson_target": "<provider-reported target>",
  "qboson_task_ids": ["<redacted-safe task id>"],
  "sampling_mode": "sampling",
  "requested_samples": 0,
  "returned_samples": 0,
  "precision_policy": {},
  "remote_call_count": 0,
  "fallback_occurred": false,
  "training": {},
  "generation": {},
  "baseline_metrics": {},
  "guided_metrics": {},
  "acceptance": {
    "system": "pass|fail",
    "application": "pass|fail"
  },
  "limitations": []
}
```

The exact schema must be reviewed before implementation. This illustration does
not authorize a serialized public contract or a new capability claim.

## Principal risks and mitigations

| Risk | Consequence | Required mitigation |
| --- | --- | --- |
| Gate-model and CIM semantics are conflated | Incorrect public API and misleading results | Keep Kaiwu on an explicit optimization and sampling path; reject general circuits |
| Ising sign or auxiliary-spin convention differs | Apparently valid but wrong solutions | Independent energy fixtures and round-trip tests before live submission |
| Precision reduction changes the optimum | Scientific result is not comparable | Explicit policy, error report, and solution-preservation fixtures |
| QDiffusion submits once per conditioned state | Excess quota, latency, and irreproducible completion | Batch, deduplicate, cache, or enforce a declared call budget |
| Remote sampling is treated as differentiable | Invalid gradient claim | Attribute gradients only to the local energy objective and document the sampling boundary |
| A800 execution silently falls back to CPU | Hybrid claim is false | Record host, requested CUDA device, and observed GPU model and fail the acceptance gate on mismatch |
| SDK and plugin pins diverge | Installation or semantic breakage | Dedicated Python 3.10 and Torch 2.7 compatibility lane with an explicit SDK matrix |
| Cloud label is mistaken for physical product | Incorrect hardware claim | Record only provider-reported target identity; do not infer CQ-D or Shanhai model |
| Small stochastic run is overinterpreted | Unsupported effectiveness claim | Freeze the experiment, aggregate declared seeds, publish failures, and restrict the claim |

## Completion criteria

The integration is complete for its first declared scope only when all of the
following are true:

1. conversion, precision, task lifecycle, and result semantics have independent
   conformance evidence;
2. Kaiwu remains an optional dependency and local FlagQuantum behavior is
   unchanged when it is absent;
3. real QBoson optimization and sampling tasks have completed and are resumable
   from persisted task identities;
4. the QDiffusion system acceptance workload passes independently on
   `jp-a800-171` and `jp-a800-172`, using an observed NVIDIA A800 and QBoson
   hardware without fallback;
5. the frozen protein experiment satisfies the preregistered application
   effectiveness criteria, or is explicitly reported as a failed experiment
   without weakening those criteria after the result is known;
6. capability maturity and limitations reflect only the checked-in evidence;
7. no gate-model execution, differentiability, performance, scalability, or
   quantum-advantage claim exceeds the verified scope.
