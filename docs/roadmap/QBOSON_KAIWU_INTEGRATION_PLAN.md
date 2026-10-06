# QBoson Kaiwu integration and QDiffusion acceptance plan

## Implementation status

Status as of 2026-10-06 on branch `feat/qboson-kaiwu-integration`:

- API Change Proposal 068 documents the provider-specific Ising, remote-task,
  sampler, dependency, evidence, and maturity boundaries. Its status is draft;
  no root facade, dependency extra, or capability entry is authorized by the
  prototype.

- Phase 0 host reconnaissance is complete for the currently available compute
  surface. Both `jp-a800-171` and `jp-a800-172` expose eight NVIDIA
  A800-SXM4-80GB devices, driver 580.126.20, and system Python 3.10.12. Neither
  system Python environment currently contains Torch, Kaiwu, or the Kaiwu
  PyTorch plugin. Docker is available on both hosts.
- No QBoson credentials have been supplied to the integration environment, and
  no provider quota has been approved for a FlagQuantum run. The authenticated
  capability probe and all live submissions therefore remain blocked by those
  explicit prerequisites. Kaiwu authentication uses `user_id` and `sdk_code` to
  initialize a local license; `sdk_code` must be handled as a secret.
- On 2026-10-06 the authenticated platform beginner tutorial accepted the
  independently computed Max Cut result, marked the tutorial mastered, and
  displayed a notification that free real-machine credits had been issued. The
  platform subsequently returned inconsistent service and tutorial states. A
  later authenticated dashboard snapshot showed SPQC-1000 as available with
  zero sampling credits and one optimization credit, while the SDK page again
  labeled the beginner tutorial incomplete. The task table and 30-day totals
  both remained zero. The account Resource Bill then resolved the cross-target
  balance: two unused SPQC-550 optimization credits and one unused SPQC-1000
  optimization credit are available, while SPQC-1, SPQC-550, and SPQC-1000 all
  have zero sampling credits and zero sampling use. The API view reports zero
  API applications and zero calls. This is account-state reconnaissance, not
  approval to spend any displayed optimization credit, and it does not satisfy
  the required sampling quota or establish an SDK project assignment. No
  provider task was submitted.
- On 2026-10-06 a credential-free support request was submitted through the
  authenticated platform. It asks QBoson for the Kaiwu 1.3.1 CPython 3.10 Linux
  distribution and its digest and terms, an SDK-capable project number,
  sampling quota, and written decisions covering isolated-container use and
  open-source adapter publication without SDK-wheel redistribution. Until the
  account-bound response is retained and reviewed, the SDK approval, project,
  sampling-quota, and live-provider gates remain closed.
- The authenticated SDK page currently offers platform downloads for Kaiwu
  1.4.1 and retains a 1.3.1 release-note entry describing sample-mode task
  submission. It also shows that an account-bound SDK authorization code exists,
  but that code was kept masked and was not copied, revealed, or imported into
  the development environment. These observations do not substitute for the
  pinned 1.3.1 package, project assignment, credentials, or use approval.
  The Linux selector still exposes Python 3.10 `v1.3.1`; the `1.4.0` changelog
  records namespace-package, solver-base, task-mode, sampling/cache/upload, and
  `solve_qubo` interface changes. The acceptance contract therefore remains
  explicitly pinned to `1.3.1`. Reviewing or adopting `1.4.x` is a separate
  migration and must not happen through the download page's default selection.
- The versioned Kaiwu 1.3.1 API reference confirms the exact adapter contract:
  `CIMOptimizer` uses legacy task modes `"quota"` and `"sample"`, sampling
  accepts 10 through 2000 results, task identity is based on `task_name` plus
  the Ising matrix, and `project_no=None` is technically valid. The Remote
  adapter already maps the stable FlagQuantum vocabulary to those legacy modes,
  enforces the sampling range, omits `project_no` when absent, and persists the
  documented identity inputs. Final acceptance nevertheless continues to
  require a reviewed project assignment so provider work and quota have an
  accountable scope. That is an evidence policy, not a claim that the SDK
  constructor always requires a project number; an account-default assignment
  would need an explicit reviewed decision and contract update before use.
- The Phase 1 pure-data boundary is implemented under
  `flagquantum/ecosystem/kaiwu`. Ising validation, independent
  Kaiwu-convention energy evaluation, symmetric QUBO-to-Ising encoding with
  auxiliary-spin decoding, and an explicit integer precision policy have
  focused tests. Explicit symmetry tolerance is now a normalization rule: it
  decides whether an input may be accepted, but every returned matrix is
  exactly symmetric for the stricter Remote boundary. Boolean, complex,
  string, nonfinite, and negative scalar controls fail with owned boundary
  errors. Symmetry normalization avoids intermediate overflow, while
  QUBO conversion, independent energy evaluation, and precision reporting
  reject finite inputs whose derived values overflow. The effective integer
  magnitude is capped at `2^53`, preserving exact conversion back to float64
  for error evidence instead of allowing large-int saturation to look exact.
  Optional conformance tests match Kaiwu Community 1.0.7 at revision
  `b648b531c034bd6ae9b7a34fed994c717967cc72` for energy and QUBO conversion.
- The public Kaiwu Community tree does not contain a simulated-annealing
  optimizer; the replacement test therefore uses its real `IsingSolver`
  contract with an exhaustive implementation. Conformance against the
  provider-distributed local annealer remains part of the proprietary SDK lane
  and cannot be claimed from the public source test.
- The first Phase 2 security boundary is implemented under
  `flagquantum/remote/kaiwu`: credentials are explicit, redacted, and
  non-serializable, and environment discovery requires the complete dedicated
  `QBOSON_USER_ID` plus `QBOSON_SDK_CODE` pair. No vendor SDK import or network
  operation occurs during credential resolution. Explicit and environment
  values are trimmed and must remain nonempty and printable; embedded control
  or format characters fail without echoing either credential.
- Each quota-consuming CLI converts the resolved pair into an explicit
  non-serializable credential object and immediately removes
  `QBOSON_USER_ID`/`QBOSON_SDK_CODE` from its process environment before SDK
  client construction. Later plugin code and accidental child processes cannot
  inherit the secrets; the retained in-memory strings remain available only for
  recursive evidence-leak refusal.
- A Kaiwu-specific experimental task lifecycle now covers single submission,
  normalized status, fail-closed result validation, bounded waiting, private
  receipt persistence, and restore without resubmission. Its in-memory fake
  proves timeout and recovery semantics, spin and energy validation, matrix
  identity, sample-count limits, and explicit `fallback_occurred=false`. It is
  deliberately not exported from `flagquantum.remote`. A pinned adapter now
  implements only documented SDK behavior; real provider task-ID, target, and
  raw-state mappings remain unavailable until an approved SDK response can be
  inspected.
- Task names and optional project numbers must be nonempty printable text before
  sampler construction or Remote submission. Restored receipt task/project and
  provider task/target identities additionally require canonical surrounding
  whitespace, and provider status text rejects control characters. The final
  acceptance validator independently enforces the same receipt identity rule,
  preventing log or JSON injection without inventing an undocumented vendor
  character alphabet or length limit.
- All four quota-consuming CLI paths now share one project-number and
  task-prefix normalizer. They trim once and reject empty, control, or format
  characters before credential resolution or SDK construction, so an invalid
  orchestration identifier cannot unnecessarily expose credentials to the
  process or initialize the vendor lane before the owned input boundary fails.
- Generic receipt restoration now validates runtime types, exact matrix size
  and identity, mode-specific sample limits, nonempty project and provider
  identities, and an aware UTC submission timestamp before a client operation.
  Status and result validation also rejects non-string provider states,
  non-mapping metadata, non-tuple samples or energies, boolean spins, and
  non-real energies through stable FlagQuantum-owned errors rather than leaking
  incidental Python exceptions across the Remote boundary. Submission now also
  rejects non-receipt client responses and invalid runtime task modes through
  owned errors; wait and sampler controls reject boolean, string, complex, or
  nonfinite numeric inputs without leaking incidental type errors.
- The Remote matrix boundary now tests scalar numeric type before conversion,
  so NumPy complex values cannot silently discard their imaginary part and
  NumPy or Python strings cannot be accepted merely because `float(...)` can
  parse them. NumPy integer and floating scalars remain supported.
- Generic receipt save and restore now also reject a public, missing,
  non-directory, or symlinked parent before opening or publishing a file. A
  mode-0600 receipt inside a replaceable directory is no longer treated as
  authoritative recovery state. Receipt publication synchronizes both file
  contents and parent-directory metadata. Both publication and restore remain
  anchored to one `O_NOFOLLOW` parent descriptor and reject a changed visible
  directory binding; publication removes a candidate receipt if the parent is
  replaced before synchronization. Recovery also rejects duplicate JSON object
  keys at any nesting level rather than accepting an ambiguous last value.
  Receipt reads now keep the no-follow file descriptor open through parsing and
  recheck its complete stable metadata plus visible leaf binding afterwards;
  leaf replacement and non-standard `NaN`/`Infinity` constants fail before a
  recoverable job is constructed. Generic receipts and authoritative SDK
  recovery bundles are also capped at 64 MiB before JSON parsing, preventing an
  unbounded same-user file from exhausting memory at the recovery boundary
  while retaining capacity for a dense 1,000-spin matrix. Publication enforces
  the same serialized-size bound before temporary-file creation, so a successful
  save cannot create a receipt that the restore path is required to reject. The
  descriptor reader itself consumes at most one byte beyond that bound before
  refusing input, so concurrent file growth cannot cause an unbounded parse
  between the initial metadata check and the stable-file postflight. Private
  receipt parents, receipt leaves, and SDK checkpoint directories must belong
  to the current effective UID as well as having owner-only mode bits; a
  privileged process cannot trust another user's recovery state merely because
  it is mode `0700` or `0600`.
- The shared QDiffusion private-evidence boundary applies the same effective-UID
  ownership requirement to configuration, approval, source-preflight, resource,
  training, evaluation, and acceptance component records, including directory
  and leaf rechecks after descriptor-relative reads and during exclusive
  publication. Final acceptance assembly also freezes an output parent owned by
  that UID and rechecks ownership through validation and atomic publication.
- Independent final validation requires the manifest, every declared member,
  and every file and directory in the closed-world acceptance tree to belong to
  the current effective UID; ownership is checked before manifest parsing and
  repeated during member resolution and exact-tree validation.
- Protein-training output roots, generated run directories, checkpoints, and
  retained workflow artifacts must be owned by the current effective UID.
  Stable source and artifact snapshots now include UID in their metadata
  identity, so an ownership change after capture invalidates the evidence.
- Evaluation and portability replay independently enforce current-effective-UID
  ownership on the retained training artifacts and selected checkpoint before
  they consume those files; digest equality and owner-only mode are not enough.
- The streamed A800 runner extracts the outer input and all three reviewed
  archives with `--no-same-owner`. This normalizes tmpfs content to the
  container's effective UID instead of weakening the ownership gate for files
  whose workstation UID differs from the container UID.
- Before opening SSH, the stream gate now applies the shared effective-UID
  validation to both the local transfer directory and evidence-output parent;
  an unsafe local capture destination fails before an A800 run is consumed.
- The synchronous ecosystem sampler now matches the
  `kaiwu-pytorch-plugin` `solve(ising_matrix)` surface, delegates every unique
  matrix to the Remote lifecycle, deduplicates identical matrices, enforces a
  hard remote-call budget, and exposes receipts and the last result. Integer
  scaling remains explicit. A source-level conformance test passes through
  `BoltzmannMachine.condition_sample()` at plugin revision
  `f047bce7b1077449967bbe9e9fab5741542b48d4`; this is interface evidence only,
  not the pinned Python 3.10/Torch 2.7 compatibility or A800 acceptance lane.
- Each sampler instance now serializes the complete cache lookup, submission,
  wait, budget-accounting, and evidence transaction. Concurrent identical
  plugin calls therefore share one registered Remote job and cannot race past
  a one-call quota ceiling before either call records its receipt.
- The SDK client now also serializes the complete authoritative-receipt check,
  solution-cache lookup, provider operation, and cache publication for one task
  identity. Concurrent direct Remote status or result consumers therefore
  share the completed SDK operation instead of both entering `solve()` before
  the first thread publishes its result. Result-schema inspection uses the same
  process-wide SDK boundary.
- The sampler now reserves that quota slot before calling the Remote client and
  never returns it after an exception. A response failure after the provider
  may have observed the task can no longer leave the counter at zero and permit
  a different task to cross the declared hard ceiling.
- Pulling a Python 3.10 container from Docker Hub on `jp-a800-171` timed out.
  This is an environment provisioning constraint, not evidence of an SDK or
  FlagQuantum defect. A pinned Python 3.10/Kaiwu environment must be supplied
  or made reachable before proprietary SDK conformance and A800 execution.
- The existing `flagquantum/flagtree:0.7.0-validation` image on both A800 hosts
  observes GPU 0 as `NVIDIA A800-SXM4-80GB`, but contains Python 3.12.3 and
  Torch 2.13.0+cu129. A committed development probe can exercise the plugin,
  bounded sampler, explicit fake transport, backward pass, and parameter
  update in that image, but it must remain classified as preliminary evidence
  because it neither uses the frozen composite Python 3.10/Torch 2.7/Kaiwu
  1.3.1 lane nor QBoson hardware.
- A read-only 2026-10-06 inventory pass also examined eleven already-present,
  immutable image IDs across the two hosts, including the common
  `flagquantum/sc27-pennylane` and vLLM images plus the host-local `fq-mps`,
  chemistry, SDD, TD, and FlagQuantum compatibility images. Every isolated
  network-disabled probe reported Python 3.12.x with Torch 2.13.x or 2.14.x;
  none supplied the required Python 3.10/Torch 2.7 pair. The same-content common
  candidates were bound by full image IDs, while the identically named
  `fq-mps008-stock:3.7.1` tag resolved to different full IDs on the two hosts
  and was checked separately. This is environment-availability reconnaissance,
  not package approval or acceptance evidence, and it does not justify relaxing
  the frozen runtime contract.
- The earlier configured-index probe found no `kaiwu==1.3.1` package, but that
  external state has changed. PyPI now publishes the Qboson-owned CPython 3.10
  Linux wheel `kaiwu-1.3.1-cp310-none-manylinux1_x86_64.whl`, uploaded on
  2026-06-26 with SHA-256
  `7334cabd4ff0ae02e042d1c38ed292211573e83e2ed8e92fdf41af52e8991455`.
  Its PyPI metadata has no license identifier or license files. The wheel has
  now been downloaded into an owner-only temporary review directory and its
  bytes independently reproduce that published digest. Static ZIP inspection
  confirms `Requires-Python: >=3.10`, author `Qboson Inc`, the documented
  dependency declarations, CPython 3.10 Linux tagging, compiled extension
  modules, and no `LICENSE`, `COPYING`, or `NOTICE` member. It has not been
  installed, imported, or executed. This identity and metadata review is not a
  source audit, license approval, or permission to use or redistribute the
  package; the explicit package and service-terms review still precedes the
  pinned SDK lane.
- The same inspection is now reproducible through a bounded, no-import wheel
  inspector. Against the exact private review copy it records 62 archive
  members, 45 compiled extension modules, the nine declared dependencies, no
  license metadata, and no legal license-file member. Its mode-0600 evidence
  record has SHA-256
  `22fa0ac88da9aeed12a6d9668d8b7a23995874427aea2fb514379de242934249`
  and fixes execution, provider use, and redistribution approval to `false`.
  It is deliberately a separate schema from the SDK approval record and cannot
  open a credential or quota-consuming path.
- The service-terms source is no longer unknown. The Kaiwu 1.3.1 license page
  links to the public QBoson Quantum Cloud Platform User Service Agreement;
  the currently rendered agreement is effective 2026-07-09 and expressly
  covers Kaiwu SDK, Kaiwu-PyTorch-Plugin, remote APIs, and
  SPQC-1000/SPQC-550/SPQC-X computing services. It limits SDK/API use to the
  agreement, prohibits sale, transfer, or sublicensing without written
  permission, allocates installation/use risk to the user, and includes an
  own-use restriction plus platform-data and confidentiality provisions. This
  is contract-source evidence, not legal approval. Organizational development,
  isolated-container execution, adapter distribution, and any separate wheel
  license still require an explicit review decision before acquisition.
- The frozen QDiffusion config now turns that decision into a fail-closed gate.
  It binds the exact Kaiwu 1.3.1 wheel filename, source, SHA-256, agreement
  identity/effective date, review timestamp, approval reference, and separate
  organizational-use, isolated-container, host-staging, and
  adapter-distribution approvals. The SDK redistribution policy is fixed to
  `no-sdk-redistribution`. Final validation requires exactly one Kaiwu
  distribution in the environment lock and cross-checks both its version and
  reviewed artifact digest; a version-only lock or prose-only approval can no
  longer pass acceptance.
- Standalone SDK approval validation now fixes the published and independently
  reproduced wheel SHA-256 as part of the reviewed 1.3.1 identity. A merely
  well-formed digest for different bytes can no longer authorize the pinned
  lane, even if its filename, version, and source labels are copied unchanged.
- That gate now also executes before provider use rather than only during final
  assembly. A strict, private standalone SDK-approval record supports the
  earlier live smoke without depending on unfinished protein inputs; its digest
  is retained in smoke evidence. Smoke and all four system/training/evaluation/
  replay entrypoints verify the approved Kaiwu version and wheel digest against
  the exact environment-lock distribution before credential discovery, license
  initialization, or quota submission; training, evaluation, and replay also
  do so before loading their executable plugin workflows. Final validation
  independently repeats the binding.
- The private SDK/use approval contract is now version 1.1 and also binds the
  exact SDK-capable project number, its account-bound assignment-review time,
  and a retained assignment reference. The live smoke and all three larger
  quota-consuming launchers require their normalized project argument to match
  that reviewed value before credential discovery. Readiness reports a stable
  mismatch reason, and final validation independently rejects either a smoke
  project or any executable-component receipt from another project. A merely
  printable project string can therefore no longer open the provider gate. The
  rights-review timestamp must also be on or after the bound service agreement's
  effective date; approval of an earlier terms version cannot authorize the
  current SDK lane. Rights-review and project-assignment timestamps more than
  five minutes in the future are rejected before readiness or credential
  discovery, preventing a not-yet-effective approval from opening a
  quota-consuming entrypoint.
- The SDK approval, environment lock, and live frozen-config readers now share
  a bounded descriptor-relative private-input path. They require an owner-only
  non-symlink parent and owner-only regular leaf, open the leaf with
  `O_NOFOLLOW`, and recheck parent/leaf identity and metadata after reading.
  Public directories, parent or leaf symlinks, oversized inputs, and path
  replacement therefore fail before credentials rather than becoming trusted
  control state.
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
- Client initialization now extends that credential-free preflight through the
  pinned module structure: `license.init`, `CheckpointManager.save_dir`, and
  callable `cim.CIMOptimizer` must all exist before credentials are resolved or
  the license initializer runs. The already-preflighted module object is then
  used for initialization, avoiding a second import with different structure.
- A pinned Kaiwu 1.3.1 client now implements the documented checkpoint model:
  submission calls `solve` once, later polls use the same task-name and matrix
  identity, restoration recreates that identity, and completed spins are
  independently scored. SDK globals are scoped and restored, and malformed or
  leaking vendor failures fail closed. Because documented APIs do not expose a
  stable provider task ID or provider-reported target, the client records those
  evidence gaps and cannot yet satisfy hardware acceptance.
- The version-pinned client now maps FlagQuantum's stable `optimization` and
  `sampling` modes to the Kaiwu 1.3.1-documented `quota` and `sample` strings.
  It no longer probes a newer `TaskMode` vocabulary or forwards 1.4.1 mode
  names into the 1.3.1 lane. Supporting 1.4.1 requires a separately pinned and
  tested adapter instead of runtime version guessing.
- `KaiwuSDKClient` and every live QDiffusion/smoke entrypoint now reject an SDK
  version other than 1.3.1 before license initialization or credential
  resolution. This prevents a nominal 1.4.1 run from silently using the pinned
  1.3.1 task-mode contract.
- The 1.3.1 documentation declares `get_task_result(ising_matrix) -> dict` but
  does not document that dictionary's fields; the current 1.4.1 documentation
  still does not define provider task-ID or target keys. The pinned client now
  records a value-free schema of the returned dictionary (field names, value
  types, lengths, dtypes, and shapes) after completion. Raw values are never
  retained, and this diagnostic structure does not automatically map or accept
  a provider identity. This absence was rechecked on 2026-10-06 against the
  official 1.3.1 and latest CIM API pages; both continue to document only the
  dictionary return type rather than a stable result-field schema.
- Result-schema diagnostics now bound field counts, field-name length and
  syntax, dimensions, dtype names, and sequence-type inspection. Unsafe names
  are omitted, and inspection failure remains redacted and nonfatal to valid
  samples rather than causing unbounded or value-bearing evidence output.
- Solution-array decoding is now inside the same redacted SDK boundary as the
  vendor call itself. Exceptions from `np.asarray`, provider-controlled element
  comparison, list conversion, or integer conversion become a stable
  cause-free `KaiwuSDKError`; their dynamic text cannot enter logs or evidence.
- The decoder inspects element types before integer normalization. A boolean
  solution array containing `True` can no longer pass the numerical `+1`
  comparison and be converted into apparently valid integer spin evidence.
- Terminal `wait()` failures now expose only the normalized failed or cancelled
  category. The raw provider status remains available through the explicit job
  property for controlled evidence handling, but is no longer interpolated
  into exception text where a future provider vocabulary could leak dynamic
  service details into logs.
- Before its first SDK operation, that client now atomically persists a
  mode-0600, credential-free recovery bundle in the Kaiwu checkpoint directory.
  This closes the ambiguous-submission window: a process restart reuses the
  original timestamp and task/matrix identity, while corrupt or conflicting
  bundles fail before any SDK operation.
- Recovery coverage now includes an indeterminate first SDK-operation failure:
  a newly constructed client with the same frozen request reuses the original
  receipt timestamp and sole deterministic recovery path before retrieving the
  result, rather than creating a second local task identity.
- Checkpoint directories now fail before license initialization unless they are
  private, regular directories. Recovery and explicit job receipts are synced
  before atomic no-overwrite publication and reopened without following
  symlinks. Parent-directory access is descriptor-relative and binding-checked,
  so public, non-regular, partial, replaced, or redirected files fail before an
  SDK task operation.
- The pinned client now freezes the checkpoint directory device/inode before
  license initialization and rechecks that identity before recovery access and
  on entry to and exit from every vendor checkpoint context. Replacement before
  an operation prevents SDK access; replacement during an operation is detected
  as a failed attempt and must never trigger an automatic retry because the
  provider may already have observed the original task identity.
- Every status and result lookup now repeats that frozen-directory check both
  before and after reading its authoritative recovery bundle, including when a
  completed solution is already cached in memory. A replacement private
  directory containing copied valid-looking receipt bytes therefore cannot
  legitimize cached provider results or bypass the original checkpoint inode.
- Pinned 1.3.1 recovery also enforces an exact top-level schema, an aware UTC
  submission timestamp, and absent provider task/target identities. Local
  receipt editing therefore cannot fabricate the provider evidence that the
  documented SDK mapping does not expose.
- SDK recovery matrices now reuse the strict Remote matrix boundary instead of
  coercing JSON values with `float(...)`. Boolean and numeric-string elements
  that compare equal to original coefficients are rejected before SDK access,
  so type-forged checkpoint content cannot preserve a false matrix identity.
- The SDK loader also applies the complete receipt validator before comparing a
  stored request with the in-memory request. Floating-point `matrix_size` and
  `requested_samples` values can no longer exploit Python's numeric equality
  with integers and reach the optimizer before the outer job is constructed.
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
  through every nested evidence key and value before JSON serialization and
  file creation. The QDiffusion system, training, and replay writers use the
  same pre-serialization rule, so JSON escaping cannot hide a credential from
  the refusal check.
- Phase 2 smoke evidence now binds acceptance to an explicit real-SDK transport
  marker. Injected clients remain test evidence even when they return plausible
  provider identities, and the live command writes its diagnostic record but
  exits nonzero whenever hardware acceptance remains closed.
- Smoke, system, training, evaluation, and replay records now require an
  existing private, non-symlink output directory and a final path that does not
  already exist. Every quota-consuming entrypoint validates all final evidence
  destinations before credential resolution, while the shared exclusive writer
  repeats the check at publication time and synchronizes file contents and
  parent-directory metadata. Invalid local destinations therefore fail before
  provider initialization, and a raced file is never overwritten after quota
  has been consumed.
- Evidence publication now anchors temporary creation, exclusive linking,
  rollback, and synchronization to one `O_NOFOLLOW` directory descriptor. It
  verifies the visible parent inode before and after linking and removes the
  candidate record if that binding changes, preventing a parent-path swap from
  redirecting a live record into another directory.
- All four quota-consuming CLIs now also validate the Kaiwu checkpoint root as
  an existing absolute, private, non-symlink directory before credential
  resolution. `KaiwuSDKClient` independently repeats its private-directory
  check before license initialization, so the early check does not weaken the
  provider boundary.
- Offline artifact preflight, environment-lock, transfer-verification, and A800
  development records now use that same durable writer. Missing, public, or
  symlinked output parents therefore fail before publication across both the
  preparatory and quota-consuming evidence lanes.
- Quota-consuming smoke and system execution now convert keyboard interruption
  and post-submission failures into failed attempted records. Smoke retains any
  available task receipt and raw status, stops before submitting its next task,
  and keeps all hardware gates closed; this aligns it with the training and
  replay failure-evidence behavior.
- Smoke, system, training, and replay failure evidence now records only a fixed
  exception category with an empty message. Arbitrary vendor, plugin, parser,
  artifact, and dynamically named exception text is never serialized. Remote
  matrix conversion and receipt restoration also discard untrusted exception
  causes, preventing malformed inputs from reappearing in tracebacks while
  preserving the owned failure classification.
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
- A 2026-10-06 read-only recheck found the same system state on both aliases:
  Python 3.10.12, eight A800-SXM4-80GB devices with driver 580.126.20, Docker
  29.1.3, and no system Torch or Kaiwu module. No remote files were changed;
  the pinned execution lane remains an external provisioning prerequisite.
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
- That development rehearsal has now passed independently on `jp-a800-171` and
  `jp-a800-172` at source revision
  `f122e6cf38f7dd09f26615bd6a5f7a0c3bad01fb` and plugin revision
  `f047bce7b1077449967bbe9e9fab5741542b48d4`. Each run observed
  `NVIDIA A800-SXM4-80GB` on `cuda:0`, completed ten bounded fake-transport
  sampler calls, produced a finite objective and nonzero gradient and parameter
  update, passed token constraints, and declared no fallback. The private
  streamed development-record SHA-256 digests are
  `90138c081453a1bd7f7c442c634590dd98a79d79d4d73ef305bd2e915af57a10`
  for `jp-a800-171` and
  `ad0a1f227ef4fcb6df045f19d68d119d958a06324abeca5c17c79f8137265c74`
  for `jp-a800-172`. The reviewed
  archives were streamed into an auto-removed, network-disabled, read-only
  container whose input, workspace, and evidence existed only in tmpfs; no
  source or evidence was written to either host filesystem. The runs used the
  pre-existing Python 3.12.3/Torch 2.13.0+cu129 validation images, contacted no
  QBoson service, consumed no provider quota, and produced
  `development_fake_transport` records with `qboson_hardware_used=false`,
  `real_provider_evidence=false`, and `system_acceptance=false`. They therefore
  establish only the two-host A800 development path, not the pinned-runtime,
  real-provider, system, portability, or protein acceptance gates.
- The development runner now implements the approved no-remote-persistence
  path directly: it validates a private reviewed bundle and post-extraction
  preflight locally, streams all inputs over SSH into a network-disabled,
  read-only, auto-removed container, uses tmpfs for both the extracted workspace
  and remote record, and streams JSON back to an exclusive local mode-0600
  file. It uses no host bind mount and disables container logging. A separate
  `COPYFILE_DISABLE=1` plus `tar --no-xattrs` prevents macOS extended
  attributes from entering the outer tar stream alongside the manifest-bound
  inputs. The local capture gate rejects identity drift, non-A800 or
  non-`cuda:0` execution,
  unbounded call accounting, or any record that claims real provider use or
  acceptance. After suppressing host extended attributes, the committed path
  passed without tar metadata warnings on both hosts at revision
  `30373d3966ca1227ea73b4548d30443c3c7f131c`; the private development records
  have SHA-256 digests
  `4f989311863169c72ef2fd38031f54ea699d5fef8382102243ac8129ae3acbee`
  (`jp-a800-171`) and
  `6b755aa6e79749633e89ef4db7572ca269fd3a57995026a351acc227c311c1b7`
  (`jp-a800-172`). Both observed A800-SXM4-80GB on `cuda:0`, made ten bounded
  fake-transport calls, produced the same finite objective, nonzero gradient
  and parameter update, and retained all acceptance flags as false. These
  remain development-only records and are not inputs to the live acceptance
  manifest.
- The same bounded streaming rehearsal was repeated after the sampler quota,
  recovery, and evidence-publication hardening at source revision
  `648102538e23ea84759e858b4c57fd2ae5576108`. Both hosts again observed
  `NVIDIA A800-SXM4-80GB` on `cuda:0`, completed ten calls within a 64-call
  budget, produced the same finite objective, nonzero gradient, and parameter
  update, passed token constraints, and reported no fallback. The new private
  mode-0600 records have SHA-256 digests
  `65a9617d94e3864a3a8fb1af6848c364979169713cad646986a95587bf9b31f1`
  (`jp-a800-171`) and
  `262ae985ce78bdfba5d916893eb9fe701ad769d595f3130fa4bac3d67306f383`
  (`jp-a800-172`). Each record passed the offline retained-record validator and
  is bound to its host-specific immutable validation-image ID, the reviewed
  transfer manifest, and the host-specific extraction preflight. These runs
  again used the explicit in-memory fake transport, contacted no QBoson
  service, consumed no provider quota, and kept `qboson_hardware_used`,
  `real_provider_evidence`, and `system_acceptance` false. They update only the
  two-host development evidence; they do not satisfy any live-provider,
  pinned-runtime, portability, protein, or final acceptance gate.
- After the explicit `cuda:0` entrypoint restriction and the system, training,
  portability, and evaluation producer gates were hardened, the same isolated
  rehearsal passed again on both hosts at source revision
  `d35f3ec35aafa3bd71c57ff0f878544bb3f65be2`. The private mode-0600 record
  digests are
  `0582c7c2c595b07bac291fcc5099b19464373a9b28cacf2fe945eeaed5e1b830`
  (`jp-a800-171`) and
  `43c49e51e688ef5569d439207434dfb1f6a12dbfcd2d1bda51767df807334d0e`
  (`jp-a800-172`). Both offline retained-record validations passed; each run
  observed `NVIDIA A800-SXM4-80GB` on explicit `cuda:0`, made ten calls within
  the 64-call development budget, reported no fallback, and retained
  `qboson_hardware_used=false`, `real_provider_evidence=false`, and
  `system_acceptance=false`. The reviewed sources and records again existed on
  the hosts only inside network-disabled, read-only, auto-removed tmpfs
  containers with logging disabled. This is current-revision development
  evidence only and does not advance a live QBoson or protein acceptance gate.
- After adding the staged offline readiness audit and its checkpoint-directory,
  credential-format, SDK-approval-alignment, common-transfer-manifest, and A800
  `cuda:0` gates, the bounded rehearsal was repeated at source revision
  `fa5d8bdb4c455a9e2182dd26363df5a43c4ebccb`. Both independently streamed
  records passed the retained-record validator. Their private mode-0600 SHA-256
  digests are
  `5a994e49947208db5069a0ebe51e96c3387bcfc879188afc41a3b83f21269264`
  (`jp-a800-171`) and
  `dd15bcd97fb38abe945373ab496593a88dc77b1ee97cf80b0c199400b4bc9463`
  (`jp-a800-172`). Each run observed `NVIDIA A800-SXM4-80GB` on explicit
  `cuda:0`, completed ten calls within the 64-call development budget, passed
  the bounded QDiffusion forward, sampling, backward, update, and generation
  slice, and reported no fallback. The reviewed archives again existed on each
  host only inside a network-disabled, read-only, auto-removed container with
  its input, workspace, and evidence on tmpfs and container logging disabled.
  No QBoson credentials, service, or quota were used. The transport remains the
  explicit in-memory fake and `qboson_hardware_used`, `real_provider_evidence`,
  and `system_acceptance` remain false; these are refreshed independent
  single-host A800 development records, not multi-node, distributed, domestic-
  accelerator, live-provider, system, or protein acceptance evidence.
- The isolated development rehearsal was refreshed again at source revision
  `902a3bb70e81352efb40ba13ded8d788611b1894` after the provider-identifier
  pre-credential gate, current platform state, and asset intake were committed.
  Both streamed
  records passed the offline retained-record validator from a persistent local
  mode-0700 evidence directory. Their mode-0600 SHA-256 digests are
  `fb4be8b82ebe5f3f2905bb81c8dbca5fdb8d8bc2f1bc5a80a28dac556f5e276f`
  (`jp-a800-171`) and
  `922716d8e1627c104774091282a0c7ff9d9a0fbf26ab4b4b906510a9993d44e8`
  (`jp-a800-172`). Each independent run observed
  `NVIDIA A800-SXM4-80GB` on explicit `cuda:0`, completed ten calls within the
  64-call development budget, produced a finite objective plus nonzero gradient
  and parameter update, passed token constraints, and reported no fallback.
  The reviewed inputs and remote record existed on each host only inside a
  network-disabled, read-only, auto-removed container using tmpfs and disabled
  logging. The runs used no QBoson credential, service, or quota and retain
  `transport=in_memory_fake`, `qboson_hardware_used=false`,
  `real_provider_evidence=false`, and `system_acceptance=false`. They refresh
  only the two independent single-host A800 development records and establish
  no live-provider, system, protein, distributed, multi-node, or domestic-
  accelerator acceptance.
- Retained development records now have an offline `validate-record` path that
  rechecks the private record, extraction preflight, retained manifest, host,
  image, and revision hash chain without SSH or provider access. Source
  preflight loading also uses the bounded descriptor-relative private reader,
  rejecting public or replaced parents and oversized inputs rather than relying
  on an ordinary path read.
- Transfer and post-extraction verification now load the reviewed manifest
  through the same bounded descriptor-relative private reader. The extraction
  verifier rejects a manifest whose captured digest changes after the bundle
  pass, and streamed development-record capture obtains its source-preflight
  digest from the full private semantic validator rather than an ordinary path
  hash.
- Each reviewed source archive is now opened once through an owner-only parent
  descriptor. Its digest and tar-member safety scan consume that same file
  description, after which leaf metadata, visible path binding, parent binding,
  and the exact colocated archive-name set are rechecked. Replacement can no
  longer splice independently valid bytes into the hash and archive passes.
- The bundle builder now captures each freshly generated archive as a stable
  no-follow snapshot and derives the manifest digest from it. All three archive
  snapshots plus the exclusive manifest snapshot remain bound until both host
  verification passes finish; a replacement during self-verification makes the
  build fail instead of returning a reviewed manifest.
- Post-extraction verification independently reopens each archive through the
  same private descriptor boundary, rehashes it against the captured manifest,
  and compares extracted content using that identical open description. An
  archive replaced after the initial bundle pass cannot become the authority
  for a source-preflight record.
- Extracted regular files are now hashed through no-follow descriptors bound to
  their observed inode. The verifier retains and finally rechecks the complete
  path set and stable file/directory metadata, so a source mutation after its
  content comparison cannot survive into an `extracted_content_verified=true`
  record.
- Extraction preflight and execution-time source-root validation now share that
  stable-tree implementation. Runtime FlagQuantum and plugin verification no
  longer reintroduce ordinary path reads after the stronger extraction gate;
  both recompute the same path-aware content-set identity while rejecting
  mid-read mutation and platforms without no-follow support.
- A separate quota-guarded live-system command now composes that QDiffusion
  slice with `KaiwuSDKClient`. It binds execution to the preregistered config
  hash and exact software lane, persists attempted receipts, checks repeat
  retrieval without a new sampler submission, scans output for both credential
  values, and fails the system gate when provider task or target IDs are absent.
- The quota-consuming live-system entrypoint now requires the requested device
  to resolve exactly to `cuda:0` before credential resolution or client
  construction. Bare `cuda`, another CUDA index, CPU, and malformed device
  names fail locally instead of consuming QBoson quota for a run that the
  acceptance validator must later reject. The two development entrypoints use
  the same explicit device restriction.
- The live-system producer independently repeats the acceptance-critical
  runtime checks before writing `acceptance.system=pass`: proven completed
  provider use, A800 plus exact `cuda:0`, proposal/energy/generation placement,
  finite objective and gradients, and positive finite gradient and parameter
  update. This prevents a direct helper call or future CLI refactor from
  emitting a passing component that only the final bundle validator rejects.
- The portability producer applies the same fail-closed rule: a pass now
  requires proven completed provider use, one consistent provider target,
  A800 execution requested and observed on `cuda:0`, a finite energy objective,
  validated tokens, non-resubmitting retrieval, and complete precision
  evidence. Merely reporting a CUDA output device while the requested runtime
  device was CPU cannot produce a passing replay record.
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
- Per-matrix precision evidence now serializes both the original and submitted
  matrix digests with its scale and error values. System, training, and replay
  validation require unique original identities, exact aggregate
  recomputation, and complete equality between submitted precision identities
  and Remote receipt matrix identities; a report count alone cannot establish
  coverage.
- The same per-matrix evidence now records the canonical float64 coefficient
  range plus the exact arithmetic-mean symmetry normalization and
  round-half-to-even policy. Final validation rejects absent, nonfinite,
  inverted, or substituted normalization metadata rather than relying on an
  implementation comment to describe the lossy conversion.
- Precision evidence now also retains the original plugin object type and
  dtype. Every sampler transfer carries both the original-matrix and submitted
  matrix digests, and system validation requires the complete
  `(original, submitted, type, dtype)` identity set to equal the precision
  evidence set. A receipt-linked submission can no longer conceal a substituted
  plugin input at the A800-to-CPU boundary.
- One shared live-record predicate now applies those receipt, normalization,
  original-input, and sampler-transfer checks before system or portability
  status can be written as passing, and before training can claim complete
  precision evidence. It also requires each serialized evidence value to equal
  its retained `IntegerPrecisionReport`. The final validator independently
  requires both system and portability completeness flags, preventing a
  component from temporarily overstating a result that only fails during final
  assembly.
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
- Frozen protein inputs now also require HTTPS acquisition and license-evidence
  URLs, an approved license identifier, and a timezone-aware review time. The
  validator rejects insecure or credential-bearing URLs, fragments,
  `NOASSERTION`, `UNKNOWN`, `UNLICENSED`, and placeholders before credentials
  are resolved. The current candidate DPLM checkpoint reference and metadata
  both identify revision
  `49b7125a5d28c6418fcc2f3c4fe799352ac1488b`; its seven-file inventory exposes
  checkpoint and tokenizer files but no license file, and the metadata has no
  license field. Its official code repository is Apache-2.0 and describes
  pretrained weights, but the exact separately hosted checkpoint bytes have not
  been authoritatively linked to that notice. The checkpoint therefore remains
  blocked rather than being treated as approved merely because it is publicly
  downloadable. A 2026-10-06 intake refresh also identifies UniProt release
  `2026_03` and its canonical human reference-proteome FASTA from the official
  `RELEASE.metalink`; that record declares CC BY 4.0, size 7,728,297, and
  upstream MD5 `4e4f5aca22ba12eabda1e347765db069`. The canonical filename
  matches the plugin's missing default input, while the separately published
  `_additional` FASTA is excluded from the candidate semantics. The official
  `previous_releases` index was also checked on 2026-10-06 and listed releases
  only through `2026_02`; no immutable `2026_03` archive path is yet available,
  so the intake must not invent one and must freeze the exact approved
  `current_release` bytes if acquisition precedes archival. The official
  ESM README directly maps `esm2_t33_650M_UR50D` to the selected checkpoint URL
  and its repository is MIT-licensed. These facts narrow source and license
  review; they do not authorize acquisition or establish that repository
  licenses govern separately hosted model bytes. Immutable identities, explicit
  approval, decompression provenance, and final SHA-256 digests remain required.
  The asset intake runbook records that no model or dataset download has yet
  been authorized or performed.
- Protein training now rehashes the frozen config, dataset, shared base
  checkpoint/tokenizer tree, and ESM2 checkpoint after the workflow returns.
  Portability replay repeats that check and revalidates the transferred test
  FASTA and trained energy checkpoint; ESM2 evaluation revalidates its local
  checkpoint and every consumed training output after metric calculation. Any
  mid-run input drift prevents passing candidate evidence, and final assembly
  independently requires `artifact_inputs_unchanged=true` for training,
  evaluation, and replay components.
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
- Complete training-component validation now also binds every seed to the
  frozen FlagQuantum, plugin, Python, Torch, Kaiwu SDK, and environment-lock
  lane; the primary A800 and `cuda:0`; both system and per-seed call budgets;
  the artifact preflight and trained-checkpoint digests; safe output names; and
  the exact held-out, baseline, guided, history, metric, and quality artifact
  identities. A plausible provider receipt cannot compensate for missing or
  substituted workflow evidence.
- Provider samples per Ising request are now preregistered in the experiment
  config and enforced by the system, protein-training, portability, assembly,
  and final-validation paths. Protein records also reconcile ordered provider
  task IDs, one provider-reported target, unique matrix identities, requested
  sample counts, and aware submission timestamps against their receipts, and
  retain bounded scale/error summaries for every precision report. A bare
  precision-report count is no longer sufficient evidence.
- System, protein-training, and portability components now serialize the full
  Remote receipt identity, including its schema and matrix size. One shared
  final-evidence check requires the exact field set, nonempty task and project
  names, sampling mode and count, a positive matrix size, matrix digest,
  provider task and target identities, and an aware UTC submission time; it
  also rejects duplicate task or matrix identities within a component.
- System-probe and protein-training call budgets are now separate. A validator
  derives a conservative per-seed submission bound from the pinned plugin's
  actual positive/negative energy and generation loops; the illustrative full
  config requires up to 71,269 submissions per seed, so its protein budget stays
  unresolved until experiment size and QBoson quota are explicitly approved.
- A credential-free quota planner now exposes the same calculation as a
  strict-JSON report, broken down into the two-host system slice, each protein
  seed's structural/training/generation calls, the replay fixture, and the
  two-task smoke. The illustrative config derives a conservative maximum of
  213,846 distinct provider task submissions in total and deliberately reports
  an unresolved declared ceiling while its per-seed protein budget is
  `<required>`. Deduplication may lower an actual run. Polling and repeat
  retrieval are excluded from the submission unit, and the report states that
  it is planning data rather than quota approval or provider evidence.
- Completed training records now hash the exact held-out, baseline, guided,
  history, and sequence-quality artifacts. A local-only ESM2 evaluator verifies
  that chain, loads one frozen checkpoint file without implicit download, checks
  aligned sequence identities, and computes candidate cosine/L2 evidence on the
  primary A800 without spending further QBoson quota.
- The training producer now requires its newly created run directory to be a
  real direct child of the seed output root. It captures the selected energy
  checkpoint and all seven workflow artifacts through stable no-follow file
  snapshots, then cross-revalidates the complete set before recording their
  identities; symlinked or concurrently replaced outputs cannot enter a
  completed training component. The top-level workflow output root is validated
  as an existing owner-only directory before credentials, and directory/file
  creation runs under a restored `umask 077`; group- or other-accessible output
  evidence is rejected. The producer retains all eight output snapshots across
  the frozen-input postflight and rechecks them immediately before exclusive
  record publication, closing the gap between output hashing and evidence
  write.
- The protein-training CLI now distinguishes a locally completed workflow from
  an acceptance-eligible seed component. It still publishes the private record
  for diagnosis, but exits successfully only when real pinned SDK transport,
  completed QBoson use, unique provider tasks and target, the per-seed quota,
  precision evidence, frozen-input postflight, and the trained-checkpoint
  identity are all complete. An injected or identity-incomplete run can no
  longer return success merely because the upstream workflow exited normally.
- The shared training-record loader used by both ESM2 evaluation and replay-host
  execution now revalidates real SDK transport, QBoson-use and identity flags,
  sampling receipts, call budget, no-fallback state, and precision completeness.
  Injected or incomplete training evidence therefore fails before either A800
  evaluation work or another quota-consuming replay can begin.
- Evaluation and replay consumers now also require the recorded training run
  directory to remain owner-only and non-symlinked, and reject any consumed
  training artifact or selected checkpoint whose permissions were widened.
  Evaluation performs both this check and the frozen ESM2 identity pass before
  initializing CUDA, while replay performs it before resolving credentials.
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
- The ESM2 evaluation producer now runs that same component validator before
  exclusive publication, including the closed top-level schema and version.
  Baseline and guided metric mappings must contain exactly the twelve frozen
  metric names; out-of-range values, invalid sequences, missing metrics, and
  undeclared extensions fail locally instead of surviving until final bundle
  assembly.
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
- Final bundle publication now syncs every newly written file, the component
  directory, the staging root, and the output parent after the atomic directory
  rename. A successful assembler exit therefore represents durable evidence
  publication rather than only page-cache visibility.
- The assembler freezes the output parent's device/inode and private mode for
  that publication window, then rechecks it after staging creation, final
  validation, atomic rename, and directory sync. Replacing the parent and
  planting a same-named attacker staging directory cannot redirect publication.
- The output parent is also kept open throughout publication. Final rename and
  directory sync are descriptor-relative, so a replacement in the interval
  after the last visible-path check remains anchored to the originally opened
  private directory and is then reported as a failed publication.
- A credential-free local golden-path script now verifies clean checkouts at the
  pinned Kaiwu Community and Kaiwu PyTorch Plugin revisions, clears provider
  credential variables, explicitly gates source-only conformance without
  declaring an unapproved dependency extra, and runs conversion, lifecycle,
  sampler, plugin, and live-probe contract tests without network access or
  provider quota. It is explicitly local conformance evidence rather than A800
  or QBoson evidence. The entrypoint now prepends a dedicated `sitecustomize`
  guard that denies DNS resolution and IP socket connections in pytest and its
  Python subprocesses; Hugging Face offline variables are retained but are no
  longer the only network control. After adding the quota planner, pinned 1.3.1
  mode mapping,
  fail-closed SDK-lane selection, xattr-free stream contract, and pre-credential
  private-output validation, credential clearing, persisted-failure redaction,
  printable canonical provider-identity validation, and exact plugin metadata
  provenance to that gate, the complete path passed at revision
  `bb4a11bc0d3d71f77ca867993590bf721aea9632` on 2026-10-06 with 477 tests and
  no provider credentials present.
- The credential-free golden path was rerun at revision
  `c03aed52dd5896d6494194b451fdcdb6cc7c14ad` after the current producer and
  evidence-gate hardening. All 672 selected tests passed against Kaiwu
  Community revision `b648b531c034bd6ae9b7a34fed994c717967cc72` and Kaiwu
  PyTorch Plugin revision `f047bce7b1077449967bbe9e9fab5741542b48d4`, including
  the source-conformance tests that the ordinary dependency-free suite skips.
  The runner cleared all QBoson credential variables and its socket guard kept
  the process offline. This run used the available local development lane
  (Python 3.12.14 and Torch 2.14.0), so it updates Phase 1 local conformance
  only; it does not establish the frozen Python 3.10/Torch 2.7/Kaiwu 1.3.1
  environment, A800 execution, provider use, or acceptance.
- After reconciling the authenticated-platform and protein-asset state, the
  complete credential-free golden path was rerun at revision
  `088c98bc7563f3748fe71cca42638eb7db5840f7` on 2026-10-06. All 701 selected
  tests passed in 27.89 seconds against the same clean Kaiwu Community revision
  `b648b531c034bd6ae9b7a34fed994c717967cc72` and Kaiwu PyTorch Plugin revision
  `f047bce7b1077449967bbe9e9fab5741542b48d4`. The runner cleared provider
  credentials and denied network access. The available local lane was Python
  3.12.14 and Torch 2.13.0, so this remains Phase 1 local conformance evidence;
  it does not satisfy the pinned SDK, A800, live-provider, or QDiffusion
  acceptance gates.
- A value-free offline readiness command now composes the existing frozen-config,
  quota, approved-SDK environment, two-host source-preflight, common transfer
  manifest, and protein-artifact validators into one fail-closed inventory. It
  validates credential and project-variable presence and format without
  recording their values, validates the private Kaiwu checkpoint directory,
  emits stable reason codes for missing or invalid inputs, and performs no
  provider operation.
  Provider-smoke readiness uses the standalone approval without requiring the
  unfinished protein config; system readiness additionally requires that the
  frozen approval is identical. Its explicit provider-smoke, system-probe, and
  protein-experiment selectors make the process exit status enforce the chosen
  stage; the latter two also require an observed NVIDIA A800 at `cuda:0`, while
  retaining the final protein gate as the default. This closes the
  local handoff checklist but does not resolve the absent approval, SDK artifact,
  credentials, quota, frozen protein assets, or any live evidence.
- The readiness audit now also requires a private, closed-schema Resource Bill
  snapshot whose validity window is no longer than 24 hours. It rejects
  duplicate, incomplete, negative, boolean, expired, or future-dated resource
  rows and requires one SPQC target to expose at least one optimization and one
  sampling resource before the two-task provider smoke can be called ready.
  This prevents the frozen budget ceiling from being mistaken for an available
  account allocation. The authenticated 2026-10-06 state therefore fails with
  `sampling_resource_unavailable`. Its private mode-0600 snapshot has SHA-256
  `1ebe3cfc866213d8152076dd19f08b0486034011bcd48da989f4fd69da785222`;
  the snapshot remains observation only and does not authorize spending or
  establish a project assignment.
- The Phase 2 live-smoke command now requires that same private resource
  snapshot and validates it before SDK approval loading, credential discovery,
  license initialization, or task submission. Its SHA-256 is retained in the
  smoke record. Final assembly copies the exact snapshot into the immutable
  component bundle, and offline validation independently binds its digest,
  closed schema, validity interval, and same-target optimization-plus-sampling
  balance to the smoke timestamp. It also rejects either smoke receipt if its
  submission predates snapshot capture or follows snapshot expiry, so a
  hand-built record cannot attach earlier provider work to a later account
  observation. Directly invoking the quota-consuming smoke can therefore no
  longer bypass the readiness-only resource check.
- The three larger quota-consuming entrypoints now also require a current
  private Resource Bill snapshot before credential discovery or SDK client
  initialization. System and portability runs require one Sampling target to
  cover the frozen remote-call ceiling; each protein-training run requires one
  Sampling target to cover its frozen per-seed ceiling. These checks prevent a
  direct CLI invocation from starting a run whose declared task budget exceeds
  the observed balance, without treating the observation as a reservation.
- System, training, and portability records now retain the exact Resource Bill
  digest, UTC check time, mode, and declared call ceiling. Final assembly accepts
  repeated `--provider-resources` inputs, deduplicates them by content digest,
  and requires the copied snapshot set to equal the set referenced by smoke and
  executable records. Offline validation recomputes each budget decision at its
  retained check time and rejects remote receipts that predate either the check
  or snapshot capture, or that follow the snapshot's validity window.
- All four quota-consuming launchers now bind the validated snapshot's
  `valid_until` timestamp to `KaiwuSDKClient.submission_deadline`. The Remote
  boundary refuses a new submit after that UTC instant before it creates a
  recovery bundle or invokes the SDK, while explicit restore, status, and
  result retrieval for an existing identity remain available.
- The isolated development rehearsal was refreshed on both validation hosts at
  source revision `10e4b2345c5e18219ae369e7a7821848bf8ad46c`, after the
  provider-resource budget, submission-deadline, and reviewed-project gates
  were committed. Both
  retained mode-0600 records passed the offline manifest, extraction-preflight,
  host, image, and revision hash-chain validator. Their SHA-256 digests are
  `cc5ca884b442be77201af71dce89133f4dc4505ae788bff288aec88ea10ea9ba`
  (`jp-a800-171`) and
  `6c19ee2e86e5bc05bd985555d2f19b20ea5c589eb475ab950eec8f87d271dff8`
  (`jp-a800-172`). Each independent run observed
  `NVIDIA A800-SXM4-80GB` on explicit `cuda:0`, completed ten calls within the
  64-call development budget, produced a finite objective plus nonzero gradient
  and parameter update, passed token constraints, and reported no fallback.
  Reviewed source and evidence existed on each host only inside a
  network-disabled, read-only, auto-removed container whose input, workspace,
  and output were tmpfs-backed and whose logging was disabled. No QBoson
  credential, service, or quota was used. The records remain
  `development_fake_transport` with `qboson_hardware_used=false`,
  `real_provider_evidence=false`, and `system_acceptance=false`; they establish
  neither live-provider or protein acceptance nor distributed, multi-node, or
  domestic-accelerator support.
- The same isolated rehearsal was refreshed after the recovery-state ownership
  and bounded-descriptor hardening at source revision
  `cd90a5da109404524c9eceeb4093e3aa1509e9a5`. Both private mode-0600 records
  passed the offline manifest, extraction-preflight, host, image, and revision
  hash-chain validator. Their SHA-256 digests are
  `349b6e2e3fc58cd7a2db7adf7d74ce8adb5edd0ffc19de824e709dd220ca582d`
  (`jp-a800-171`) and
  `1e0357b04701d7e4871c3cdeab25bb9834de964d770046a245a745da29da03a1`
  (`jp-a800-172`). Each independent run observed
  `NVIDIA A800-SXM4-80GB` on explicit `cuda:0`, completed ten calls within the
  64-call development budget, passed the bounded forward, fake sampling,
  backward, update, and generation slice, and reported no fallback. Reviewed
  source and evidence existed on each host only inside a network-disabled,
  read-only, auto-removed container with a tmpfs-backed workspace and disabled
  logging. No QBoson credential, service, or quota was used. The records remain
  `development_fake_transport` with `qboson_hardware_used=false`,
  `real_provider_evidence=false`, and `system_acceptance=false`; they refresh
  only the two independent single-host A800 development records and establish
  no live-provider, system, protein, distributed, multi-node, or domestic-
  accelerator acceptance.
- After extending effective-UID ownership to shared evidence, final bundle
  publication, protein outputs, and stable snapshots, and after normalizing
  streamed extraction ownership, the rehearsal passed again at source revision
  `f44323912fda05c8e39dcc3ea75a5ebd39c40c0a`. The two retained mode-0600
  records passed their offline manifest, extraction-preflight, host, image, and
  revision hash-chain validation. Their SHA-256 digests are
  `450f1940bb9542b530aaeaafd84368b68ca0e112eb1d515adcfedd230461e6c9`
  (`jp-a800-171`) and
  `b31525376161b1d09099a2556e8db8e1e776435d0ea6f8aa5ad9f3fc108a4fa1`
  (`jp-a800-172`). Each independent run observed
  `NVIDIA A800-SXM4-80GB` on explicit `cuda:0`, completed ten calls within the
  64-call development budget, passed forward, bounded fake sampling, backward,
  update, generation, and token constraints, and reported no fallback. Inputs,
  workspace, and evidence existed on each host only in tmpfs inside a
  network-disabled, read-only, logging-disabled, auto-removed container. No
  QBoson credential, service, or quota was used. Both records remain
  `development_fake_transport` with `qboson_hardware_used=false`,
  `real_provider_evidence=false`, and `system_acceptance=false`; they establish
  no live-provider, system, protein, distributed, multi-node, or domestic-
  accelerator acceptance.
- The isolated rehearsal was refreshed at source revision
  `d329b10419c6470757bb0e1d789d06033980935e` after freezing the reviewed
  Kaiwu 1.3.1 artifact identity, adding the bounded no-import wheel inspection,
  and recording the protein-source candidates. The two mode-0600 records passed
  independent offline manifest, extraction-preflight, host, image, and revision
  hash-chain validation. Their SHA-256 digests are
  `2a87e4cf12ed3d1b6fcb0e1f83a4c4ea9f9937106db9c4d62b968a29f9cabec8`
  (`jp-a800-171`) and
  `2f41113d78b639f7dc1530303834743e1b96c346534e6f4b4f068f229cbf5e1b`
  (`jp-a800-172`). Each independent run observed
  `NVIDIA A800-SXM4-80GB` on explicit `cuda:0`, completed ten calls within the
  64-call development budget, produced a finite objective plus nonzero gradient
  and parameter update, passed token constraints, and reported no fallback.
  Source, workspace, and remote evidence existed only in tmpfs inside a
  network-disabled, read-only, logging-disabled, auto-removed container; no
  source or evidence was written to either validation host. No QBoson
  credential, service, or quota was used. Both records remain
  `development_fake_transport` with `qboson_hardware_used=false`,
  `real_provider_evidence=false`, and `system_acceptance=false`; they refresh
  only the two independent A800 development paths and establish no live-
  provider, system, protein, distributed, multi-node, or domestic-accelerator
  acceptance.
- Machine-checked draft API gates now keep Kaiwu out of the stable root,
  ecosystem parent, remote parent, and capability-maturity registry; normal
  FlagQuantum imports are also proven not to resolve the optional vendor
  package. The repository boundary inventory rejects Kaiwu imports from Core,
  Runtime, and Simulation, preserving the planned `ecosystem + remote` split.
- The draft boundary gate also imports both provider-specific subpackages under
  a vendor-import trap and statically enforces the dependency direction:
  `ecosystem/kaiwu` has no direct vendor import, while `remote/kaiwu` cannot
  depend on Ecosystem, PyTorch, or the Kaiwu PyTorch Plugin. NumPy remains
  permitted solely as SDK-boundary matrix marshalling. The only allowed vendor
  import site is the lazy loader in `remote/kaiwu/sdk.py`.
- Sampler calls now retain explicit transfer accounting for the plugin-produced
  CPU NumPy Ising matrix, canonical CPU float64 tensor, submitted host tuple,
  returned CPU int8 NumPy samples, cache use, and result shape. System evidence
  also records the A800 device that originated the matrix and received the
  reconstructed samples; the final validator rejects missing or inconsistent
  A800-to-CPU and CPU-to-A800 boundary records.
- Each sampler transfer now also retains the exact submitted-matrix SHA-256.
  System-component validation aligns every non-cached transfer with its Remote
  receipt in order and requires the same matrix digest, matrix width, requested
  sample count, and returned sample shape, closing the gap between individually
  valid transfer and provider records.
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
- The acceptance runbook is now contract-tested against every core CLI parser:
  each documented command must include all `required=True` long options and
  may not contain an option absent from the corresponding implementation. CLI
  drift therefore fails locally before an A800 or quota-consuming session.
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
- Final evidence now retains the frozen protein-artifact preflight as a required
  component instead of leaving only an external digest. Every seed's training
  record and the replay-host portability record must reference that one copied
  preflight, whose config and four artifact identities are independently
  checked during offline final validation.
- The standalone SDK approval and Phase 2 provider smoke are now required final
  components. Offline validation requires the approval object to equal the
  frozen config, binds its exact file digest into the smoke, and independently
  proves one successful optimization plus one successful sampling task with
  real provider task/target identities, the independently recomputed exact
  2-by-2 smoke-matrix identity, no fallback, and the frozen environment lock
  before Phase 4 evidence can pass. Each smoke task also retains its receipt
  schema, project number, and aware-UTC submission time, which must precede the
  enclosing smoke record. The bounded smoke also retains its complete spin and
  energy vectors; final validation checks sample count, width, spin domain, and
  independently recomputes every energy under the FlagQuantum Ising convention.
  The smoke and per-task objects use closed field sets and fixed claim
  limitations, so an unvalidated extension cannot carry hidden evidence or
  sensitive values while still passing.
- Provider identity completeness is now computed rather than self-asserted: the
  two smoke tasks must have distinct SDK task names, distinct provider task IDs,
  and one common printable provider target. The task-name rule is required
  because the pinned 1.3.1 recovery identity is `task_name + ising_matrix` and
  both smoke modes use the same matrix. The target is retained at smoke-record
  level and is checked against both task results before hardware acceptance can
  be true.
- Provider use is now recorded independently from whole-workflow acceptance.
  A completed, identity-bearing result through the exact SDK transport remains
  `qboson_hardware_used=true` when a later smoke task, local QDiffusion step, or
  artifact postflight fails, so consumed quota is not erased from evidence.
  The run and acceptance gates still fail; a submitted receipt without a
  validated result does not by itself claim hardware use.
- All four quota-consuming evidence producers now require the exact pinned
  `KaiwuSDKClient` type before claiming real transport or hardware use. An
  injected subclass cannot promote itself through `isinstance` plus a
  caller-controlled transport flag.
- Final validation now builds one global provider-task identity set across the
  Phase 2 smoke, both system probes, every protein-training seed, and the
  portability replay. Reusing one provider task ID across nominally distinct
  remote components fails acceptance instead of overstating independent quota
  submissions.
- The four executable QDiffusion component records now have closed top-level
  and evidence-bearing nested schemas for system, training, evaluation, and
  portability evidence. Precision summaries, system training/generation and
  transfer objects, workflow-artifact identities, evaluation metrics,
  portability artifacts/fixture, and fixed claim limitations reject missing or
  undeclared fields. Completed remote components must retain no failure. The
  provider-result schema in both Phase 2 smoke and system evidence is restricted
  to the pinned client's value-free type/shape grammar, preventing raw provider
  values from hiding inside schema metadata. A source contract test compares
  top-level schemas with the actual payload constructors so producer/validator
  drift cannot be hidden by hand-written acceptance fixtures. That contract now
  also scans post-construction `payload[...]` assignments. Training and replay
  artifact-postflight failures reuse the closed `run_completed`, `failure`, and
  `artifact_inputs_unchanged` fields instead of appending an undeclared
  diagnostic field that would make every real producer record unassemblable.
- Portability replay validation now independently binds its dataset, base
  checkpoint, tokenizer, evaluation model, and trained checkpoint identities to
  the frozen config and selected training record. Its training seed, fixture
  index, and step count must equal the preregistered generation lane, while the
  retained objective, generated length, and generated-sequence digest must be
  well formed. A structurally complete but cherry-picked replay can no longer
  satisfy final acceptance.
- Executable component timestamps are now evidence rather than unchecked
  strings. System, training, evaluation, and portability records require an
  aware UTC `recorded_at` at or after preregistration, the SDK rights review,
  project-assignment review, and the retained Phase 2 smoke; every remote
  receipt must itself follow those prerequisites and be submitted no later than
  its enclosing record. The two smoke receipts must likewise follow
  preregistration, rights review, and project-assignment review. Training cannot
  predate the primary system probe, and each training receipt must follow that
  probe. Evaluation cannot predate its linked training record. Portability and
  each of its receipts cannot predate either the replay-host system probe or its
  selected training record.
- The frozen config now records a distinct machine-reported hostname for each
  validation alias. All four live QDiffusion entrypoints require
  `--expected-hostname` to equal that frozen mapping before observing the local
  hostname, and final validation independently binds every executable
  component's `observed_hostname` to its declared `execution_host`. This
  preserves the real alias/hostname distinction found during Phase 0 instead
  of treating a caller-controlled alias or Docker container ID as host proof.
- The frozen acceptance config now has a closed top-level and nested schema.
  Software, host identities, artifact provenance, training, generation,
  evaluation, precision, metric, threshold, and SDK-approval sections reject
  undeclared fields; the documented template is tested against the validator's
  field sets. An extra credential-like value can no longer hide in a config
  extension and be copied into the final self-contained evidence bundle.
- The frozen software lane now includes the FlagQuantum package version in
  addition to the exact source revision. System, training, evaluation, and
  portability producers record the executing version; preflight rejects a
  runtime mismatch, component validation rechecks it, and final host records
  retain it. A source checkout with the expected commit but altered package
  version metadata can no longer pass unnoticed.
- The retained environment lock is now closed at both record and distribution
  levels. Only the exact inventory policy, Python version, and sorted package
  identity/version/artifact/content digests are accepted; arbitrary metadata
  or credential-like extensions fail before runtime comparison and before the
  lock is copied into final evidence. Builder tests assert that produced locks
  match these same field sets. Wheel METADATA and artifact digests now come
  from the same stable no-follow snapshots, which remain bound through lock
  publication; the installed distribution inventory likewise cross-revalidates
  every captured RECORD file after hashing the complete set.
- Final assembly now rejects non-private or symlinked inputs and builds in a
  private sibling staging directory. The requested evidence directory appears
  atomically only after the independent validator passes, preventing a failed
  run from leaving a misleading partial bundle at the declared output path.
  Each input's immediate parent must also be a private real directory, and the
  publisher repeats that validation internally so a direct helper call cannot
  bypass the CLI gate. Its output parent is checked by the same rule before a
  staging directory or final atomic rename is allowed. Destination absence is
  checked with no-follow metadata both before staging and after final
  validation, so a dangling symlink cannot be mistaken for an unused path or
  overwritten during publication.
- Final assembly and independent validation now bind every private evidence
  member to one descriptor-anchored byte capture. Assembly refuses a config or
  component whose captured digest differs from the record used to construct
  the final host evidence, then copies those same captured bytes. Validation
  computes each declared digest and parses JSON from the same bytes, including
  the environment lock, closing the check-then-reread replacement window.
- Revalidation now rejects a symlink anywhere in a manifest member path and
  requires the manifest, config, host records, and copied components to remain
  inaccessible to group and other users.
- Final evidence validation now applies a closed-world file-tree check: member
  paths must be unique normalized relative POSIX paths; unlisted files, extra
  empty directories, and group- or other-accessible directories are rejected
  instead of being silently ignored beside declared evidence.
- The final manifest, all member references, both aggregated host records, and
  their nested artifact, precision, training, generation, transfer, metric,
  application, and portability objects now have closed field sets. Both host
  records must explicitly require the complete component bundle, so a
  hand-authored manifest cannot omit or disable that flag to bypass validation
  of the source components. Their fixed limitations also classify the outcome
  as development evidence, distinguish independent single-device A800 runs
  from distributed or domestic-accelerator support, and reject performance,
  production, quantum-advantage, or scalability claims.
- Every QDiffusion configuration and evidence reader now uses one strict JSON
  parser that rejects duplicate object keys at every nesting level. A source
  scan in the credential-free golden path prevents frozen config, environment
  lock, preflight, transfer, component, or final manifest readers from silently
  returning to last-key-wins parsing. The parser also rejects non-standard
  `NaN`/`Infinity` constants and numeric exponents that decode to nonfinite
  floats anywhere in nested evidence.
- The credential-free quota planner and protein evaluator now read their frozen
  configuration and provider-linked training record through the same bounded,
  descriptor-anchored private-input gate. Public files or parents and symbolic
  links fail before planning or GPU work.
- Frozen protein file and model-directory identities now use the shared stable
  no-follow snapshot layer. The dataset profile consumes the captured FASTA
  inode and rechecks its leaf and parent identity after parsing, while model
  trees retain the unchanged `tree-sha256-v1` path-aware digest format. A
  dataset swap between hashing and split validation therefore fails preflight.
- Protein preflight now also binds that staged FASTA to the reviewed gzip
  source artifact. The frozen configuration records the archive SHA-256, byte
  length, `gzip` format, and `gzip-exact-bytes-v1` policy; preflight streams
  decompression under a 4 GiB bound and requires the resulting byte count and
  digest to equal the captured FASTA exactly. The source-archive lineage is
  retained in the preflight component and independently checked by final
  acceptance validation. The exact archive file snapshot now remains anchored
  through primary training and portability replay, with checks around plugin
  work and at postflight; decompression also fails as soon as it exceeds the
  captured FASTA length rather than continuing through oversized output. The
  stable file capture now rejects an oversized FASTA or a gzip whose byte
  length differs from the frozen value before hashing its body, and repeats the
  size check on the opened no-follow descriptor.
- Protein preflight now returns the exact stable snapshots from the pass that
  produced its record instead of reopening the paths to establish a second
  identity. Primary training and portability replay retain those snapshots and
  revalidate the dataset, base checkpoint, tokenizer, and evaluation model
  before and after plugin consumption and once more during postflight; a
  same-content path replacement cannot separate preflight identity from the
  assets used by either run.
- The isolated development probe was refreshed on both validation hosts at
  source revision `4f19b770264a0f4e558625fc622b51a1328f914a`. Both runs used
  `cuda:0`, observed an NVIDIA A800-SXM4-80GB, completed ten bounded sampler
  calls, produced the finite objective `1.3875621557235718`, a nonzero finite
  gradient and parameter update, and passed token constraints with no fallback.
  The locally retained mode-0600 records validate to SHA-256
  `0662fd2424bd2d124d99d2cb9c91bc1abffc99ee20ba8b746087dded40d39ac9`
  for `jp-a800-171` and
  `92bcc9a0ce61c13f01cdf3a468e85368886db6cf80f9ac8c7a12d88955174004`
  for `jp-a800-172`. The transport remained `in_memory_fake`; networking,
  host writes, QBoson credentials, provider service, and quota were not used,
  and all live-provider, system, protein, domestic-accelerator, multi-node, and
  distributed acceptance gates remain closed.
- ESM2 evaluation now retains those stable snapshots for every training FASTA
  and quality JSON after matching the training record. Plugin FASTA reads are
  bracketed by identity checks, quality JSON is parsed from its captured
  descriptor, and both evaluation and portability replay revalidate the
  original snapshots after consumption instead of relying only on a later
  same-digest path lookup.
- Evaluation rechecks its ESM2 and consumed-training snapshots again after the
  evidence payload is constructed and immediately before exclusive record
  publication. Replay likewise finishes postflight by rechecking all four
  frozen artifact snapshots after the training-output and checkpoint checks,
  so payload construction cannot leave an unverified last-write window.
- Portability replay now carries the selected trained-energy checkpoint as a
  stable snapshot after matching its name and digest to the primary training
  record. The plugin weight loader is bracketed by identity checks, generation
  rechecks it again, and final postflight validates the original snapshot; a
  checkpoint swap cannot separate recorded identity from loaded weights.
- ESM2 evaluation now passes the exact file snapshot whose digest matched the
  frozen config into the local loader, revalidates it immediately before and
  after `load_model_and_alphabet_local`, and checks it again after metrics. The
  recorded evaluation-model identity and the weights actually loaded can no
  longer come from different path instances.

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

The initial compatibility environment should be pinned as a separate test lane.
At the reviewed plugin revision, packaging metadata declares Python `>=3.8`,
while the plugin README and installation guide declare Python 3.10 and its
requirements file pins Torch 2.7.0, NumPy 2.2.6, and Kaiwu 1.3.1. The published
Kaiwu 1.3.1 wheel metadata independently requires Python `>=3.10`. The composite
lane therefore freezes Python 3.10 without misrepresenting the broader plugin
packaging marker. Current Kaiwu documentation describes SDK 1.4.1. The project
must validate the chosen SDK version instead of assuming that 1.3.1 and 1.4.1
are interchangeable. FlagQuantum's broader Torch support must not be narrowed
for users who do not install the integration.

Before production support, record the Kaiwu package source, hashes or lockfile,
license, cloud-service terms, credential requirements, and an exit path that
allows the provider client to be replaced without modifying problem consumers.
The identified service-terms source is the public
[QBoson Quantum Cloud Platform User Service Agreement](https://platform.qboson.com/agreement?type=QBoson-SPQC-Platform-Users-Agreement),
effective 2026-07-09. Recording that source does not approve it or establish a
wheel license; retain the organizational review decision and any separately
required written permission as evidence.

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
  "observed_tensor_device": "<device>",
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
