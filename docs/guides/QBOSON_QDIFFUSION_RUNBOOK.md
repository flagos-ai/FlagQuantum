# QBoson QDiffusion validation runbook

## Status and scope

This runbook executes the development, live-provider, two-host system, and
protein-effectiveness stages defined by the
[QBoson Kaiwu integration plan](../roadmap/QBOSON_KAIWU_INTEGRATION_PLAN.md).
It does not authorize provider spending, source transfer, credential sharing,
or a capability claim. Obtain those approvals separately.

The validation hosts are `jp-a800-171` and `jp-a800-172`. Each run is an
independent single-host, single-GPU execution. The pair is not a multi-node or
distributed run, and NVIDIA A800 evidence is not domestic-accelerator evidence.

Run every Python command below from the root of the reviewed extracted
FlagQuantum checkout. Commands use `python -s -m ...` so the current checkout
is imported as a module and user-site packages cannot silently replace it.

## External coordination status

On 2026-10-06 the authenticated QBoson beginner tutorial accepted the Max Cut
answer, marked the tutorial mastered, and displayed a notification that free
real-machine credits had been issued. Later authenticated views were
inconsistent: one reported SPQC-1000 out of service, while a subsequent view
reported it available with zero sampling credits and one optimization credit;
the SDK page also relabeled the tutorial incomplete. The task table and 30-day
task totals remained zero. Treat the tutorial notification and displayed
optimization balance as account reconnaissance only, not as proof that the
required quota pair is stable or approved. Do not submit a provider task until
both required quota classes are shown consistently and separately approved for
this validation.

On the same date, a credential-free platform support request asked for the
Kaiwu 1.3.1 CPython 3.10 Linux package and its digest and terms, an SDK-capable
project number, sampling quota, and written confirmation for isolated-container
use and open-source adapter publication without redistributing the SDK wheel.
The platform displayed `Submitted successfully`; no ticket identifier was
provided. Retain the response received through the account-bound channel as a
private review input. Do not copy account identifiers or SDK authorization
codes into the repository while recording that response.

The authenticated SDK page currently offers Kaiwu 1.4.1 downloads and retains
a 1.3.1 release-note entry stating that sample-mode task submission was added.
It displays an account-bound SDK authorization code only in masked form. That
secret was not revealed, copied, or imported into the integration environment.
The presence of a masked code is not an approved credential handoff and does
not relax the pinned-version, project, quota, package, or terms gates.

## Required inputs

Do not begin a live run until all entries are available and reviewed:

- full FlagQuantum and Kaiwu PyTorch Plugin Git revisions;
- an approved Python 3.10 environment with the exact Torch, NumPy, plugin, and
  Kaiwu SDK versions frozen in `acceptance_config.json`;
- the approved Kaiwu 1.3.1 wheel, its SHA-256 digest, source, license or package
  terms, cloud-service terms, and approved redistribution boundary; public
  availability on PyPI is source evidence, not approval by itself;
- an organizational review decision for the public
  [QBoson Quantum Cloud Platform User Service Agreement](https://platform.qboson.com/agreement?type=QBoson-SPQC-Platform-Users-Agreement),
  effective 2026-07-09. It expressly covers Kaiwu SDK, KPP, remote APIs, and
  SPQC services, restricts SDK/API sale, transfer, and sublicensing, and states
  an own-use restriction. Record whether the intended development, container,
  and adapter-distribution model is permitted or needs separate written
  authorization; do not infer approval from public access to the agreement;
- an assigned QBoson project number and sufficient optimization and sampling
  quota;
- `QBOSON_USER_ID` and `QBOSON_SDK_CODE`, supplied through a private process
  environment rather than command-line arguments, files in the repository, or
  evidence records;
- the frozen protein dataset split, DPLM checkpoint, tokenizer, generation
  settings, seed list, ESM2 evaluation model, training settings, and their
  immutable identities;
- explicit authorization to transfer the reviewed source bundles to both
  validation hosts.

The SDK authorization code is a secret. Do not paste either credential into a
ticket, chat transcript, shell history, command line, log, checkpoint, receipt,
or evidence JSON.

Each quota-consuming CLI resolves the dedicated pair into an explicit
in-memory object and then removes `QBOSON_USER_ID` and `QBOSON_SDK_CODE` from
its own environment before constructing the SDK client. Do not depend on those
variables remaining available to later plugin code or child processes.

Every quota-consuming evidence writer recursively scans all nested JSON keys
and string values for both resolved credentials before serialization and
refuses to create the record on a match. Quotes and backslashes cannot bypass
that scan after JSON escaping; embedded control or format characters are
rejected earlier during credential resolution. A refusal is a security failure
to investigate, not a reason to edit and republish the attempted record.

Before provisioning either host, run the credential-free local golden path
against clean checkouts at the pinned upstream revisions:

```bash
bash examples/qdiffusion_kaiwu/run_local_conformance.sh \
  /absolute/src/kaiwu_community \
  /absolute/src/kaiwu-pytorch-plugin \
  /absolute/flagquantum-venv/bin/python
```

This validates local conversion and provider-boundary contracts without network
access, credentials, quota, A800 access, or the proprietary SDK. It does not
replace any live gate.

Next, run the offline readiness audit before invoking any live command:

```bash
python -B -s -m examples.qdiffusion_kaiwu.audit_readiness \
  --config /private/acceptance_config.json \
  --environment-lock /private/environment_lock.json \
  --sdk-approval /private/sdk-approval.json \
  --checkpoint-dir /private/kaiwu-checkpoints \
  --plugin-root /src/kaiwu-pytorch-plugin \
  --primary-source-preflight /private/jp-a800-171-extraction-preflight.json \
  --replay-source-preflight /private/jp-a800-172-extraction-preflight.json \
  --dataset /private/proteins.fasta \
  --base-checkpoint /private/dplm_150m \
  --tokenizer /private/dplm_150m \
  --evaluation-model /private/esm2_t33_650M_UR50D.pt \
  --require-stage protein-experiment
```

Arguments may be omitted on an inventory run; missing prerequisites are reported
with stable reason codes. The command checks credential and project-variable
presence and basic format, never records their values, and performs no network
or provider operation. It also validates the private checkpoint directory. A
zero exit means all locally inspectable prerequisites for the protein experiment
are present and valid. It is not evidence that QBoson, either A800 host, or the
acceptance workload has run.

The report separates provider-smoke readiness from system and protein readiness.
The first requires the standalone SDK approval but not the unfinished protein
configuration. Later stages additionally require that the approval embedded in
the frozen configuration exactly matches that standalone record and that the
current process observes an NVIDIA A800 at `cuda:0`. Select
`--require-stage provider-smoke` before section 5, `system-probe` before section
6, or the default `protein-experiment` before section 9. The process exit status
tracks the selected stage while the JSON always reports all three gates.

## Current environment facts

As measured on 2026-10-05, both hosts expose NVIDIA A800-SXM4-80GB devices with
driver 580.126.20 and system Python 3.10.12. The existing
`flagquantum/flagtree:0.7.0-validation` image exposes one requested A800 but
contains Python 3.12.3, Torch 2.13.0+cu129, CUDA runtime 12.9, and no Kaiwu SDK.

That image is suitable only for the fake-transport development rehearsal. It
must not be used as evidence for the frozen Python 3.10, Torch 2.7, NumPy 2.2.6,
and Kaiwu 1.3.1 acceptance lane.

## 1. Freeze the experiment before execution

Copy `examples/qdiffusion_kaiwu/acceptance_config.example.json` into a private
evidence directory. Replace every placeholder, including full revisions and
dataset/checkpoint SHA-256 digests, before running a baseline or guided sample.
Do not change thresholds after seeing results.

Separately copy `examples/qdiffusion_kaiwu/sdk_approval.example.json` to
`/absolute/private-evidence/sdk-approval.json`. Populate it only from the
retained organizational review decision, keep it mode `0600`, and copy the
same approval object into `acceptance_config.json` under `kaiwu_sdk`. The
standalone record lets the earlier live smoke enforce the rights decision
without prematurely requiring the protein artifacts; the frozen config carries
the same decision into every later QDiffusion stage and final evidence.

The `kaiwu_sdk` section is an executable rights gate, not a self-approval form.
Populate its review timestamp and approval reference only from the retained
organizational decision. Set each use-approval field to JSON `true` only when
that exact use has been approved. The fixed `no-sdk-redistribution` policy
allows distribution of the FlagQuantum adapter, not the Kaiwu wheel. The
validator binds the frozen Kaiwu version and artifact SHA-256 to the exact
Kaiwu distribution in the environment lock; changing either requires a new
reviewed configuration.

Use mode `0700` for evidence and checkpoint directories:

```bash
install -d -m 700 /absolute/private-evidence /absolute/private-kaiwu-checkpoints
cp examples/qdiffusion_kaiwu/acceptance_config.example.json \
  /absolute/private-evidence/acceptance_config.json
cp examples/qdiffusion_kaiwu/sdk_approval.example.json \
  /absolute/private-evidence/sdk-approval.json
chmod 600 \
  /absolute/private-evidence/acceptance_config.json \
  /absolute/private-evidence/sdk-approval.json
```

Every quota-consuming CLI rejects a relative, missing, symlinked, or
group/other-accessible checkpoint directory before it resolves credentials.
The SDK client repeats that private-directory check before it initializes the
license. Recovery receipts are synced to a private temporary file and atomically
published without replacement; resume rejects public, non-regular, or symlinked
receipt files before any SDK task operation. Receipt reads and writes remain
anchored to the opened private checkpoint directory and reject a replaced
parent binding rather than following it. A receipt read retains its no-follow
leaf descriptor through strict parsing, then checks file metadata and visible
inode binding; duplicate keys, non-standard nonfinite JSON constants, and
same-content leaf replacement all fail without task submission. The client also
freezes the checkpoint directory device/inode and rechecks it around every
vendor operation. If that identity changes during a call, preserve the original
directory and recovery bundle, classify the attempt as indeterminate, and do
not retry automatically: the provider may already have received the task.

Review the completed JSON and record its digest:

```bash
sha256sum /absolute/private-evidence/acceptance_config.json
```

Set `FROZEN_REQUESTED_SAMPLES` from the reviewed JSON value before invoking any
provider command; do not rely on a CLI default. For the illustrative template:

```bash
export FROZEN_REQUESTED_SAMPLES=10
```

Review the exported value against the frozen file in the same shell session.
Changing it does not change the experiment: every QDiffusion live entrypoint
and final validation require it to equal
`acceptance_config.json.requested_samples`; the earlier smoke record is also
cross-checked against that value during final validation.

The config must designate one primary host and the other portability-replay
host. It must contain at least three fixed seeds and a positive remote-call
budget. It must also retain the approved Kaiwu rights decision described above;
public wheel availability or a checked agreement URL cannot satisfy that gate.
The config, SDK approval, and environment lock readers anchor each file name to
an opened owner-only parent descriptor, refuse parent or leaf symlinks and
group/other permissions, bound input size, and recheck the visible path binding
after reading. Keep all three files in the private evidence directory; copying
a template without correcting its mode fails closed.

## 2. Verify transferred inputs

Transfer only after approval. On each host, compare every archive against the
reviewed transfer manifest before extraction. Reject missing, extra, or
mismatched artifacts. Extract into a new private directory; do not overwrite a
previous run.

Build the reviewed source bundle from clean checkouts before transfer. The
builder requires the pinned plugin and Community revisions, creates a new mode
0700 directory, writes mode-0600 archives and manifest, and self-verifies for
both target aliases. Manifest digests come from stable no-follow snapshots of
the freshly generated archives; the builder retains those three snapshots and
the manifest snapshot until both self-verification passes complete, so a
same-content replacement during review fails the build:

```bash
python -B -s -m examples.qdiffusion_kaiwu.build_transfer_bundle \
  --flagquantum-root /absolute/src/FlagQuantum \
  --plugin-root /absolute/src/kaiwu-pytorch-plugin \
  --community-root /absolute/src/kaiwu_community \
  --output-dir /absolute/private/flagquantum-qboson-transfer
```

Do not add another `.tar.gz` file to that directory after review. The host-side
verifier requires the colocated archive set to equal the manifest exactly.

Place the three archives next to the reviewed manifest and verify them before
using `tar` or another extraction tool:

```bash
mkdir -m 700 /absolute/private-evidence
python -B -s -m examples.qdiffusion_kaiwu.verify_transfer_bundle \
  --manifest /absolute/transfer/flagquantum-qboson-a800-bundle.manifest.json \
  --target-host jp-a800-171 \
  --output /absolute/private-evidence/transfer-preflight.json
```

Repeat with `jp-a800-172` on the other host. The verifier requires the exact
three reviewed artifact roles, checks each revision-derived filename and
SHA-256 digest, binds each archive's internal root to its declared revision, and
scans gzip-tar members without extraction. Absolute or parent-traversing names,
duplicate names, links, devices, FIFOs, and other special members fail closed.
The manifest must be a mode-0600 regular file in an owner-only real directory
and is read through that directory descriptor with a 4 MiB bound. Extracted-tree
verification captures it again through the same gate and requires the digest to
remain identical to the bundle-verification pass before parsing it.
Each mode-0600 source archive is likewise opened once relative to that private
directory: SHA-256 and tar-member inspection consume the same descriptor, whose
metadata and visible path binding are rechecked afterwards. The exact colocated
archive-name set is also rechecked at the end of the bundle pass.
Its mode-0600 output records the machine hostname and target alias but is only
transfer-preflight evidence; it proves neither A800 execution nor QBoson use.

Extract all three reviewed archives into one new, otherwise empty, private
directory. Before importing or running any extracted source, use `-B` to prevent
bytecode creation and bind every extracted file to the reviewed archives:

```bash
cd /absolute/private/extracted/FlagQuantum-REVIEWED_REVISION_PREFIX
python -B -s -m examples.qdiffusion_kaiwu.verify_extracted_bundle \
  --manifest /absolute/transfer/flagquantum-qboson-a800-bundle.manifest.json \
  --extraction-root /absolute/private/extracted \
  --target-host jp-a800-171 \
  --output /absolute/private-evidence/extraction-preflight.json
```

Repeat with the other target alias. The extracted-tree verifier rejects changed,
missing, extra, linked, or special filesystem entries, and records the exact
manifest and source revisions. It reopens every archive through the private
directory descriptor, rechecks its manifest SHA-256, and uses that same file
description for extracted-content comparison; changing an archive after the
initial bundle pass therefore fails closed. Each extracted regular file is
hashed from a no-follow descriptor, and the complete file/directory set plus
inode and content-relevant metadata are rechecked before the record is emitted.
The same stable-tree implementation is used again by every execution entrypoint
when it recomputes the FlagQuantum and plugin roots from this preflight; runtime
source validation cannot fall back to ordinary path reads.
Preserve both preflight records and the bundle manifest with the run evidence.
These records remain preflight-only evidence.
Public Kaiwu Community source is conformance input, not a substitute for the
proprietary SDK.

## 3. Run the development-only A800 rehearsal

This stage uses the existing validation image and an in-memory fake transport.
It checks proposal forward, conditioned Boltzmann sampling through
`KaiwuSampler`, the energy objective, backward, an optimizer update, and one
guided generation step on the observed A800.

Run `examples/qdiffusion_kaiwu/run_a800_development_probe.sh` from the reviewed
local checkout with absolute transfer-directory, extraction-preflight, and
local output paths. The local gate revalidates the bundle and requires the
extraction-preflight record to bind the supplied source and plugin revisions to
the target alias before opening SSH. Supply the stable SSH alias as
`EXECUTION_HOST` and the separately observed machine hostname as
`EXPECTED_HOSTNAME`. The runner streams all reviewed inputs to standard input of
an auto-removed container. It disables networking and container logging,
exposes only GPU 0, uses a read-only root filesystem, places its input,
extracted workspace, and remote evidence only in tmpfs, and uses no host bind
mount. The remote JSON is streamed back through standard output and is written
only to one exclusive local mode-0600 record after a second fail-closed
validation. No source or evidence file is written to the validation host.
Pass the host's full reviewed `sha256:...` image ID as the final argument. Do
not use the mutable `flagquantum/flagtree:0.7.0-validation` tag directly: the
tag currently resolves to different image IDs on `jp-a800-171` and
`jp-a800-172`. The runner verifies and records the exact ID before execution.

The record must retain all of these values:

- `evidence_class=development_fake_transport`;
- `transport=in_memory_fake`;
- `qboson_hardware_used=false`;
- `real_provider_evidence=false`;
- `system_acceptance=false`.

Never promote this record into the live acceptance manifest.

Revalidate each retained local development record offline before relying on
its summary or digest. This reads only private local files and performs no SSH,
container, credential, or provider operation:

```bash
python -B -s -m examples.qdiffusion_kaiwu.stream_development_evidence validate-record \
  --execution-host jp-a800-171 \
  --expected-hostname bm-baai-dx-zone1-lc-a800-80g-15-171 \
  --source-revision FULL_FLAGQUANTUM_REVISION \
  --plugin-revision f047bce7b1077449967bbe9e9fab5741542b48d4 \
  --source-preflight /absolute/private-evidence/jp-a800-171-extraction-preflight.json \
  --validation-image-id sha256:FULL_REVIEWED_IMAGE_DIGEST \
  --transfer-manifest /absolute/private-evidence/flagquantum-qboson-a800-bundle.manifest.json \
  --record /absolute/private-evidence/jp-a800-171-development.json
```

Repeat with the second host's hostname, preflight, image ID, and record. The
command prints the validated development-record SHA-256 only after its source
preflight and retained transfer manifest hash chain match.

## 4. Establish the approved SDK lane

Create an isolated Python 3.10 environment from the approved wheel set. Verify
package files and versions before setting credentials. Do not install an
unreviewed package merely because it shares the name `kaiwu`.

After separately reviewing every installation artifact's source, license, and
terms, place exactly one wheel for every installed distribution in a private
wheelhouse. Build the mode-0600 lock from that complete set while the isolated
environment is active. Shell expansion is safe only when that directory
contains the exact reviewed set; extra, missing, duplicate, or version-mismatched
wheels fail closed. The evidence parent must already be a private mode-0700 real
directory:

```bash
python3 -B -s -m examples.qdiffusion_kaiwu.build_environment_lock \
  --artifact /absolute/private-reviewed-wheelhouse/*.whl \
  --output /absolute/private-evidence/environment-lock.json
```

The builder reads bounded wheel METADATA and obtains the wheel digest from the
same stable no-follow snapshot, retaining every reviewed wheel snapshot through
lock publication. It also hashes every file in each installed distribution's
RECORD set and cross-revalidates the complete captured set, without installing
or executing the wheel. A replacement during metadata parsing, inventory
collection, or record publication fails closed. `environment_lock.example.json`
documents the schema only; its three illustrative rows are not a complete lock.
Independently verify the generated lock offline before credentials are present:

```bash
python3 -B -s -m examples.qdiffusion_kaiwu.verify_environment_lock \
  --lock /absolute/private-evidence/environment-lock.json
```

The builder and verifier reject a different Python patch version, missing or
additional distributions or wheels, version drift, duplicate names,
placeholders, missing or changed installed files, symlinks, and non-private
lock permissions. They do not download, install, or approve packages; artifact
review and installation remain separate controlled steps.
Record the verified lock's SHA-256 as
`software.environment_lock_sha256` in `acceptance_config.json` before freezing
the configuration. Every live, training, evaluation, and replay command below
must receive that same private lock file.

At minimum, record:

```bash
python3 --version
python3 -c 'import numpy, torch; print(numpy.__version__, torch.__version__)'
python3 -c 'import kaiwu; print(kaiwu.__version__)'
```

The observed values must exactly match `acceptance_config.json`. A `+cu...`
Torch build suffix is part of the observed version and must not be silently
discarded. The live client also checks `license.init`,
`CheckpointManager.save_dir`, and callable `cim.CIMOptimizer` on that same
module before credentials are resolved. Importing FlagQuantum without Kaiwu
installed must continue to work in the normal environment.

## 5. Run the guarded provider smoke test

Use a fresh task prefix and the assigned project. The following command submits
one optimization task and one sampling task and may consume quota:

Both identifiers must be nonempty printable text without control or format
characters. Receipt restoration and final validation also reject surrounding
whitespace in task, project, provider-task, and provider-target identities.

```bash
python3 -B -s -m examples.qdiffusion_kaiwu.qboson_live_smoke \
  --checkpoint-dir /absolute/private-kaiwu-checkpoints \
  --environment-lock /absolute/private-evidence/environment-lock.json \
  --sdk-approval /absolute/private-evidence/sdk-approval.json \
  --output /absolute/private-evidence/qboson-smoke-attempt-001.json \
  --project-no "$QBOSON_PROJECT_NO" \
  --task-prefix "flagquantum-smoke-${RUN_ID}" \
  --expected-sdk-version 1.3.1 \
  --requested-samples "$FROZEN_REQUESTED_SAMPLES" \
  --acknowledge-provider-cost I_ACKNOWLEDGE_QBOSON_QUOTA_USAGE
```

Before credential discovery, the smoke command requires the private approval
record, validates its exact field set and four explicit approvals, and binds
its Kaiwu version and wheel SHA-256 to the exact environment-lock distribution.
The record digest is retained in smoke evidence. A missing, public, symlinked,
placeholder, mismatched, or unapproved record fails without initializing the
SDK license or consuming quota.

`QBOSON_PROJECT_NO` is not a credential, but it should still be managed in the
private run environment. The command has no simulator fallback. It writes the
diagnostic record and exits nonzero with hardware acceptance closed if the
pinned SDK mapping cannot supply stable provider task and target identities.
Only the command's real SDK path records `transport=kaiwu_cim`,
`real_provider_evidence=true`, and `qboson_hardware_used=true`; injected test
clients cannot produce hardware acceptance.

The project and task prefix must be nonempty. The output path must be absolute,
its parent must already be a private real directory, and the output itself must
not exist. These conditions are checked before credential resolution and again
at exclusive publication; a raced or accidentally reused filename is never
overwritten after quota has been consumed. Publication remains anchored to one
non-symlink directory descriptor and rolls back the candidate record if the
visible parent directory is replaced during the write.

The smoke record includes a redacted `provider_result_schema` from the SDK's
documented `get_task_result` dictionary. It contains field names and structural
metadata only—never task values, result strings, credentials, or raw vendor
errors. Use it to review candidate task-ID and target mappings against an
approved SDK response; do not promote acceptance based only on a suggestive key
name.

Stop here if authentication, quota, provider status, spin validation, energy
recomputation, task identity, or target identity is unresolved. Do not move to
QDiffusion by replacing the provider with a local solver.

## 6. Run one live QDiffusion system attempt

Run this command inside the exact frozen lane on one observed A800. If a
container is used, set its hostname to the reviewed host hostname so the
recorded identity is not a random container ID.

```bash
python3 -B -s -m examples.qdiffusion_kaiwu.qdiffusion_system_live \
  --config /absolute/private-evidence/acceptance_config.json \
  --checkpoint-dir /absolute/private-kaiwu-checkpoints \
  --output /absolute/private-evidence/system-attempt-001.json \
  --execution-host jp-a800-171 \
  --expected-hostname "$EXPECTED_MACHINE_HOSTNAME" \
  --source-revision "$FLAGQUANTUM_REVISION" \
  --plugin-revision "$KAIWU_PLUGIN_REVISION" \
  --plugin-root /absolute/src/kaiwu-pytorch-plugin \
  --source-preflight /absolute/private-evidence/extraction-preflight.json \
  --environment-lock /absolute/private-evidence/environment-lock.json \
  --project-no "$QBOSON_PROJECT_NO" \
  --task-prefix "flagquantum-qdiffusion-${RUN_ID}" \
  --device cuda:0 \
  --expected-sdk-version 1.3.1 \
  --requested-samples "$FROZEN_REQUESTED_SAMPLES" \
  --acknowledge-provider-cost I_ACKNOWLEDGE_QBOSON_QUOTA_USAGE
```

The command validates the frozen config and A800 before resolving credentials.
It also recomputes the executing FlagQuantum tree and the supplied plugin tree
against the post-extraction preflight, places that reviewed source on the import
path, and rejects a preloaded or installed QDiffusion module from another root.
It uses explicit integer precision, a hard remote-call budget, and no fallback.
It records every distinct original-matrix precision report, all returned task
receipts, the training update, generation constraints, and repeat retrieval of
the last task.

Final assembly and independent revalidation reconcile those system receipts
against the remote-call count, matrix digests, requested sample count,
`qboson_task_ids`, and the single provider target. They also require the exact
SDK client, successful run, real-provider/QBoson flags, no fallback, and no
resubmission; summary task fields without their matching receipts fail closed.

It also records the explicit host/device boundary: the energy model's CUDA
device, the plugin-produced CPU NumPy Ising matrices, FlagQuantum's canonical
CPU float64 matrices and submitted host tuples, returned CPU int8 samples,
cache hits, and the CUDA target used when the plugin reconstructs tensors. The
number of non-cached transfer records must equal the remote-call count.

The current adapter deliberately fails the system gate when provider task or
target fields are unavailable. Do not edit the output to make it pass. Inspect
an approved, redacted real SDK response and update the provider mapping with
tests instead.

## 7. Interrupt and resume safely

A local timeout or interruption must not create a replacement provider task.
The Kaiwu SDK client writes a credential-free recovery bundle before its first
SDK task operation and uses the documented `task_name + ising_matrix` identity.

To resume:

1. preserve the exact checkpoint directory, frozen config, source revisions,
   project, task prefix, sample count, precision policy, and seed;
2. choose a new evidence output filename because records are immutable and the
   writer refuses to overwrite them;
3. rerun the same command with every task-identity input unchanged;
4. verify that restored receipts keep their original `submitted_at` and matrix
   digest;
5. retain both the failed attempt record and resumed record.

The separately saved job receipt is not a replacement for the SDK checkpoint
directory. Every solve, status, and result operation reopens the deterministic
recovery bundle there and requires an exact receipt match before contacting the
SDK. Missing or edited checkpoint state must fail; never create a replacement
receipt to continue a run.

Do not delete or edit recovery bundles. Do not change the task prefix to bypass
an identity conflict. Do not infer that timeout means cancellation. A corrupt
or conflicting bundle must fail before SDK access and requires investigation,
not automatic resubmission.

## 8. Repeat independently on the other A800 host

Run the same frozen system configuration on the second host with a distinct
task prefix, checkpoint directory, output file, and provider task identities.
Use `jp-a800-172` and its observed machine hostname when the first host is the
primary, or reverse the roles exactly as frozen in the config.

Both hosts must independently observe `cuda:0` and an NVIDIA A800. Sharing a
QBoson task ID between host records is forbidden. These runs demonstrate
portability across two environments, not distributed execution.

## 9. Run the frozen protein experiment

Only the configured primary host runs every frozen seed for the full DPLM
baseline-versus-guided experiment. The second host runs one fixed guided replay.

For every seed, retain:

- baseline and guided inputs, output sequences, and failure state;
- the exact dataset split, checkpoint, tokenizer, and generation identities;
- every QBoson receipt, including timed-out or failed attempts;
- ESM2 mean and median cosine and L2 distances;
- uniqueness, repeat ratio, validity, and length compliance;
- identity and Jensen-Shannon metrics only under their frozen definitions.

The primary decision is lower mean ESM2 cosine distance. Guided generation must
improve it, contain no invalid sequence, retain at least 95 percent of baseline
uniqueness, and increase repeat ratio by no more than 0.05 absolute. A failed
effectiveness result is a valid experiment result and must not be relabeled as
a provider transport failure or followed by post-hoc threshold changes.

The dataset and checkpoint are not currently supplied by this repository.
Their acquisition, licenses, hashes, and frozen revisions must be resolved
before this step; the upstream example's published metrics are not FlagQuantum
evidence.

At plugin revision `f047bce7b1077449967bbe9e9fab5741542b48d4`, the Git tree
contains no FASTA or model checkpoint even though its default helpers reference
`data/UP000005640_9606.fasta`, `airkingbd/dplm_150m`, a local
`/data2/wwx/models/dplm_150m` path, and `ckpt/best_epoch_9.pt`. The ESM2 loader
also resolves `esm2_t33_650M_UR50D` outside the plugin tree. Treat each as an
external artifact: stage it locally, review its license, freeze its revision and
content digest, and prohibit implicit downloads during the acceptance run.

The frozen base DPLM checkpoint is loaded independently for the proposal and
energy backbones in the first acceptance scope. The guided energy checkpoint is
then produced by the frozen training procedure. Record its digest in both host
records; the replay host must load that exact primary-host artifact.

### 9.1 Protein asset intake status

The following intake review was refreshed on 2026-10-06. It identifies
candidate sources only; it does not authorize acquisition, redistribution, or
use. No dataset or model file was downloaded during this review.

| Artifact | Candidate source and observed state | Intake decision |
| --- | --- | --- |
| Human proteome FASTA | [UniProt REST stream](https://rest.uniprot.org/uniprotkb/stream?compressed=false&format=fasta&query=%28proteome%3AUP000005640%29); UniProt publishes [CC BY 4.0 license information](https://www.uniprot.org/help/license). | Source and license evidence identified. Acquisition still requires explicit authorization, a frozen revision or retrieval identity, a content digest, and a completed review timestamp. |
| DPLM 150M checkpoint and tokenizer | [`airkingbd/dplm_150m`](https://huggingface.co/airkingbd/dplm_150m) identifies itself as the 150M checkpoint and links the official implementation. Its `main` reference and public metadata resolved to candidate commit `49b7125a5d28c6418fcc2f3c4fe799352ac1488b` on 2026-10-06. The seven-file inventory contains checkpoint and tokenizer files but no license file, and the metadata has no license field. The linked [implementation repository](https://github.com/bytedance/dplm) is Apache-2.0, but that notice is not treated as an authoritative license for the separately hosted model bytes. | Blocked. Do not download, stage, or fill the checkpoint/tokenizer `license_id` fields until the model publisher or an approved organizational review explicitly resolves the weights and tokenizer rights. |
| ESM2 evaluation model | [`facebook/esm2_t33_650M_UR50D`](https://huggingface.co/facebook/esm2_t33_650M_UR50D) declares MIT. The configured binary source remains the Meta-hosted checkpoint URL. | Source and license evidence identified. Acquisition still requires explicit authorization, an immutable revision or release identity, a content digest, and a completed review timestamp. |

Before downloading any approved artifact, replace mutable branch names with an
immutable source revision or release identity and record the exact expected
files. A short web-interface commit label is not sufficient by itself. The
download must occur outside the acceptance containers, and the resulting local
artifacts must pass the offline preflight below before any credential or quota
lookup. Public availability is not approval.

Before either host run, verify the four staged inputs offline. File artifacts
use ordinary SHA-256; directory snapshots use the path-aware
`tree-sha256-v1` algorithm implemented by the preflight tool. The tool rejects
relative paths, symlinks, special files, digest mismatches, existing output
records, and missing, public, or symlinked output parents:

```bash
python -B -s -m examples.qdiffusion_kaiwu.preflight_protein_artifacts \
  --config /absolute/evidence/acceptance-config.json \
  --dataset /absolute/artifacts/UP000005640_9606.fasta \
  --base-checkpoint /absolute/artifacts/dplm_150m \
  --tokenizer /absolute/artifacts/dplm_150m \
  --evaluation-model /absolute/artifacts/esm2_t33_650M_UR50D \
  --output /absolute/evidence/artifact-preflight.json
```

Run with network access disabled. The mode-0600 output deliberately declares
itself preflight-only and is not evidence of a successful QBoson run. It also
parses the frozen FASTA, rejects empty or duplicate records and unsupported
residue symbols, applies the declared length filter and record cap, and proves
that the deterministic validation/test split produces exactly the configured
generation sequence count.
File identities and directory-tree identities use the same stable no-follow
snapshot layer as reviewed source verification. FASTA profiling reopens the
captured dataset inode and rechecks both the leaf and parent identity after
parsing, so the dataset digest and corpus profile cannot come from different
path contents.

The frozen configuration includes every plugin knob that changes the selected
corpus or generated sequences: record-length bounds, record cap, validation and
test ratios, proposal/noise/energy temperatures, candidate count, generation
steps, resampling policy, QBoson samples per Ising request, ESM2
pairing/pooling, and evaluation batch size. The current values select 640
eligible records, 32 test sequences at a 0.05 test ratio, and 10 returned spins
per provider request. Every live command's `--requested-samples` must equal the
frozen value. Changing any of these values creates a different experiment and
requires a new preregistered configuration identity.

The pinned plugin's complete workflow imports `build_qdiffusion` once and then
uses it for structural validation, training, proposal-only baseline, and guided
generation. Invoke that workflow inside
`bound_qdiffusion_workflow(workflow_module, kaiwu_sampler)`. The context wraps
and later restores the imported factory, removes the workflow's local `sa`/`cim`
construction hints, injects the same bounded `KaiwuSampler` into every
generator, and fails if a returned energy model does not retain that exact
sampler. Use a single non-concurrent process. Merely setting
`sampler_type="cim"` does not test the FlagQuantum `remote/kaiwu` boundary and
is outside this acceptance lane.

Run each configured seed separately on the declared primary host. The launcher
validates the A800/software lane, performs artifact preflight before credentials
are resolved, maps every frozen knob into the plugin dataclasses, injects the
FlagQuantum sampler into the complete workflow, and records the best trained
energy-checkpoint digest:

```bash
python -B -s -m examples.qdiffusion_kaiwu.qdiffusion_protein_training_live \
  --config /absolute/evidence/acceptance-config.json \
  --plugin-root /absolute/src/kaiwu-pytorch-plugin \
  --dataset /absolute/artifacts/UP000005640_9606.fasta \
  --base-checkpoint /absolute/artifacts/dplm_150m \
  --tokenizer /absolute/artifacts/dplm_150m \
  --evaluation-model /absolute/artifacts/esm2_t33_650M_UR50D.pt \
  --artifact-preflight-output /absolute/evidence/seed-1701-preflight.json \
  --sdk-checkpoint-dir /absolute/private/kaiwu-checkpoints \
  --workflow-output-root /absolute/private/qdiffusion-runs \
  --run-record /absolute/evidence/seed-1701-training.json \
  --execution-host jp-a800-171 \
  --expected-hostname bm-baai-dx-zone1-lc-a800-80g-15-171 \
  --source-revision FULL_FLAGQUANTUM_REVISION \
  --plugin-revision f047bce7b1077449967bbe9e9fab5741542b48d4 \
  --source-preflight /absolute/evidence/extraction-preflight.json \
  --environment-lock /absolute/evidence/environment-lock.json \
  --project-no APPROVED_PROJECT \
  --task-prefix qdiffusion-protein \
  --seed 1701 \
  --expected-sdk-version 1.3.1 \
  --requested-samples "$FROZEN_REQUESTED_SAMPLES" \
  --acknowledge-provider-cost I_ACKNOWLEDGE_QBOSON_QUOTA_USAGE
```

Repeat with a new exclusive preflight and training-record path for every frozen
seed. A training record is not system or application acceptance; do not promote
it until ESM2 evaluation and both host gates pass. Each completed training
component must also prove that its sampler is bound to the exact pinned SDK
client, contain at least one sampling receipt with provider task and target
identities, retain complete precision evidence, and stay within its per-seed
call budget. The assembler and independent final validator reject injected
clients, zero-call records, missing identities, fallback, and incomplete receipt
or precision sets even if the workflow artifacts and hash links are otherwise
valid. They also require the exact frozen software lane, primary-host A800 and
`cuda:0` request, system and per-seed budgets, a positive conservative call
bound within quota, safe run/checkpoint names, artifact-preflight and checkpoint
digests, and the fixed seven-item held-out/baseline/guided/history/metrics/quality
artifact set with exact relative paths and content hashes. The ESM2 evaluator
and replay-host runner apply the provider checks while loading the selected
training record, before loading a model, allocating A800 work, resolving
credentials, or submitting another task.

Provision `--workflow-output-root` in advance as an owner-only real directory;
the launcher validates it before resolving credentials. The plugin runs with a
temporarily enforced `umask 077`, which is restored on every exit, and must
create exactly one owner-only, real, non-symlink run directory directly under
the seed output root. The launcher captures the selected best checkpoint and
the seven declared workflow artifacts with owner-only, no-follow stable
snapshots and cross-revalidates all eight before writing their identities into
the training record. A public or symlinked output, an output outside that
directory, or a file replaced while the output set is being captured fails the
component. Those exact eight snapshots remain live across frozen-input
postflight and are checked again immediately before the exclusive training
record is published; a change during postflight closes the run through its
existing redacted failure fields.

The exclusive preflight returns the exact stable no-follow snapshots used to
produce its record; the launcher does not reopen the paths to create a second
identity. Training and replay retain the dataset, base checkpoint, tokenizer,
and evaluation-model snapshots, revalidate all four before and after plugin
consumption, and check them again during postflight. Consequently an input
rewritten or replaced during either run fails the component even when the
replacement has the same bytes and would pass a later digest-only check.

For training evidence, those provider checks reconcile the ordered task-ID
list and single provider-reported target with every receipt; require unique
matrix identities, the frozen sample count, and aware submission timestamps;
and validate the retained precision-policy scale and error ranges. Recording
only a receipt count or precision-report count is not sufficient.

Do not reuse the bounded system probe's `remote_call_budget` for this command.
The plugin invokes the sampler once per conditioned example for positive energy,
once per negative candidate, and once per generated candidate per step. With
the illustrative 640-record, 20-epoch, four-candidate, 64-step configuration,
the conservative bound is 71,269 submissions per seed, including the objective
call that precedes each baseline and guided generation. Therefore
`training.remote_call_budget_per_seed` is deliberately left as a required value:
resize and preregister the experiment or secure matching project quota before
running. The validator rejects a budget below the bound; deduplication may lower
actual submissions but must not be assumed during quota planning.

After a seed completes, evaluate its already-generated held-out outputs; do not
generate them again. The evaluator recomputes every training-artifact digest,
requires aligned FASTA headers and the exact frozen count, and loads the ESM2
checkpoint only through the local-file API. System, training, evaluation, and
replay recompute the executing FlagQuantum tree plus the actual `--plugin-root`
file count and content-set digest against the host extraction preflight before
importing the plugin workflow. The system probe, protein training, and replay
also verify the resolved `kaiwu.torch_plugin` and QDiffusion module files are
inside that root. Every loaded `kaiwu.torch_plugin.*` and `dplm.*` transitive
module must resolve inside the corresponding reviewed plugin subtree:

Keep the selected training run directory and its files owner-only until both
consumers finish. Evaluation and replay reject a public or symlinked run
directory, public consumed artifacts, and a public trained checkpoint even when
their byte digests still match. Evaluation completes the run-directory,
artifact, and frozen ESM2 checks before initializing `cuda:0`; replay completes
the training-directory and checkpoint checks before credential resolution.

The evaluator retains a stable no-follow snapshot for every consumed training
FASTA and quality JSON. Plugin FASTA reads are bracketed by identity checks,
quality JSON is parsed from the captured descriptor, and the complete set is
rechecked after metrics and again immediately before the evaluation record is
published. Replay retains the same snapshots around plugin consumption. Its
trained energy checkpoint is likewise captured after matching the training
record, checked immediately before and after the plugin weight loader, and
rechecked after generation. Replay's final postflight ends with another check
of all four frozen input snapshots immediately before publication. Rehashing
later is not used as a substitute for this in-run identity binding.
The ESM2 checkpoint follows the same rule: the snapshot whose digest matches
the frozen config is passed into the evaluator, brackets
`load_model_and_alphabet_local`, and remains unchanged through metric
calculation.

Training revalidates the retained stable snapshots and rehashes the frozen
config and all four input roles after the workflow returns. Replay repeats that
check and also revalidates the transferred test FASTA and trained checkpoint.
Evaluation revalidates its ESM2 checkpoint and all consumed training outputs
after metrics are computed. Preserve the original inputs unchanged until each
command exits; a component without `artifact_inputs_unchanged=true` cannot
enter the final acceptance bundle.

```bash
python -B -s -m examples.qdiffusion_kaiwu.qdiffusion_protein_evaluate \
  --config /absolute/evidence/acceptance-config.json \
  --training-record /absolute/evidence/seed-1701-training.json \
  --run-directory /absolute/private/qdiffusion-runs/seed-1701/RUN_DIRECTORY \
  --plugin-root /absolute/src/kaiwu-pytorch-plugin \
  --evaluation-model /absolute/artifacts/esm2_t33_650M_UR50D.pt \
  --output /absolute/evidence/seed-1701-evaluation.json \
  --execution-host jp-a800-171 \
  --expected-hostname bm-baai-dx-zone1-lc-a800-80g-15-171 \
  --source-revision FULL_FLAGQUANTUM_REVISION \
  --plugin-revision f047bce7b1077449967bbe9e9fab5741542b48d4 \
  --source-preflight /absolute/evidence/extraction-preflight.json \
  --environment-lock /absolute/evidence/environment-lock.json \
  --expected-sdk-version 1.3.1
```

This stage uses the A800 but consumes no additional QBoson quota. Preserve one
evaluation record per seed; final application metrics must be assembled across
the complete preregistered seed set rather than selected post hoc. Assembly and
independent validation require each evaluation component to match the frozen
FlagQuantum/plugin/Python/Torch/environment lane, primary host, NVIDIA A800
device, and exact ESM2 checkpoint digest. They also require bounded finite
metrics, zero invalid sequences, secret redaction, and an explicit statement
that this local evaluation consumed no provider quota and is candidate evidence
only.

On the configured replay host, load the preregistered seed's exact best energy
checkpoint and the training run's hashed test FASTA. The fixed replay performs
one objective call and three guided steps with the frozen four candidates, for
a conservative 17-call budget:

```bash
python -B -s -m examples.qdiffusion_kaiwu.qdiffusion_portability_replay_live \
  --config /absolute/evidence/acceptance-config.json \
  --plugin-root /absolute/src/kaiwu-pytorch-plugin \
  --dataset /absolute/artifacts/UP000005640_9606.fasta \
  --base-checkpoint /absolute/artifacts/dplm_150m \
  --tokenizer /absolute/artifacts/dplm_150m \
  --evaluation-model /absolute/artifacts/esm2_t33_650M_UR50D.pt \
  --artifact-preflight-output /absolute/evidence/replay-preflight.json \
  --training-record /absolute/evidence/seed-1701-training.json \
  --training-run-directory /absolute/transferred/seed-1701/RUN_DIRECTORY \
  --trained-checkpoint /absolute/transferred/seed-1701/RUN_DIRECTORY/checkpoints/BEST.pt \
  --sdk-checkpoint-dir /absolute/private/kaiwu-checkpoints \
  --output /absolute/evidence/jp-a800-172-portability.json \
  --execution-host jp-a800-172 \
  --expected-hostname bm-baai-dx-zone1-lc-a800-80g-15-172 \
  --source-revision FULL_FLAGQUANTUM_REVISION \
  --plugin-revision f047bce7b1077449967bbe9e9fab5741542b48d4 \
  --source-preflight /absolute/evidence/replay-extraction-preflight.json \
  --environment-lock /absolute/evidence/environment-lock.json \
  --project-no APPROVED_PROJECT \
  --task-prefix qdiffusion-portability \
  --expected-sdk-version 1.3.1 \
  --requested-samples "$FROZEN_REQUESTED_SAMPLES" \
  --acknowledge-provider-cost I_ACKNOWLEDGE_QBOSON_QUOTA_USAGE
```

The runner verifies the checkpoint remains inside the recorded run directory,
recomputes its digest, verifies every referenced training artifact, and records
repeat retrieval without resubmission. Assembly and independent revalidation
also reconcile the replay's exact SDK client, A800 device, receipt count,
matrix digests, requested and returned samples, provider task IDs and target,
call budget, precision reports, fallback state, and token constraints. A replay
summary cannot pass after its underlying remote evidence is removed or changed.
This demonstrates portability between the two available A800 environments
only; it is not multi-node or distributed execution.

## 10. Assemble the immutable evidence bundle

After both system probes, the portability replay, and every configured
training/evaluation seed pass, assemble them without manually copying metrics:

```bash
python -B -s -m examples.qdiffusion_kaiwu.assemble_acceptance \
  --config /absolute/evidence/acceptance-config.json \
  --environment-lock /absolute/evidence/environment-lock.json \
  --primary-system /absolute/evidence/jp-a800-171-system.json \
  --replay-system /absolute/evidence/jp-a800-172-system.json \
  --primary-source-preflight /absolute/evidence/jp-a800-171-extraction-preflight.json \
  --replay-source-preflight /absolute/evidence/jp-a800-172-extraction-preflight.json \
  --transfer-manifest /absolute/evidence/flagquantum-qboson-a800-bundle.manifest.json \
  --sdk-approval /absolute/private-evidence/sdk-approval.json \
  --provider-smoke /absolute/private-evidence/qboson-smoke-attempt-001.json \
  --artifact-preflight /absolute/evidence/artifact-preflight.json \
  --portability /absolute/evidence/jp-a800-172-portability.json \
  --training-record /absolute/evidence/seed-1701-training.json \
  --training-record /absolute/evidence/seed-1702-training.json \
  --training-record /absolute/evidence/seed-1703-training.json \
  --evaluation-record /absolute/evidence/seed-1701-evaluation.json \
  --evaluation-record /absolute/evidence/seed-1702-evaluation.json \
  --evaluation-record /absolute/evidence/seed-1703-evaluation.json \
  --evidence-dir /absolute/final/qboson-qdiffusion-acceptance
```

The target directory must not exist. The assembler verifies component schemas,
config identities, both revision-bound extraction preflights, their exact shared
transfer manifest, host roles, seed coverage, training/evaluation links, and the
selected portability checkpoint before averaging metrics. Training components
are revalidated against the frozen software/A800 lane, both call budgets, safe
output names, preflight and checkpoint digests, and the exact seven workflow
artifact identities; a missing or substituted artifact fails assembly. It
copies every source record and the transfer manifest into a private component directory,
copies the exact environment lock as a top-level member, hashes those copies,
creates the two final host records and manifest, then runs
`validate_acceptance.py` on the result. Every input must be an absolute,
mode-0600 regular file rather than a symlink. Assembly occurs in a private
sibling staging directory and is atomically published only after final
validation; a failed run does not leave the requested evidence directory. Both
the initial and final absence checks use no-follow metadata: an existing entry,
including a dangling symlink created before or during validation, is preserved
and causes publication to fail.
Subsequent validation repeats the private-file check for the manifest, config,
environment lock, host records, and every component; it verifies that the lock
digest equals the frozen config and rejects a symlink anywhere in a member path
rather than following it. Member paths must be unique normalized relative POSIX
paths, and the on-disk file tree must exactly equal the manifest's declared set;
unlisted files are rejected.
Missing, extra, replaced, or selectively omitted source or seed records fail.
The two final host records also retain the frozen FlagQuantum package version
and an exact limitations list. Both must explicitly require the complete source
component bundle. Do not delete that flag or edit the limitations: validation
must reject a record that drops its development-only classification, conflates
the independent A800 runs with distributed or domestic-accelerator support, or
adds performance, production, quantum-advantage, or scalability claims.

## 11. Independently revalidate final evidence

The live-system probe record is an attempt record, not by itself the final
acceptance record. The assembler is the only supported path for joining
accepted system evidence, primary application metrics, and the replay result
into exactly two host records. Do not hand-edit its records or construct a
replacement manifest from `acceptance_manifest.example.json`.

Run the fail-closed validator from the same source revision:

```bash
python3 -B -s -m examples.qdiffusion_kaiwu.validate_acceptance \
  /absolute/final/qboson-qdiffusion-acceptance/manifest.json
```

The validator recomputes the decision. It rejects changed software or config,
missing host coverage, shared provider tasks, fake transport, CPU execution,
missing target identity, fallback, retrieval resubmission, invalid training or
generation values, incomplete precision evidence, altered thresholds, disabled
component-bundle validation, changed claim limitations, and a guided result
that misses the preregistered application criteria.

## Failure classification

| Observation | Classification | Next action |
| --- | --- | --- |
| Python, Torch, NumPy, plugin, or SDK mismatch | Environment failure | Rebuild the approved lane; do not submit |
| Missing or partial credentials | Preflight failure | Repair private environment; do not submit |
| Authentication or quota rejection | Provider control-plane failure | Preserve redacted record and coordinate account access |
| Local timeout with recovery bundle | Incomplete provider task | Resume the same identity |
| Corrupt or conflicting recovery bundle | Identity-integrity failure | Stop and investigate; do not resubmit |
| Invalid spin, shape, count, status, or energy | Provider/result-contract failure | Preserve evidence and stop |
| Missing provider task ID or target | Evidence mapping incomplete | Keep acceptance failed and review a redacted response |
| CPU tensor or non-A800 observation | System-path failure | Correct device execution; do not claim A800 validation |
| Finite system run but failed protein metric | Application-effectiveness failure | Report it without weakening thresholds |

Attempt records intentionally retain only one stable failure category and an
empty message. Do not add raw vendor, parser, model, artifact, or plugin
exception text to the evidence bundle; diagnose it in a separately approved
private session without weakening the immutable record.

## Claim boundary

A passing bundle establishes only the declared A800 plus QBoson QDiffusion
scope. It does not establish a general gate-model backend, differentiation
through remote sampling, domestic-accelerator support, distributed execution,
performance superiority, production maturity, quantum advantage, or a named
QBoson product unless that exact product is provider-reported and reviewed.
