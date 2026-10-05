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

## Required inputs

Do not begin a live run until all entries are available and reviewed:

- full FlagQuantum and Kaiwu PyTorch Plugin Git revisions;
- an approved Python 3.10 environment with the exact Torch, NumPy, plugin, and
  Kaiwu SDK versions frozen in `acceptance_config.json`;
- the proprietary Kaiwu wheel, its SHA-256 digest, source, license, and approved
  redistribution boundary;
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

Use mode `0700` for evidence and checkpoint directories:

```bash
install -d -m 700 private-evidence private-kaiwu-checkpoints
cp examples/qdiffusion_kaiwu/acceptance_config.example.json \
  private-evidence/acceptance_config.json
```

Review the completed JSON and record its digest:

```bash
sha256sum private-evidence/acceptance_config.json
```

The config must designate one primary host and the other portability-replay
host. It must contain at least three fixed seeds and a positive remote-call
budget.

## 2. Verify transferred inputs

Transfer only after approval. On each host, compare every archive against the
reviewed transfer manifest before extraction. Reject missing, extra, or
mismatched artifacts. Extract into a new private directory; do not overwrite a
previous run.

Confirm the source identities from the extracted trees and preserve the bundle
manifest with the run evidence. Public Kaiwu Community source is conformance
input, not a substitute for the proprietary SDK.

## 3. Run the development-only A800 rehearsal

This stage uses the existing validation image and an in-memory fake transport.
It checks proposal forward, conditioned Boltzmann sampling through
`KaiwuSampler`, the energy objective, backward, an optimizer update, and one
guided generation step on the observed A800.

Run `examples/qdiffusion_kaiwu/run_a800_development_probe.sh` with absolute
source, plugin, and output paths. Supply the stable SSH alias as
`EXECUTION_HOST` and the separately observed machine hostname as
`EXPECTED_HOSTNAME`. The runner disables networking, exposes only GPU 0, mounts
both code trees read-only, and writes one exclusive mode-0600 record.

The record must retain all of these values:

- `evidence_class=development_fake_transport`;
- `transport=in_memory_fake`;
- `qboson_hardware_used=false`;
- `real_provider_evidence=false`;
- `system_acceptance=false`.

Never promote this record into the live acceptance manifest.

## 4. Establish the approved SDK lane

Create an isolated Python 3.10 environment from the approved wheel set. Verify
package files and versions before setting credentials. Do not install an
unreviewed package merely because it shares the name `kaiwu`.

At minimum, record:

```bash
python3 --version
python3 -c 'import numpy, torch; print(numpy.__version__, torch.__version__)'
python3 -c 'import kaiwu; print(kaiwu.__version__)'
```

The observed values must exactly match `acceptance_config.json`. A `+cu...`
Torch build suffix is part of the observed version and must not be silently
discarded. Importing FlagQuantum without Kaiwu installed must continue to work
in the normal environment.

## 5. Run the guarded provider smoke test

Use a fresh task prefix and the assigned project. The following command submits
one optimization task and one sampling task and may consume quota:

```bash
python3 examples/qdiffusion_kaiwu/qboson_live_smoke.py \
  --checkpoint-dir private-kaiwu-checkpoints \
  --output private-evidence/qboson-smoke-attempt-001.json \
  --project-no "$QBOSON_PROJECT_NO" \
  --task-prefix "flagquantum-smoke-${RUN_ID}" \
  --expected-sdk-version 1.3.1 \
  --acknowledge-provider-cost I_ACKNOWLEDGE_QBOSON_QUOTA_USAGE
```

`QBOSON_PROJECT_NO` is not a credential, but it should still be managed in the
private run environment. The command has no simulator fallback. It exits with
hardware acceptance closed if the pinned SDK mapping cannot supply stable
provider task and target identities.

Stop here if authentication, quota, provider status, spin validation, energy
recomputation, task identity, or target identity is unresolved. Do not move to
QDiffusion by replacing the provider with a local solver.

## 6. Run one live QDiffusion system attempt

Run this command inside the exact frozen lane on one observed A800. If a
container is used, set its hostname to the reviewed host hostname so the
recorded identity is not a random container ID.

```bash
python3 examples/qdiffusion_kaiwu/qdiffusion_system_live.py \
  --config private-evidence/acceptance_config.json \
  --checkpoint-dir private-kaiwu-checkpoints \
  --output private-evidence/system-attempt-001.json \
  --execution-host jp-a800-171 \
  --expected-hostname "$EXPECTED_MACHINE_HOSTNAME" \
  --source-revision "$FLAGQUANTUM_REVISION" \
  --plugin-revision "$KAIWU_PLUGIN_REVISION" \
  --project-no "$QBOSON_PROJECT_NO" \
  --task-prefix "flagquantum-qdiffusion-${RUN_ID}" \
  --device cuda:0 \
  --expected-sdk-version 1.3.1 \
  --acknowledge-provider-cost I_ACKNOWLEDGE_QBOSON_QUOTA_USAGE
```

The command validates the frozen config and A800 before resolving credentials.
It uses explicit integer precision, a hard remote-call budget, and no fallback.
It records every distinct original-matrix precision report, all returned task
receipts, the training update, generation constraints, and repeat retrieval of
the last task.

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

Before either host run, verify the four staged inputs offline. File artifacts
use ordinary SHA-256; directory snapshots use the path-aware
`tree-sha256-v1` algorithm implemented by the preflight tool. The tool rejects
relative paths, symlinks, special files, digest mismatches, and existing output
records:

```bash
python examples/qdiffusion_kaiwu/preflight_protein_artifacts.py \
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

The frozen configuration includes every plugin knob that changes the selected
corpus or generated sequences: record-length bounds, record cap, validation and
test ratios, proposal/noise/energy temperatures, candidate count, generation
steps, resampling policy, ESM2 pairing/pooling, and evaluation batch size. The
current values select 640 eligible records and therefore 32 test sequences at a
0.05 test ratio. Changing any of these values creates a different experiment
and requires a new preregistered configuration identity.

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
python examples/qdiffusion_kaiwu/qdiffusion_protein_training_live.py \
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
  --project-no APPROVED_PROJECT \
  --task-prefix qdiffusion-protein \
  --seed 1701 \
  --acknowledge-provider-cost I_ACKNOWLEDGE_QBOSON_QUOTA_USAGE
```

Repeat with a new exclusive preflight and training-record path for every frozen
seed. A training record is not system or application acceptance; do not promote
it until ESM2 evaluation and both host gates pass.

Do not reuse the bounded system probe's `remote_call_budget` for this command.
The plugin invokes the sampler once per conditioned example for positive energy,
once per negative candidate, and once per generated candidate per step. With
the illustrative 640-record, 20-epoch, four-candidate, 64-step configuration,
the conservative bound is 71,048 submissions per seed. Therefore
`training.remote_call_budget_per_seed` is deliberately left as a required value:
resize and preregister the experiment or secure matching project quota before
running. The validator rejects a budget below the bound; deduplication may lower
actual submissions but must not be assumed during quota planning.

## 10. Assemble and validate final evidence

The live-system probe record is an attempt record, not by itself the final
acceptance record. Join accepted system evidence with the primary application
metrics and the replay result into exactly two host records following the
schema enforced by `validate_acceptance.py`. Create a manifest based on
`acceptance_manifest.example.json`, using relative paths and freshly computed
SHA-256 digests.

Run the fail-closed validator from the same source revision:

```bash
python3 examples/qdiffusion_kaiwu/validate_acceptance.py \
  private-evidence/acceptance_manifest.json
```

The validator recomputes the decision. It rejects changed software or config,
missing host coverage, shared provider tasks, fake transport, CPU execution,
missing target identity, fallback, retrieval resubmission, invalid training or
generation values, incomplete precision evidence, altered thresholds, and a
guided result that misses the preregistered application criteria.

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

## Claim boundary

A passing bundle establishes only the declared A800 plus QBoson QDiffusion
scope. It does not establish a general gate-model backend, differentiation
through remote sampling, domestic-accelerator support, distributed execution,
performance superiority, production maturity, quantum advantage, or a named
QBoson product unless that exact product is provider-reported and reviewed.
