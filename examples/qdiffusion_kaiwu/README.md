# QDiffusion with Kaiwu

This directory separates development checks from acceptance evidence.
Run the documented Python module commands from the reviewed FlagQuantum
checkout root. The `-s` flag disables the user-site package directory so an old
installed FlagQuantum cannot silently replace the current source tree.

## Credential-free local golden path

`run_local_conformance.sh` is the single local entry point for the conversion,
precision, remote-lifecycle, sampler, pinned Kaiwu Community, and pinned Kaiwu
PyTorch Plugin contracts. It performs no network or provider operation, removes
QBoson credential variables from the child environment, requires clean source
checkouts at the reviewed revisions, explicitly enables only those source-based
conformance tests, and normally completes in under ten minutes:

```bash
bash examples/qdiffusion_kaiwu/run_local_conformance.sh \
  /absolute/src/kaiwu_community \
  /absolute/src/kaiwu-pytorch-plugin \
  /absolute/flagquantum-venv/bin/python
```

Passing this path is local conformance evidence only. It is not A800 or QBoson
hardware evidence and cannot be included as a substitute component in the final
acceptance bundle.

`verify_transfer_bundle.py` is the pre-extraction gate for approved host
transfers. It verifies the three colocated source archives against their
reviewed manifest, binds their internal roots to declared revisions, and rejects
unsafe tar members. Its output is a private transfer-preflight record, not
execution or acceptance evidence. After extraction,
`verify_extracted_bundle.py` compares the otherwise empty extraction tree
file-for-file with those archives and rejects changes, omissions, additions,
links, and special entries. Run it with `python -B -s -m ...` before any other
import from the extracted tree so Python cannot add bytecode first.
Use `build_transfer_bundle.py` to create those three archives and the manifest
from clean pinned Git checkouts in one new private directory. The builder runs
the same verifier for both target aliases and refuses dirty or wrong-revision
inputs.

`a800_sampler_smoke.py` is a development probe. It uses the real Kaiwu PyTorch
Plugin data path and an explicitly selected in-memory fake transport. It can
verify A800 tensor placement, matrix and sample transfers, backward, and an
optimizer update. It cannot establish QBoson hardware use or QDiffusion system
acceptance.

The development probe records both the stable validation-host alias and the
machine-reported hostname because they are different on the current systems.
The container hostname is explicitly set to the observed host identity because
Docker otherwise assigns an unrelated container ID. After extracting the
reviewed source and plugin archives, invoke the bounded runner on the first host
as follows:

```bash
mkdir -m 700 private-evidence
bash examples/qdiffusion_kaiwu/run_a800_development_probe.sh \
  jp-a800-171 \
  bm-baai-dx-zone1-lc-a800-80g-15-171 \
  "$PWD" \
  /absolute/path/to/kaiwu-pytorch-plugin \
  "$PWD/private-evidence/jp-a800-171-extraction-preflight.json" \
  "$PWD/private-evidence/jp-a800-171-development.json" \
  FULL_FLAGQUANTUM_REVISION \
  f047bce7b1077449967bbe9e9fab5741542b48d4 \
  sha256:FULL_LOCAL_VALIDATION_IMAGE_ID
```

The runner executes `qdiffusion_system_development_probe.py`, covering proposal
forward, conditioned Boltzmann sampling through `KaiwuSampler`, energy
objective, backward, optimizer update, and one guided generation step. The
output is a new mode-0600 file and is never an acceptance record because the
transport is explicitly the in-memory development fake. The runner exposes
only GPU 0, disables networking, mounts both code trees read-only, uses a
read-only container filesystem, mounts the post-extraction preflight separately
read-only, and persists only the requested evidence file. The record retains
the preflight and common transfer-manifest digests. The final argument is the
full local Docker image ID, not a mutable tag. Resolve and review it
independently on each host with `docker image inspect`; the same tag currently
maps to different image IDs on the two validation hosts.

The acceptance lane uses a frozen configuration and two independent host
records:

1. Copy `acceptance_config.example.json` to an evidence directory and replace
   every placeholder before any baseline or guided experiment runs.
2. Record the file's SHA-256 digest in each host record and in a manifest based
   on `acceptance_manifest.example.json`.
3. Run the bounded system path independently on `jp-a800-171` and
   `jp-a800-172`. The configured primary host also runs every frozen seed in the
   protein experiment; the replay host runs the fixed portability fixture.
4. Validate the finished bundle:

```bash
python -s -m examples.qdiffusion_kaiwu.validate_acceptance path/to/manifest.json
```

The validator recomputes the decision from evidence. It requires an observed
A800 on both hosts, independent QBoson task identities, `kaiwu_cim` transport,
real-provider evidence, no fallback or retrieval resubmission, bounded calls,
finite training values, valid generation, and the preregistered application
thresholds. A record that uses the development fake, CPU tensors, one host, or
post-hoc metric thresholds fails closed.

Credentials never belong in the frozen configuration or evidence bundle.

## Live provider smoke test

`qboson_live_smoke.py` is a separately invoked, quota-consuming Phase 2 probe.
It submits one fixed optimization task and one fixed sampling task, uses the
same identity for bounded polling, and writes a new mode-0600 record without
credentials or raw vendor exception text. It has no simulator fallback.

The command requires `QBOSON_USER_ID`, `QBOSON_SDK_CODE`, an existing private
checkpoint directory, and an explicitly selected project. The acknowledgement
must be typed exactly so an ordinary test run cannot spend provider quota:

```bash
python -s -m examples.qdiffusion_kaiwu.qboson_live_smoke \
  --checkpoint-dir private-kaiwu-checkpoints \
  --output private-evidence/qboson-smoke.json \
  --project-no CPQC-your-project \
  --task-prefix flagquantum-smoke-20261005 \
  --acknowledge-provider-cost I_ACKNOWLEDGE_QBOSON_QUOTA_USAGE
```

Successful tasks alone do not make this an acceptance record. The script keeps
`hardware_acceptance=false` until the pinned SDK mapping supplies both a stable
provider task ID and a provider-reported target for every task.
It also records a value-free schema of the documented SDK result dictionary so
the missing mapping can be reviewed without persisting raw provider values.

## Live QDiffusion system probe

`qdiffusion_system_live.py` joins the same bounded QDiffusion slice to
`KaiwuSDKClient`. It must run only in the frozen Python, Torch, plugin, SDK,
precision, host-role, and source-revision lane recorded by a completed
`acceptance_config.json`. It validates that lane before resolving credentials.

The command requires the same exact quota acknowledgement as the smaller live
smoke test. It writes attempted task receipts even when the QDiffusion slice
fails after submission, verifies repeat retrieval against the same last task,
and scans the serialized record for the resolved `user_id` and `sdk_code`
before creating a mode-0600 file. There is no local fallback.

This probe exits unsuccessfully until all system gates pass. In particular,
the current documented SDK adapter leaves provider task and target identity
unavailable, so a real completed run will still record `system=fail` pending a
reviewed response mapping. The full command should be generated from the
approved private runbook rather than copied with placeholder project or
credential values.

Every distinct original Ising matrix retains its own precision report even
when two inputs quantize to the same submitted matrix. Live evidence aggregates
the report count, scale-factor range, maximum absolute error, and mean of the
per-matrix mean errors; a record with fewer reports than remote calls fails the
acceptance validator.

## Live frozen protein training

`qdiffusion_protein_training_live.py` runs one preregistered seed of the pinned
plugin's complete protein training workflow on the configured primary A800. It
performs artifact and FASTA preflight before resolving credentials, forces
Transformers into offline mode, constructs every plugin dataclass from the
frozen acceptance config, and binds all workflow-created generators to one
budgeted FlagQuantum `KaiwuSampler`. The base checkpoint directory must also
contain the tokenizer files required by the pinned plugin.

The command requires absolute paths and the exact provider-cost
acknowledgement. It creates a mode-0600 preflight record plus a separate private
training record. A successful record contains the best trained energy
checkpoint digest and remote receipts, but deliberately marks system and
application acceptance `not_evaluated`: all configured seeds, ESM2 evaluation,
and the independent replay-host gate still remain.

The top-level `remote_call_budget` protects the bounded system probe only. Full
protein training uses `training.remote_call_budget_per_seed`. The validator
computes a conservative worst-case bound from the frozen record count, epochs,
training candidates, test count, both baseline/guided objectives, generation
candidates, and generation steps; it rejects a smaller per-seed budget before
credentials are used.

`qdiffusion_protein_evaluate.py` performs the subsequent local A800 evaluation
without submitting any new provider task. It verifies the training record's
hash chain for the held-out test FASTA, baseline and guided FASTA files, and
sequence-quality summaries. It loads ESM2 exclusively through
`load_model_and_alphabet_local`, checks aligned headers and exact sequence
counts, then emits cosine/L2 plus sequence-quality metrics in a private record.
That record remains candidate evidence until it is assembled with all frozen
seeds and the two independent system-gate records.

`qdiffusion_portability_replay_live.py` transfers one preregistered primary-host
energy checkpoint and its hashed held-out FASTA to the replay host. It rebuilds
the DPLM generator from frozen local artifacts, injects a fresh bounded
`KaiwuSampler`, loads the exact trained energy weights, and runs one fixed
objective plus a short guided generation. The fixture seed, record index, and
step count are part of the frozen config; replay cannot silently select a better
example. The command is quota-consuming and requires the same exact cost
acknowledgement as other live paths.

`assemble_acceptance.py` is the only path from component records to the final
two-host manifest. It requires exactly one successful training and ESM2 record
for every frozen seed, verifies each evaluation links to its training record,
checks the replay uses the preregistered seed's exact checkpoint, computes
arithmetic means across all seed metrics, and reruns the fail-closed acceptance
validator. It also requires and copies the primary- and replay-host
post-extraction preflight records, then verifies that every execution record
links to the correct host preflight and their common transfer manifest. The
manifest itself is a required copied component, so the final evidence remains
self-contained after the temporary transfer directory is unavailable. The
inputs must be absolute, private regular files rather than symlinks. Assembly
uses a private sibling staging directory and publishes the requested output
directory only after the final validator passes. The output includes hashed
copies of every component record, so deleting or replacing a source record
makes the final manifest invalid. Revalidation also rejects public permissions
and symlinks in any manifest member path.
