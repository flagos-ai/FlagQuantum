# QDiffusion with Kaiwu

This directory separates development checks from acceptance evidence.
Run the documented Python module commands from the reviewed FlagQuantum
checkout root. The `-B` flag prevents bytecode files from mutating reviewed
source trees, and `-s` disables the user-site package directory so an old
installed FlagQuantum cannot silently replace the current source tree.

## Credential-free local golden path

`run_local_conformance.sh` is the single local entry point for the conversion,
precision, remote-lifecycle, sampler, pinned Kaiwu Community, and pinned Kaiwu
PyTorch Plugin contracts. It performs no network or provider operation, removes
QBoson credential variables from the child environment, requires clean source
checkouts at the reviewed revisions, explicitly enables only those source-based
conformance tests, and normally completes in under ten minutes. A dedicated
`sitecustomize` guard denies DNS resolution and IP socket connections throughout
the pytest process tree, including Python subprocesses; offline-model variables
alone are not treated as proof that the path cannot use the network:

```bash
bash examples/qdiffusion_kaiwu/run_local_conformance.sh \
  /absolute/src/kaiwu_community \
  /absolute/src/kaiwu-pytorch-plugin \
  /absolute/flagquantum-venv/bin/python
```

Passing this path is local conformance evidence only. It is not A800 or QBoson
hardware evidence and cannot be included as a substitute component in the final
acceptance bundle.

Before any live invocation, run the value-free offline readiness audit. Every
argument is optional so an initial run can enumerate missing inputs with stable
reason codes; provide the complete set when performing the final preflight:

```bash
python -B -s -m examples.qdiffusion_kaiwu.audit_readiness \
  --config /private/acceptance_config.json \
  --environment-lock /private/environment_lock.json \
  --sdk-approval /private/sdk-approval.json \
  --provider-resources /private/provider-resources.json \
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

The audit validates credential and project-variable presence and basic format
without recording their values, then checks a private, at-most-24-hour-old
Resource Bill snapshot against the optimization-plus-sampling requirement of
the provider smoke, the private Kaiwu checkpoint directory, frozen quota
ceiling, SDK environment,
standalone/frozen approval alignment, common transfer manifest, and protein
artifacts, and performs no network or provider operation. Provider-smoke
readiness is reported independently of the unfinished protein configuration;
use `--require-stage provider-smoke` for that earlier gate and omit the config,
source-preflight, and protein arguments. `system-probe` selects the intermediate
gate and additionally requires an observed NVIDIA A800 at `cuda:0`. Run that
gate inside the intended validation host environment. The default
`protein-experiment` gate exits zero only when the complete
experiment is ready. A passing report is readiness information, not provider,
execution, hardware, or acceptance evidence.

Create the resource snapshot from an authenticated read-only Resource Bill
view by copying `provider_resources.example.json` to a mode-0600 file in a
mode-0700 directory. Record all six SPQC target/mode rows without account IDs,
batch IDs, credentials, or project numbers. `captured_at` and `valid_until`
must be aware UTC timestamps no more than 24 hours apart. The provider-smoke
gate passes only when one target has at least one optimization and one sampling
resource. A snapshot is account-state reconnaissance; it does not authorize
spending and does not replace the explicit live-command cost acknowledgement.

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
inputs. Pre-create every evidence output parent as a mode-0700 real directory;
the preparatory tools share the live-evidence writer and reject relative output
paths, missing or public parents, and symlinked parents.

`a800_sampler_smoke.py` is a development probe. It uses the real Kaiwu PyTorch
Plugin data path and an explicitly selected in-memory fake transport. It can
verify A800 tensor placement, matrix and sample transfers, backward, and an
optimizer update. It cannot establish QBoson hardware use or QDiffusion system
acceptance.

The development probe records both the stable validation-host alias and the
machine-reported hostname because they are different on the current systems.
The container hostname is explicitly set to the observed host identity because
Docker otherwise assigns an unrelated container ID. Run the bounded streaming
runner from the reviewed local checkout. It revalidates the local bundle and
post-extraction preflight before opening SSH, streams the archives into the
container over standard input, and captures the returned JSON in a new private
local file:

```bash
mkdir -m 700 private-evidence
bash examples/qdiffusion_kaiwu/run_a800_development_probe.sh \
  jp-a800-171 \
  bm-baai-dx-zone1-lc-a800-80g-15-171 \
  /absolute/private-transfer-bundle \
  "$PWD/private-evidence/jp-a800-171-extraction-preflight.json" \
  "$PWD/private-evidence/jp-a800-171-development.json" \
  FULL_FLAGQUANTUM_REVISION \
  f047bce7b1077449967bbe9e9fab5741542b48d4 \
  sha256:FULL_LOCAL_VALIDATION_IMAGE_ID
```

The runner executes `qdiffusion_system_development_probe.py`, covering proposal
forward, conditioned Boltzmann sampling through `KaiwuSampler`, energy
objective, backward, optimizer update, and one guided generation step. The
output is a new local mode-0600 file and is never an acceptance record because
the transport is explicitly the in-memory development fake. The remote runner
exposes only GPU 0, disables networking and container logging, uses a read-only
container filesystem, and places the transferred archives, extracted trees,
and remote evidence only in tmpfs. It uses no bind mount and the auto-removed
container leaves no source or evidence file on the validation host. The outer
stream combines `COPYFILE_DISABLE=1` with `tar --no-xattrs` so unreviewed Apple
extended attributes are not sent alongside the manifest-bound files. Before
plugin import, the probe recomputes both extracted trees against the streamed
post-extraction preflight. The local capture gate independently requires the
fake-transport classification, closed acceptance fields, exact source,
preflight, manifest, host and image identities, A800 `cuda:0` placement, and
bounded call accounting before publishing the record. The final argument is
the full remote Docker image ID, not a mutable tag. Resolve and review it
independently on each host with `docker image inspect`; the same tag currently
maps to different image IDs on the two validation hosts.

The acceptance lane uses a frozen configuration and two independent host
records:

1. Copy `acceptance_config.example.json` to an evidence directory and replace
   every placeholder before any baseline or guided experiment runs. Its parent
   must be a real mode-0700 directory and the config itself must be mode `0600`.
   Freeze the machine-reported hostname for each validation alias under
   `host_identities`; the two values must be distinct. Every live command's
   `--expected-hostname` must match this mapping, while `--execution-host`
   remains the stable `jp-a800-171` or `jp-a800-172` alias.
   The config schema is closed: do not add ad hoc metadata or credentials to
   the top level or any nested section. Undeclared keys are rejected before a
   live entrypoint resolves credentials, and the final validator repeats that
   check before retaining the config.
   The environment lock is closed in the same way: its top-level object and
   every distribution entry may contain only the documented inventory fields.
2. Record the file's SHA-256 digest in each host record and in a manifest based
   on `acceptance_manifest.example.json`.
3. Run the bounded system path independently on `jp-a800-171` and
   `jp-a800-172`. The configured primary host also runs every frozen seed in the
   protein experiment; the replay host runs the fixed portability fixture.
4. Validate the finished bundle:

```bash
python -B -s -m examples.qdiffusion_kaiwu.validate_acceptance path/to/manifest.json
```

The validator recomputes the decision from evidence. It requires an observed
A800 on both hosts, independent QBoson task identities, `kaiwu_cim` transport,
real-provider evidence, no fallback or retrieval resubmission, bounded calls,
finite training values, valid generation, and the preregistered application
thresholds. A record that uses the development fake, CPU tensors, one host, or
post-hoc metric thresholds fails closed.

Live system, training, and portability writers also reconcile per-matrix
precision identities with provider receipts and their sampler transfer origins
and require serialized values to equal the retained precision reports before
setting their own completeness or pass fields. Final validation repeats those
checks and requires the component completeness flags; component-local status is
not trusted as acceptance evidence.

Credentials never belong in the frozen configuration or evidence bundle.

The frozen config separately binds the reviewed Kaiwu wheel filename, source,
SHA-256, public service-terms identity and effective date, review timestamp,
approval reference, and the approved organizational, isolated-container, host
staging, and adapter-distribution uses. The project policy remains
`no-sdk-redistribution`: FlagQuantum may distribute its adapter, not the
proprietary wheel. All approval fields must be literal booleans, and each must
be `true`; placeholders and inferred approval fail before credential
resolution. Final validation also requires the exact Kaiwu wheel digest and
version in the environment lock to match this reviewed SDK record.

`sdk_approval.example.json` defines the same strict approval object as a
standalone private record. It exists so the Phase 2 provider smoke can enforce
the SDK rights gate before the later protein configuration is complete. Its
path must be absolute; the file must be regular, non-symlinked, and mode `0600`.
The smoke records its SHA-256 and cross-checks the approved Kaiwu version and
wheel digest against the verified environment lock before credential discovery.
Version 1.1 also retains the reviewed SDK-capable project number, assignment
review time, and assignment reference. Every quota-consuming CLI requires its
project argument to match that private record before credentials are resolved.
Later QDiffusion entrypoints repeat the cross-check using the identical object
embedded under `acceptance_config.json.kaiwu_sdk`.

The frozen-config, standalone-approval, and environment-lock readers share one
bounded private-input primitive. It opens each leaf relative to an anchored
owner-only parent descriptor with `O_NOFOLLOW`, rejects public or symlinked
parents and leaves, and rechecks the path and metadata after the read. This
prevents a nominally mode-0600 file in an unsafe or replaced directory from
authorizing provider use.

Each frozen protein input also records an HTTPS acquisition source, an approved
license identifier, an HTTPS license-evidence source, and the timezone-aware
time of that review. `NOASSERTION`, `UNKNOWN`, `UNLICENSED`, missing evidence,
and placeholder values fail before credential resolution. A public download is
not license approval. The current intake status and exact freeze procedure are
recorded in [`ASSET_INTAKE.md`](ASSET_INTAKE.md); in particular, the candidate
Hugging Face DPLM artifact has no license metadata and has not yet been linked
authoritatively to the Apache-2.0 notice in the official code repository. It
therefore remains a hard acceptance blocker.

`build_environment_lock.py` builds the private dependency-lane lock directly
from an explicitly reviewed wheel set. It reads each wheel's bounded METADATA,
hashes its bytes, and requires an exact one-to-one name/version match with every
distribution in the running isolated Python environment. It also hashes each
distribution's installed RECORD file set, including file names, sizes, and
contents. The generated lock is then checked by `verify_environment_lock.py`;
its Python version, complete distribution inventory, and installed contents
must match the runtime exactly, and its file digest must match the frozen
config. `environment_lock.example.json` documents the schema only and is not a
complete lock. Every quota-consuming path verifies the real lock before
credential resolution. Neither command installs or downloads a package, and
supplying a wheel does not turn it into an approved artifact.

## Live provider smoke test

`qboson_live_smoke.py` is a separately invoked, quota-consuming Phase 2 probe.
It submits one fixed optimization task and one fixed sampling task, uses the
same identity for bounded polling, and writes a new mode-0600 record without
credentials or raw vendor exception text. Both resolved credential values are
scanned recursively through nested keys and values before JSON serialization,
so escaped quotes, backslashes, or newlines cannot bypass the refusal check. It
has no simulator fallback. Its output parent must already be a private,
non-symlink directory, and the final path must not exist. Every quota-consuming
entrypoint validates its final evidence destinations before credential
resolution; the exclusive writer repeats that validation at publication time.
The writer anchors temporary creation, exclusive linking, cleanup, and
directory synchronization to one `O_NOFOLLOW` directory descriptor, and
refuses publication if the visible parent binding changes. Live evidence is
published without replacement only after both file contents and
parent-directory metadata are synchronized. A
timeout, provider failure, malformed result, or keyboard interruption after a
task receipt exists is converted into a failed attempted record; the smoke
sequence stops without submitting its next task, and hardware acceptance stays
closed. Failure records retain only a stable category and an empty message;
arbitrary vendor or plugin exception text is never persisted.

Provider-use facts are independent of whole-command acceptance. If a validated
real-SDK result with provider task and target identity is returned before a
later task or local step fails, `qboson_hardware_used` and
`real_provider_evidence` remain true: the already-consumed provider work is not
erased. `run_completed`, the relevant acceptance gate, and hardware acceptance
remain false. A receipt-only timeout or failed attempt is retained for recovery
but is not by itself sufficient to prove hardware use.
The live smoke, system, training, and replay producers require the exact pinned
`KaiwuSDKClient` type; an injected subclass remains test transport even if a
caller sets a real-transport flag.

The command requires `QBOSON_USER_ID`, `QBOSON_SDK_CODE`, an existing absolute
private checkpoint directory, the project assigned in the reviewed SDK approval,
and a current
private provider-resource snapshot that passes the optimization-plus-sampling
gate. The snapshot is loaded before SDK approval, credentials, or license
initialization, and its exact digest is retained in the smoke record. All
quota-consuming entrypoints reject a missing, public, relative, or symlinked
checkpoint directory before resolving credentials; the SDK client repeats the
check before license initialization. After resolving a valid pair, each CLI
copies it into an explicit in-memory credential object and removes both
variables from the process environment before SDK client construction, so
later plugin code or child processes cannot inherit them. The acknowledgement
must be typed exactly so an ordinary test run cannot spend provider quota:

Task prefixes and project numbers must be nonempty printable text. Control or
format characters fail before SDK submission; restored provider task/target
identities and final evidence must also be canonical without surrounding
whitespace.

```bash
python -B -s -m examples.qdiffusion_kaiwu.qboson_live_smoke \
  --checkpoint-dir /absolute/private-kaiwu-checkpoints \
  --environment-lock /absolute/private-evidence/environment-lock.json \
  --sdk-approval /absolute/private-evidence/sdk-approval.json \
  --provider-resources /absolute/private-evidence/provider-resources.json \
  --output /absolute/private-evidence/qboson-smoke.json \
  --project-no CPQC-your-project \
  --task-prefix flagquantum-smoke-20261005 \
  --acknowledge-provider-cost I_ACKNOWLEDGE_QBOSON_QUOTA_USAGE
```

Successful tasks alone do not make this an acceptance record. The script keeps
`hardware_acceptance=false` until the command is using the real SDK transport
and the pinned SDK mapping supplies both a stable provider task ID and a
provider-reported target for every task. Injected clients are always recorded
as `transport=injected_test`, `real_provider_evidence=false`, and
`qboson_hardware_used=false`. The command preserves the diagnostic record but
returns a nonzero exit status whenever hardware acceptance remains closed.
It also records a value-free schema of the documented SDK result dictionary so
the missing mapping can be reviewed without persisting raw provider values.

## Live QDiffusion system probe

`qdiffusion_system_live.py` joins the same bounded QDiffusion slice to
`KaiwuSDKClient`. It must run only in the frozen Python, Torch, plugin, SDK,
precision, host-role, and source-revision lane recorded by a completed
`acceptance_config.json`. It validates that lane before resolving credentials.
The command requires the absolute extracted plugin root and recomputes both the
executing FlagQuantum tree and plugin tree against the extraction preflight. It
also rejects `kaiwu.torch_plugin` or QDiffusion modules imported from any other
installed or preloaded location.

The command requires the same exact quota acknowledgement as the smaller live
smoke test. It writes attempted task receipts even when the QDiffusion slice
fails after submission, verifies repeat retrieval against the same last task,
and scans the serialized record for the resolved `user_id` and `sdk_code`
before creating a mode-0600 file. It also requires a current private provider
resource snapshot and verifies that one Sampling target can cover the frozen
`remote_call_budget` before credentials or the SDK client are initialized.
There is no local fallback.

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

Before credentials or model imports, the training launcher requires a current
private provider-resource snapshot whose balance on one Sampling target covers
the frozen per-seed remote-call ceiling. The portability launcher applies the
same check to the frozen system/replay ceiling. These are fail-closed spending
guards; the snapshots remain observations rather than provider reservations.
Each record retains a closed gate containing the snapshot digest, UTC check
time, Sampling mode, and declared ceiling. Final validation replays that check
against the copied snapshot and rejects a receipt that predates its gate or
follows the snapshot's validity window. The live launchers also bind that
window's `valid_until` value to the SDK client's submission deadline, so a late
new task fails before provider use while recovery and retrieval remain usable.

After training returns, the command rehashes the frozen config, FASTA, base
checkpoint/tokenizer tree, and ESM2 checkpoint before publishing its record.
Any drift marks the attempted run incomplete even if provider work succeeded.
The portability replay repeats that postflight and also rechecks the transferred
test FASTA and trained energy checkpoint. Local ESM2 evaluation rechecks both
its model checkpoint and every consumed training output after metric
calculation. Final validation requires `artifact_inputs_unchanged=true` in all
three component types. A postflight failure is represented through the existing
closed `run_completed`, `failure`, and `artifact_inputs_unchanged` fields; the
producer does not append a diagnostic-only field that final assembly would
reject, and exception messages remain absent from evidence.

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

Before requesting quota, render the complete submission plan without resolving
credentials or importing Kaiwu:

```bash
python -B -s -m examples.qdiffusion_kaiwu.plan_quota \
  /absolute/private/path/to/frozen-acceptance-config.json
```

The strict-JSON report separates the two system hosts, every protein seed, the
portability replay, and the two-task smoke test. Its unit is distinct provider
task submissions; status polls and repeated result retrievals are explicitly
excluded. For the illustrative template it derives a conservative maximum of
71,269 submissions per protein seed and 213,846 submissions overall, while
leaving the declared total ceiling unresolved because
`training.remote_call_budget_per_seed` is still `<required>`. Deduplication may
reduce an actual run. This is a planning calculation, not quota approval or
execution evidence.

The quota planner treats that frozen configuration as private control input:
both the file and its immediate real directory must be owner-only, and symbolic
links are rejected.

`qdiffusion_protein_evaluate.py` performs the subsequent local A800 evaluation
without submitting any new provider task. It verifies the training record's
hash chain for the held-out test FASTA, baseline and guided FASTA files, and
sequence-quality summaries. It also reloads the primary-host extraction
preflight and requires its file and transfer-manifest digests to match the
training record rather than inheriting those claims. System, training,
evaluation, and replay recompute the executing FlagQuantum tree and actual
plugin-root file count and content-set digests before importing the workflow.
They also bind the executing `flagquantum.version.__version__` to the frozen
software lane and retain it in every component and final host record.
System, training, and replay additionally reject the core QDiffusion modules
when their resolved files are outside that plugin root. All loaded
`kaiwu.torch_plugin.*` and `dplm.*` transitive modules are checked as well. The
evaluator loads ESM2 exclusively through
`load_model_and_alphabet_local`, checks aligned headers and exact sequence counts,
then emits cosine/L2 plus sequence-quality metrics in a private record.
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
environment lock and transfer manifest are required copied members, so the
final evidence remains self-contained after the temporary preparation
directories are unavailable. The frozen protein-artifact preflight is also a
required copied component: every training record and the portability replay
must reference its exact digest, allowing offline revalidation of the dataset,
checkpoint, tokenizer, and evaluation-model identities. The
standalone SDK approval, every uniquely referenced provider-resource snapshot,
and successful Phase 2 provider smoke are required copied components as well.
Repeat `--provider-resources` during assembly when the staged executions used
different snapshots. Final validation
requires the approval to equal the frozen config decision, binds both private
input digests to the smoke, verifies that the smoke time fell within the
snapshot's bounded validity interval, rechecks that one target had both
required modes, and independently
checks that the smoke retained one successful optimization task and one
successful sampling task over the independently recomputed exact smoke-matrix
identity, with recoverable receipt schema/project/submission-time fields, real
provider identities, complete bounded spin/energy vectors, independent energy
recomputation, closed record/task field sets, fixed claim limitations, and no
fallback. Hardware acceptance additionally requires two distinct provider task
IDs and one common provider target, which is retained at record level and
cross-checked against both tasks. Provider task IDs must also be globally unique
across the smoke, both system probes, all training seeds, and portability
replay. System, training, evaluation, and portability component records also
use closed top-level and evidence-bearing nested field sets; missing producer
fields, undeclared extensions, altered fixed limitations, or a completed remote
component carrying a failure are rejected. Provider-result schemas retained by
the Phase 2 smoke and system probe are limited to the pinned client's
value-free type/shape grammar, not arbitrary provider values. Tests bind the
top-level field sets to the actual payload constructors. Portability validation
also rechecks all frozen artifact identities plus its preregistered training
seed, fixture index, and replay step count; a different fixture cannot pass by
retaining the same outer record shape. The inputs must be absolute, private
regular files rather than symlinks. The final manifest, every member reference,
both aggregated host records, and their artifact, precision, training,
generation, transfer, metric, application, and portability objects use closed
field sets as well. Both host records retain an exact fixed limitations list
that classifies the result as development evidence, distinguishes the two
single-device A800 runs from distributed or domestic-accelerator support, and
rejects performance, production, quantum-advantage, or scalability claims.
Both host records must explicitly require the complete component bundle;
omitting or disabling that flag cannot bypass source-record validation. Final
validation also reconstructs the execution timeline:
component timestamps must be UTC and follow
preregistration, SDK review, Phase 2 smoke, their remote submissions, and the
appropriate system/training prerequisites. Assembly
uses a private sibling staging directory and publishes the requested output
directory only after the final validator passes. The output includes hashed
copies of every component record, so deleting or replacing a source record
makes the final manifest invalid. Revalidation also rejects public permissions
and symlinks in any manifest member path, duplicate or non-normalized paths, and
files that are present in the bundle but absent from its manifest.
Every assembly input must also reside directly in an existing private,
non-symlink directory; mode-0600 alone is insufficient if another user can
replace its directory entry. The publisher repeats this check internally rather
than relying only on CLI validation. The final output parent is subject to the
same private, non-symlink rule before staging and atomic publication.
All frozen configuration, lock, preflight, transfer, component, and final
manifest JSON readers also reject duplicate object keys at every nesting level;
a byte-identical evidence file cannot rely on last-key-wins interpretation.
The same parser rejects non-standard `NaN`/`Infinity` constants and finite JSON
spellings whose decoded float would overflow, including inside nested metadata.
