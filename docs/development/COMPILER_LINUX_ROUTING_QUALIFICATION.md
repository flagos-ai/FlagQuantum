# Linux routing qualification declaration

Status: declared before measured execution on November 23, 2026. The
[original freeze manifest](evidence/compiler-linux-routing-freeze.json.gz) fixes participants,
configuration, wheel identities, the complete unchanged SDKBench Python harness,
fixture/reference bundle, thresholds and budgets. No campaign had run at this
freeze. This study checks inspected historical controls, not unseen applications.

The compressed manifest preserves the exact original JSON bytes (SHA-256
`eefcc61f0925d8d55bc5e2017a813bd9d8dd417fd34989b80ad3f96d61b6d95c`)
published at `7c83c5f82480f49143190ed82f98c8fd757722ec`. CI exposed an
unregistered field name and an inline external-origin stamp unsupported by one
reader. The [existing evidence index](evidence/compiler-installed-wheel-controls.json)
now records the same declaration with the established field vocabulary and
existing external-origin table. This metadata normalization changes no study
input or limit; the measured attempt read the original bytes. Original CI
failures remain retained, and no validator, schema allowlist or origin table is
changed.

The user authorized arranging a dedicated 24-hour window on `jp-bw1100-10`.
Environment writes stay inside a new study directory under `/data` and a new
cgroup subtree. No machine or service restart, kernel/driver upgrade, existing
job termination, global package installation or live QPU submission is allowed.
Sixteen repository runner label sets were retained and temporarily replaced with
`sdkbench-reserved-20261123`; existing jobs finish naturally. All original labels
must be restored at the end. The exclusive window starts only after draining and
admission, not when labels change. A reservation approval is not proof of an
exclusive running environment.

## Question and delivered workflows

For a fixed arbitrary-input circuit, target topology and identity logical output
order, what valid native program does each delivered workflow produce, at what
compilation time and aggregate process memory cost? Report separate CZ count,
two-body dependency depth, time, memory and failures. There is no weighted score
or universal SDK ranking.

| Registration | Delivered workflow | Exact input |
| --- | --- | --- |
| `fq-integrated-ordinary` | Default compiler plus native legalization and timed common layout restoration | FlagQuantum source `f88ebc88f8584cd580802cd219ba16d935fa6715`, installed wheel |
| `fq-integrated-artifact` | Target-artifact compilation, native RX/RZ/CZ lowering, serialization and restored layout | Same FlagQuantum wheel |
| `qiskit-release-restored` | Exact transpile level 2, identity initial layout, arbitrary input, timed common layout restoration | Qiskit 2.6.2 published wheel |
| `cirq-release-restored` | RouteCQC identity mapper, full-CZ synthesis and timed common layout restoration | Cirq Core 1.7.0 published wheel |

The two FlagQuantum paths have different delivery obligations. Report them
separately; artifact packaging and serialization are included in that path's
cost. The common restoration algorithm is the unchanged deterministic
spanning-tree leaf-token method; its emitted gates count and its runtime is timed.
Native SDK defaults are preserved where the frozen adapter specifies defaults.
Qiskit's exact synthesis and arbitrary-input options prevent exploiting known
zero input. Seeds are recorded as unsupported for deterministic paths that do
not expose them. No SDK is patched or given fixture-specific optimization.

All four registrations use the same private Python 3.12.14 environment and
hashed dependency lock, including CPU Torch 2.14.0. The frozen original P1b
profile's historical FlagQuantum 0.3.1 label remains unchanged: candidate
registration replaces the participant identity with the verified 0.4.0 wheel;
it does not pretend to execute the historical release. Published wheel metadata
observations verify bytes and non-yanked status, not independent publisher/build
attestation. SDKBench source is frozen at
`9cb8f184825e658f7378763a1f3a39c57720233d`.

## Cases, independent acceptance and limits

| Instance | Width and topology | Whole-case cap per registration |
| --- | --- | --- |
| `ROUTE-LINE-06` | Six wires, line | 600 seconds |
| `ROUTE-GRID-08` | Eight wires, 2 by 4 grid | 900 seconds |
| `ROUTE-LINE-12` | Twelve wires, line | 1,800 seconds |

Each case uses ten fresh process blocks, two warmups and three measured
compilations per block. Compiler seeds are 11, 23, 37, 51, 73, repeated twice.
The unchanged harness rotates participant order per block and executes only one
participant process at a time. Across four registrations this is 120 blocks,
360 measured repetitions and 240 warmups. Every failed warmup remains part of
the verdict. No rerun, omitted case, reduced repeat count or selected seed is
permitted to obtain a favorable result.

Each worker has one CPU/thread, 16 GiB aggregate cgroup memory including children,
no swap and no external network. A single compilation has a 120-second ceiling;
whole-case caps include the unchanged harness's complete accounting. The sum of
case ceilings is 13,200 seconds. Preparation/reference/evaluator planning retains
up to 14 hours 40 minutes total before operational slack; do not start if the
reserved window cannot accommodate all remaining ceilings. Installation ceilings
apply to each resolved environment, not a claim that building a private Python
interpreter took zero time. Retain that build and all setup failures separately.

Independent bit-block propagation and separate dense Kronecker/permutation
construction supply references. Six wires receive a complete-operator check up
to one global phase. Eight and twelve wires receive three full output-state
probes; this does not prove universal operator equivalence. Semantic absolute
error is at most 1e-8 with zero relative tolerance, normalization error at most
1e-10, and zero illegal target operations. There are no ancillas or partial CZs.
Frozen evaluators check reported layouts, topology, numerical outputs and metric
reconstruction; no optimal-routing oracle is claimed. Preserve negative controls
for omitted layout reconstruction, nonedge/partial CZ and erased noncommuting
rotations. Producer-free replay is required before conclusions.

## Host admission and interruption

Preflight observes Ubuntu 24.04.2, kernel 6.8.0-87, two EPYC 9K84 sockets,
192 physical cores and approximately 2.2 TiB RAM. SMT remains enabled. Pin CPU
96 and require sibling 288 to be idle. Seal actual host identity, OS/package
baseline, BIOS, power/clocks, CPU topology, memory, storage, load and process
inventory before participant timing. Reject changed source, wheels, registrations,
references, evaluation code or dependencies. Retain post-run baseline comparison
and host activity; no claim of independently attested immutable infrastructure is
made on this directly administered host.

All 15 unchanged resource positive/negative tests passed, including OOM, timeout,
network denial, affinity and descendant cleanup. All 12 selected installed-wheel
native semantic tests passed. These are preparation checks, not exclusivity or
performance evidence. Existing CI and unrelated workloads must drain naturally;
background system services are retained and disclosed. Do not stop or restart
processes to force admission. Zombie process inventory is recorded rather than
silently cleaned. If exclusive scheduling or idle sibling cannot be established,
keep the study unstarted and state the blocker.

Retain every raw block, worker log, supervisor outcome, warmup, failure and
producer-free replay. An interrupted attempt remains an interrupted attempt;
subsequent work requires a separately declared attempt. Never substitute sampled
RSS for cgroup `memory.peak`, relax thresholds, or turn missing evidence into a
pass. This candidate protocol remains `publication_eligible=false`; passing does
not certify a release, scalability or an SDK-wide ranking. Unseen real application
qualification remains a separate prospective step.

## Reproduction

With the frozen SDKBench checkout, verified installed wheels, registered files
and admitted Linux cgroup environment, the existing workflow is:

```sh
python -m sdkbench.candidates.cli run \
  --bundle docs/implementation --cpu 96 \
  --cgroup-parent /sys/fs/cgroup/sdkbench-compiler-qualification-20261123 \
  --registration /data/sdkbench-compiler-qualification-20261123/registrations/fq-integrated-ordinary.json \
  --registration /data/sdkbench-compiler-qualification-20261123/registrations/fq-integrated-artifact.json \
  --registration /data/sdkbench-compiler-qualification-20261123/registrations/qiskit-release-restored.json \
  --registration /data/sdkbench-compiler-qualification-20261123/registrations/cirq-release-restored.json \
  --directory /data/sdkbench-compiler-qualification-20261123/campaign-attempt-1
python -m sdkbench.candidates.cli replay \
  --bundle docs/implementation \
  --directory /data/sdkbench-compiler-qualification-20261123/campaign-attempt-1
```

Expected behavior: all twelve complete cases receive an independently evaluated
verdict, all blocks remain inspectable, and replay reconstructs acceptance from
retained artifacts. A nonzero command exit is a retained failure, not permission
to rerun. This declaration changes no implementation, acceptance contract,
public API, capability maturity or resource threshold.

## First measured outcome

The first campaign completed with exit code zero in 674.715 seconds. Every one
of the 120 blocks and 360 measured compilations was accepted: twelve complete
case/registration outcomes, with no failed or unsupported case. All 240 warmups
remain retained. No measurement was repeated after inspecting its result.
The unchanged report tool independently replayed 120 blocks and verified 515
campaign files. Its result remains `publication_eligible=false`.

The [raw evidence archive](https://github.com/FlagQuantum/FlagQuantum-evidence/releases/download/evidence-compiler-routing-2026-11-23.1/routing-public-evidence.tar.gz)
contains the exact report, blocks, worker/supervisor logs, registrations, release
metadata, original freeze, operational drivers, build/preparation failures,
resource and evaluator controls, activity heartbeats, replay records and runner
restoration. The [evidence index](evidence/compiler-installed-wheel-controls.json)
records its byte count and SHA-256, all case statistics, and the private diagnostic
archive hash. Detailed host process inventories remain on the study host;
they are excluded from this public archive. The public archive is retained as a release asset in the existing external
evidence repository; its downloaded size and SHA-256 were verified before
publication. The adjacent manifest lists all 573 original file hashes and
SHA256SUMS binds both assets. The source repository retains this compact report,
the original freeze and the existing numerical ledger. Neither location
promises indefinite retention or independent build/host attestation.

The table reports the range across all thirty accepted records for gate count
and dependency depth. Time is the median of ten block medians, each computed from
three measured compilations after two warmups. The adjacent range retains the
minimum and maximum block medians, including the slower Qiskit grid block.
Memory is the **maximum aggregate cgroup block peak**, including imports,
warmups, export and descendants; it is not compilation-only allocation or sampled
RSS. Full quartiles, individual samples and raw byte values remain archived.

| Case | Registration | CZ range | Two-body depth range | Median ms | Block-median range ms | Maximum block MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ROUTE-LINE-06 | `fq-integrated-ordinary` | 96–96 | 90–90 | 17.554 | 17.424–17.703 | 154.19 |
| ROUTE-GRID-08 | `fq-integrated-ordinary` | 116–116 | 63–63 | 23.288 | 23.129–23.449 | 153.89 |
| ROUTE-LINE-12 | `fq-integrated-ordinary` | 432–432 | 206–206 | 55.849 | 55.443–56.332 | 159.27 |
| ROUTE-LINE-06 | `fq-integrated-artifact` | 84–84 | 78–78 | 322.569 | 320.835–341.019 | 156.64 |
| ROUTE-GRID-08 | `fq-integrated-artifact` | 110–110 | 63–63 | 386.496 | 384.833–389.891 | 157.38 |
| ROUTE-LINE-12 | `fq-integrated-artifact` | 414–414 | 192–192 | 1563.259 | 1551.933–1673.022 | 172.62 |
| ROUTE-LINE-06 | `qiskit-release-restored` | 86–86 | 80–80 | 8.151 | 7.907–8.251 | 66.47 |
| ROUTE-GRID-08 | `qiskit-release-restored` | 118–124 | 68–72 | 9.100 | 8.980–17.782 | 68.17 |
| ROUTE-LINE-12 | `qiskit-release-restored` | 430–438 | 228–252 | 20.873 | 20.545–21.237 | 83.04 |
| ROUTE-LINE-06 | `cirq-release-restored` | 86–86 | 80–80 | 114.853 | 113.988–115.662 | 148.71 |
| ROUTE-GRID-08 | `cirq-release-restored` | 138–138 | 91–91 | 148.616 | 147.643–151.679 | 145.49 |
| ROUTE-LINE-12 | `cirq-release-restored` | 456–456 | 240–240 | 523.022 | 517.956–528.461 | 161.75 |

The artifact workflow emits 84/110/414 CZs, versus 96/116/432 for the ordinary
workflow on line-6/grid-8/line-12. Its gate counts and dependency depths are lower
than both compared upstream workflows on these three inspected controls. This
comes with 18.4/16.6/28.0 times the ordinary workflow's median compilation time.
The two workflows have different validation and serialization obligations; their
difference is not an isolated causal estimate for the resynthesis pass.

The ordinary workflow is 2.15/2.56/2.68 times Qiskit's median compilation time,
with larger aggregate block-memory peaks in all three cases. It also emits more
CZs and greater depth than Qiskit and Cirq on line-6. On grid-8 and line-12 it
emits fewer CZs and lower depth than the compared upstream workflows. The artifact
workflow costs approximately 39.6/42.5/74.9 times Qiskit's median compilation time.
These losses are improvement targets, not evidence to hide by choosing the best
FlagQuantum path per case. The ordinary workflow is faster than the measured
Cirq workflow in all three cases; artifact compilation is slower in all three.
No confidence interval, statistical significance or SDK-wide superiority is
claimed from ten process blocks and three already inspected small programs.

The maximum numerical residual is approximately 1.852e-14, below the unchanged
1e-8 semantic bound. All accepted outputs meet topology, full-CZ and identity
output-order obligations. Six-wire checks cover the complete operator up to a
global phase; three probes at widths eight/twelve cannot prove universal
operator equivalence. These numeric routing controls do not certify symbolic
parameter preservation, gradients, noise, hardware execution or wider programs.

## Operational record and verification limits

Admission occurred at 2026-11-23T05:36:41Z after existing jobs drained naturally.
The selected CPU was 96 with SMT sibling 288; performance governor and boost
remained enabled. The host has two EPYC 9K84 sockets, 192 physical cores,
approximately 2.2 TiB RAM and Linux 6.8.0-87. Each worker used one thread and a
16 GiB no-swap cgroup. This is local CPU compilation with complex128/float64
numerics, no numerical gradient method and no sharded or replicated workload.
Compilation is exact within the declared numerical acceptance tolerances.

The private CPython 3.12.14 build uses default configuration without PGO/LTO.
The initial build lacked bz2/lzma/sqlite; its failed preparation is retained.
Six privately extracted development packages enabled the complete replacement
interpreter, without installing system packages. The installed environment and
wheel locks are frozen; timing cannot be extrapolated to other Python builds.
All fifteen resource controls, twelve installed native semantic checks and
146 independent evaluator/tamper controls passed before measurement.

The measured campaign finished before a session interruption. The original
replay log was empty and no replay result survived. That interruption is retained
explicitly; measurement was not rerun. A separately named producer-free recovery
replay succeeded, followed by the unchanged official report tool's complete
replay. The latter used its own one-CPU/16 GiB cgroup, completed in 141.519 seconds
and peaked at 277676032 bytes. Those evaluator costs are not SDK compile times.

Measured heartbeats observed no foreign CI work. A late post-run snapshot at
05:54:24Z matched the 05:36:35Z package and stable-file baseline, and all frozen
inputs reverified. The interrupted replay introduced an observation gap, so this
is not continuous independent host attestation. Background system services and
SMT were retained. All sixteen original CI runner label sets were restored after
the report completed. The restoration record includes exact runner identities,
label sets and verification; no machine or service was restarted.

The required local smoke/unit tier previously reached 21528 passes, 796 skips,
one historical-origin failure and two disk-related setup errors. It was not a
passing full tier. The two disk-blocked checks passed unchanged in a separately
recorded Linux rerun. Original PR declaration CI failures concerned revision-field
naming and inline origin representation; the retained gzip and existing ledger
correct those metadata issues without changing validators or scientific inputs.
Both originally failing tests pass locally after the correction. Fresh exact-head
CI is still required before this documentation can merge. The unrelated wide-state
CI termination remains tracked in issue #1318; its cause is not established.

## Evidence-storage CI correction

The first results commit mistakenly tracked the 12987160-byte raw archive in
this source repository. Quality, the CPU default tier (21477 passed, 721 skipped)
and coverage shard 186 (602 passed, three skipped) each failed the unchanged
1800000-byte hygiene limit. The archive has moved byte-for-byte to the existing
external evidence repository; its original SHA-256 and all scientific results
are unchanged. No limit, test, skip or allowlist changed.

Other original CI failures remain distinct. Coverage shard 27 failed downloading
the source artifact before tests. The GPU gate refused GitHub's null binary
classification for a truncated source blob, then its Git fallback encountered a
connection reset and SSL timeout before the job was canceled. The narrow source
metadata repair is a separate integration change. These transport failures are
not compiler results or passing gates. Fresh complete exact-head CI is required
before merging this corrected report.

## Next development decision

Keep the current production compiler unchanged by this evidence-only update.
Prioritize whole-workflow compilation cost: establish where the artifact path
spends its extra time on external application programs, and distinguish native
routing, synthesis, legality/provenance verification and serialization. Reuse
existing stage instrumentation and upstream mechanisms rather than adding
another unproven cache. Retain complete semantic and evidence checks.

Genuinely held-out real programs remain outstanding. First freeze a corpus
selection/novelty policy and source/configuration identities publicly, then draw
programs without inspecting their outcomes. New parameters drawn from already
inspected QAOA/Ising/hopping/QFT/repetition recipes do not by themselves supply
unseen-program evidence. Keep SDKBench evaluators, thresholds and resource limits
unchanged; retain all failures and regressions and compare each delivered workflow
separately. This study does not complete that qualification or the broader goal.
