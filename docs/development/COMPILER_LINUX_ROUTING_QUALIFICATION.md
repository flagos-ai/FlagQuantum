# Linux routing qualification declaration

Status: declared before measured execution on November 23, 2026. The
[freeze manifest](evidence/compiler-linux-routing-freeze.json) fixes participants,
configuration, wheel identities, the complete unchanged SDKBench Python harness,
fixture/reference bundle, thresholds and budgets. No campaign has run at this
freeze. This study checks inspected historical controls, not unseen applications.

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
