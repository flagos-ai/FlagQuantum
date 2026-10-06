# Two-Node Distributed Runbook

This runbook covers operating the two-node lane that
`.github/workflows/scheduled-hardware.yml` runs: two hosts, one rank per host,
one workload sharded across the node boundary. It records how two sites were
prepared, the preflight the launcher performs, and the evidence a passing run
produces.

The lane runs one workload per invocation, selected by `--probe`: a statevector
workload that shards amplitudes and an MPS workload that shards sites. They
share the pair, the plan, the preflight and the watchdog, because those are
properties of the pair; each brings its own circuit, its own artifact schema
and its own capability entry.

Preparing a pair and passing the preflight establishes readiness and transport
health. It is not scalability evidence; the
[evidence boundary](#evidence-boundary) section states what the recorded runs
do and do not establish.

## Recorded sites

Two sites have been prepared this way. Substitute your own inventory: every
command below takes host, interface, interpreter, and path from flags rather
than from these tables, and no address or hostname from either site is recorded
in this repository.

| Site | Launch host | Peer | Devices per host | Peer route | Interpreter |
| --- | --- | --- | --- | --- | --- |
| eight-A800 pair (the recorded two-node artifact) | one host of the pair | the other host | 8 x NVIDIA A800-SXM4-80GB | one management interface, addressed by address because the peer does not resolve by name | one absolute path on a shared NFS mount |
| DGX A100 pair (the preparation procedure recorded below) | `gpu-node-0` | `gpu-node-1` | 8 x NVIDIA A100-SXM4-40GB | `192.0.2.1/23` and `192.0.2.2/23` on `enp226s0` | node-local Conda, cloned over the data plane |

The A800 runtime reported Python 3.12.13, PyTorch 2.13.0+cu130, CUDA runtime
13.0, and NCCL 2.29.7. The recorded run reached the peer over the management
interface with InfiniBand disabled, so NCCL used its socket transport; the
paired data-plane interfaces were left unused. Recheck every version rather
than trusting this snapshot.

## Requirements a pair must meet

- Each host exposes at least one device; the lane places one rank per host.
- A filesystem both hosts mount, for staging, rank output, and checkpoints.
- Non-interactive SSH from the launch host to the peer, addressed by a name or
  address the **launch host** resolves.
- One interpreter path both hosts can execute, ideally on the shared filesystem
  so the pair cannot drift apart.
- A rendezvous port free on the launch host and reachable from the peer.
- An NCCL route -- RDMA or socket -- that the run's debug log can confirm.

## Site preparation: recorded DGX A100 pair

The hosts expose paired data-plane interfaces:

| Interface | Node 0 | Node 1 |
| --- | --- | --- |
| `ens101` | `192.0.2.6/24` | `192.0.2.5/24` |
| `ens102` | `192.0.2.8/24` | `192.0.2.7/24` |
| `ens103` | `192.0.2.10/24` | `192.0.2.9/24` |
| `ens104` | `192.0.2.12/24` | `192.0.2.11/24` |
| `ens105` | `192.0.2.14/24` | `192.0.2.13/24` |
| `ens106` | `192.0.2.16/24` | `192.0.2.15/24` |
| `ens107` | `192.0.2.18/24` | `192.0.2.17/24` |
| `ens108` | `192.0.2.20/24` | `192.0.2.19/24` |
| `ens201` | `192.0.2.22/24` | `192.0.2.21/24` |

Do not assume these interfaces provide RDMA merely because their addresses are
paired. Validate the RDMA mapping and link state before selecting NCCL network
settings.

### SSH access

The launch host uses a dedicated Ed25519 key to reach the peer. Never commit
the private key or an `authorized_keys` file. A suitable local-only SSH
configuration is:

```sshconfig
Host fq-node2
    HostName 192.0.2.2
    User root
    IdentityFile /root/.ssh/id_ed25519_flagquantum
    IdentitiesOnly yes
```

Protect the configuration and test non-interactive access:

```bash
chmod 600 /root/.ssh/config
ssh -o BatchMode=yes fq-node2 hostname
ssh -o BatchMode=yes fq-node2 nvidia-smi -L
```

Expected remote hostname: `gpu-node-1`. Passwords and private keys must
never be placed in this repository, shell history, benchmark payloads, or test
artifacts.

### Project-local Conda layout

Both nodes use the same project-local layout instead of a system Conda:

```text
/srv/quantum-demo/FlagQuantum/
├── .micromamba-bin/bin/micromamba
├── .conda-root/
│   ├── bin/conda
│   └── envs/flagquantum-dev/
└── FlagQuantum/                 # repository and editable project
```

Node 0 originally bootstrapped the base environment with:

```bash
/srv/quantum-demo/FlagQuantum/.micromamba-bin/bin/micromamba create \
  -y \
  -r /srv/quantum-demo/FlagQuantum/.conda-root \
  -n base \
  -c conda-forge \
  python=3.12 \
  conda
```

It then created the development environment with:

```bash
/srv/quantum-demo/FlagQuantum/.conda-root/bin/conda create \
  -y \
  -n flagquantum-dev \
  -c conda-forge \
  python=3.12 \
  pip
```

The repository-level environment intent remains:

```yaml
name: flagquantum-dev
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pip
  - pip:
      - -e .[dev,jax]
```

For node replication, export a fully pinned environment on node 0, remove the
plain `flagquantum==...` entry produced for the editable install, create the
environment on node 1, and reinstall FlagQuantum from the node-local source:

```bash
conda activate flagquantum-dev
conda env export | sed '/^prefix:/d' > ../environment.flagquantum-dev.yml

# After transferring and creating the environment on node 1:
/srv/quantum-demo/FlagQuantum/.conda-root/envs/flagquantum-dev/bin/python \
  -m pip install -e '/srv/quantum-demo/FlagQuantum/FlagQuantum[dev,jax]'
```

The site used the following environment-local PyPI mirror when the default
index was too slow:

```bash
/srv/quantum-demo/FlagQuantum/.conda-root/envs/flagquantum-dev/bin/python \
  -m pip config --site set global.index-url \
  https://pypi.tuna.tsinghua.edu.cn/simple
```

At initial setup, the node 0 runtime reported:

```text
Python: 3.12.13
PyTorch: 2.13.0+cu130
PyTorch CUDA runtime: 13.0
NCCL: 2.29.7
Visible GPUs: 8
```

Recheck rather than trusting this snapshot. Both nodes must report compatible
driver versions and matching Python, PyTorch, CUDA runtime, NCCL, JAX, and
project dependency versions before a test.

```bash
python -c 'import sys, torch; print(sys.executable); print(sys.version); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.nccl.version()); print(torch.cuda.is_available()); print(torch.cuda.device_count())'
```

Do not copy `.conda-root` or `.venv` as part of the ordinary source `rsync`.
Conda environments contain absolute paths and machine-local links. Rebuild from
a pinned export or use an explicitly relocatable packaging workflow by default.

This site has one controlled exception: both nodes use the identical absolute
environment path and the same x86_64 platform. A stopped environment can
therefore be cloned exactly over the data plane while preserving hard links:

```bash
# Stop all users of the destination environment and remove only the incomplete
# destination environment before running this command.
rsync -aH --numeric-ids --partial \
  -e 'ssh -i /root/.ssh/id_ed25519_flagquantum -o IdentitiesOnly=yes' \
  /srv/quantum-demo/FlagQuantum/.conda-root/envs/flagquantum-dev \
  root@192.0.2.5:/srv/quantum-demo/FlagQuantum/.conda-root/envs/
```

The initial clone transferred about 6.5 GB in about 82 seconds at approximately
75.6 MB/s over `ens101`, compared with an unusably slow external wheel download.
After a raw clone, verify imports, GPU visibility, and package consistency; do
not assume that successful file transfer proves the environment is usable.

```bash
/srv/quantum-demo/FlagQuantum/.conda-root/envs/flagquantum-dev/bin/python \
  -m pip check
/srv/quantum-demo/FlagQuantum/.conda-root/envs/flagquantum-dev/bin/python \
  -c 'import torch, jax, flagquantum; print(torch.__version__, torch.version.cuda, torch.cuda.nccl.version(), torch.cuda.device_count(), jax.__version__, flagquantum.__file__)'
```

## Site preparation: recorded A800 pair

This pair has neither a node-local environment nor a name the peer resolves by.
Both properties are deliberate answers to problems the first site hit:

- The interpreter lives on the shared NFS mount, at one absolute path both
  hosts execute. There is no environment replication step and no environment
  drift to check for. The lane still verifies that the peer can run it.
- The peer is addressed by address, because the launch host cannot resolve the
  peer's name. The lane's SSH runs as the launch host, so it needs a peer name
  that resolves *here*, not one that resolves on the peer.

Everything else -- the shared filesystem, the interfaces, the NCCL route -- is
site policy the launcher takes from flags and records in `plan.json`.

## Source synchronization

The lane stages the tree itself: `--run` synchronizes `--source` into
`--staging` with `rsync -a --delete`, excluding version-control, cache, and
environment directories, and then verifies on both hosts that the interpreter
imports FlagQuantum **from the staged tree** rather than from an installed
copy. Use it rather than synchronizing by hand; a hand-copied tree and a staged
tree can disagree, and the artifact would then name a revision neither host
ran.

The staged tree is then compiled once, on the launch host, and every rank is
given `PYTHONDONTWRITEBYTECODE=1`. A shared `--staging` directory means every
rank imports the same files, and left to itself each of them writes
`__pycache__` beside the source it imported: two ranks from two hosts writing
the same bytecode file over NFS race, and the loser can stall inside the kernel
for long enough to exhaust the run timeout. Compiling once before any rank
starts, and forbidding rank-side bytecode writes, removes the race rather than
tolerating it. A staged tree that cannot be compiled stops the lane before a
rank starts, and the staging report printed before either rank is released
records how many bytecode files the staging directory held at that moment:
`bytecode_files`, which is zero.

If you must synchronize by hand, preview every synchronization first and treat
the launch host as the single source of truth:

```bash
rsync -azn \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.mypy_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='.inductor-cache/' \
  --exclude='*.pyc' \
  /srv/quantum-demo/FlagQuantum/FlagQuantum/ \
  fq-node2:/srv/quantum-demo/FlagQuantum/FlagQuantum/
```

After reviewing the dry-run output, remove `n` from `-azn` to synchronize.
Avoid `--delete` until the destination policy and generated-artifact handling
have been reviewed. Because FlagQuantum is installed editable, ordinary Python
source changes become visible without reinstalling. Reinstall after dependency,
entry-point, package-layout, or compiled-extension changes.

## Connectivity preflight

The launcher asks nine questions before it starts anything, and refuses to
launch when any answer is no. They appear as one line per check on stdout and
are retained in `preflight.json`:

```text
peer_is_reachable                 the peer answers over ssh
this_host_is_the_launch_host      this host is the launch host plan.json names
launch_address_is_local           the peer address routes out of this host
launch_host_has_a_device          nvidia-smi reports at least one device
peer_has_a_device                 the same, on the peer
launch_host_has_the_interface     the NCCL interface exists here
peer_has_the_interface            the same, on the peer
peer_has_the_interpreter          the peer executes the named interpreter
rendezvous_port_is_free           the rendezvous address binds here
```

The rendezvous port is chosen rather than assumed. The hosts are shared, and a
preflight against the default 29500 once refused a real launch because a
neighbour already held it. Pass `--master-port` only when a recorded run needs
to be reproduced exactly.

To reproduce the reachability checks by hand:

```bash
ping -c 3 192.0.2.2
nc -vz -w 3 192.0.2.2 22
ssh -o BatchMode=yes fq-node2 hostname
```

Ping failure alone is inconclusive if ICMP is filtered. Test the intended
rendezvous port in both directions with `nc`. Do not assume SSH reachability
proves that arbitrary rendezvous or NCCL ports are allowed.

## RDMA and NCCL preflight

Run on both nodes:

```bash
ibdev2netdev
rdma link show
ibstat
ls -l /sys/class/infiniband
```

Record the actual HCA-to-interface mapping and link state before setting
`NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, or `GLOO_SOCKET_IFNAME`. Begin with an
explicitly bounded NCCL smoke test and retain `NCCL_DEBUG=INFO` diagnostics.
Do not hard-code an interface policy from the inventory table alone.

The launcher writes an NCCL debug log per rank and reads the transport back out
of it: which interface was used, whether it was the configured one, and whether
socket or RDMA carried the exchange. A run whose log does not confirm the route
it was configured for fails rather than reporting a transport it cannot show.
The recorded pair reached the peer with `NCCL_IB_DISABLE=1`, so RDMA is
untested there and the artifact says so in its blockers.

## Running the lane

On the launch host, one command runs the whole lane:

```bash
python tools/multinode_launch_plan.py --run \
  --probe mps \
  --peer-host fq-node2 \
  --staging /nfs/fq-multinode-run \
  --checkpoint-directory /nfs/fq-multinode-run-checkpoints \
  --report-directory hardware-run
```

`--probe` selects the workload: `statevector` (the default), `mps`, or `tn`.
Everything else is shared.

`--peer-host` names the SSH target for node 1, so pass this site's alias as
above. The tool's own default is the maintainer's launch host: it resolves
there and will not resolve here, so a command that omits the flag reports a
preflight failure against a host that has nothing to do with this pair.

The lane is always two nodes. `--local-world-size` says how many ranks each of
them runs, and it defaults to one:

```bash
python tools/multinode_launch_plan.py --run \
  --probe tn \
  --peer-host fq-node2 \
  --local-world-size 2 \
  --staging /nfs/fq-multinode-run-tn \
  --checkpoint-directory /nfs/fq-multinode-run-tn-checkpoints \
  --report-directory hardware-run-tn
```

The default stays at one because that is the shape every recorded artifact was
taken on: a wider default would leave the checked-in evidence describing a lane
that no longer exists. A width is refused unless it is one the selected
workload can actually shard, and the rule is per workload rather than one rule
for the lane:

| Probe | Accepted widths | Why |
| --- | --- | --- |
| `statevector` | powers of two | the planner shards amplitudes by rank address bits |
| `mps` | up to six | every rank has to own at least one of the circuit's six sites |
| `tn` | one, two, or four | the probe splits four declared slices, so the width has to divide four |

The refusal happens before the launch, and the probe refuses the same shapes
again before it builds a process group, because a shape mismatch found after
that reports as a collective that never completes rather than as the shape that
was refused. The two widths the launcher has to know before a launch -- the MPS
circuit's qubit count and the tensor-network slice count -- are declared in both
`tools/multinode_launch_plan.py` and the probe that enforces them, and a unit
test reads both copies and fails if they drift apart.

Each node runs one `torchrun` that starts all of its ranks from one environment,
so the NCCL debug log is scoped to the node rather than to the rank. At one rank
per node the two are the same file; above that the file holds every local rank's
view, and the artifact records which of the two it read.

It takes three shared paths, and the names matter because the lane empties two
of them:

| Flag | Role | How it is treated |
| --- | --- | --- |
| `--staging` | the tree both ranks execute | synchronized into with `--delete` and byte-compiled once before any rank starts; must be named `fq-multinode*` and sit outside the source tree |
| `--output-directory` | per-node JSON and NCCL debug logs (defaults to `<staging>-out`) | emptied before the ranks start |
| `--checkpoint-directory` | the rank checkpoints a restarted rank resumes from | emptied before the ranks start; both ranks must see the same directory |

The output and checkpoint directories are emptied for one reason: a stale node
record left by an earlier run would otherwise be published as the evidence of a
run that failed before writing its own, and a checkpoint left by an earlier run
would be resumed from instead of the one this run wrote. All three names carry
the `fq-multinode` prefix so a mistyped path cannot be wiped.

The command refuses to start when any preflight answer is no, stages the tree,
and supervises both nodes as one unit with `tools/run_multinode_watchdog.py`.
On exit the report directory holds:

```text
plan.json                 the exact commands and environment both nodes were given
preflight.json            the nine answers
watchdog.json             which node exited with which code, and why the run ended
watchdog-logs/            one log per node
node-0.json               the artifact: every rank's record, written by world rank 0
nccl-node-{0,1}.log       the transport both nodes negotiated
```

The artifact is written once, by the node holding world rank 0, because it
carries every rank's record; a file per rank would be the same evidence written
several times over. A run at a width above one therefore still produces exactly
one `node-0.json`, and the shape it was taken at is in its `scope` block.

The launch fails when the watchdog wrote no verdict, so a run that could not
report is distinguishable from a run that reported a failure.

Replicated per-rank execution can be used as a transport smoke test, but it
must not be reported as capacity scaling.

### Stopping a lane that will not stop

The watchdog signals the process group it started, which is enough on the
launch host. A remote rank reached over `ssh` is not in that group: killing the
client leaves the remote `torchrun` and its workers running, still holding
their devices and still joined to the rendezvous. A `torchrun` worker that is
killed while a collective is outstanding does not necessarily finish
unwinding either, so a lane can leave processes behind even when it was
terminated cleanly from the outside.

That matters because the leftovers are not inert. A stale worker holds its
device, is still a member of a process group, and can answer a rendezvous a
later lane dials. Two lanes sharing one rendezvous produce exactly the failure
that reads as a FlagQuantum hang: one rank waiting inside a FlagQuantum
exchange while its peer is already inside the next collective of a different
run. Confirm the hosts are quiet before reading any multi-rank timeout as a
product defect:

```bash
pkill -9 -f 'probe_cuda_multinode'
pkill -9 -f 'torch[.]distributed[.]run'
ps -eo args | grep -E 'probe_cuda_multinode|distributed[.]run' | grep -v grep
nvidia-smi --query-compute-apps=pid --format=csv,noheader
```

Both listings have to be empty. The lane reaps on its own as well: when a run
ends for any reason but success, the watchdog runs each job's cleanup command,
which is the launcher's `--cleanup` on the launch host and the same command
over `ssh` on the peer.

## Recorded two-node evidence

Three artifacts have been recorded from this lane, all taken on the eight-A800
pair at two nodes by one rank each, one device per rank. That shape is in each
artifact's `scope` block. The lane can be run wider -- each probe accepts any
width it can actually shard, up to the devices its host has -- but no wider
recording has been made, so nothing here claims one.

Widths above one have been exercised rather than recorded, on the same pair:
statevector at two and four ranks per node, tensor network at two, MPS at two
and three. Each shape completed with `status="passed"` and both claim flags
false, at wall times within a second or two of the one-rank-per-node shape,
because the workloads are small enough that the extra ranks divide work that
was never the cost. They are reported here as a shape that runs, not as
evidence of anything scaling: a wider artifact would need the same numerical
comparisons the recorded one carries, and none of these was retained.

`artifacts/cuda_multinode_statevector_a800_jp171_jp172_20260930.json` covers
the amplitude-sharded statevector workload:

- the sharded forward pass, validated against a single-device complex128
  reference;
- the exact adjoint gradient, compared with the same reference;
- an owner-sharded optimizer step;
- a checkpoint written by one run and resumed by a restarted one, with the
  resumed leg reproducing exactly the steps the uninterrupted run computed
  after the checkpoint;
- a full-state export, which reassembles the amplitude shards through the
  runtime's own `gather_distributed_statevector` and times that gather. The
  exported state is the product, the gather is what produces it, and the leg
  records the canonical logical basis order plus a per-iteration comparison
  against the single-device reference, so the vector and its duration cannot come
  from different calls. Its durations are never read as a speedup; an all-gather
  has none to report.

Every numerical metric in it is at round-off, both ranks own half the
amplitudes, and the communication it reports is inter-node with none
intra-node. Its route was read from the NCCL debug log as RoCE rather than the
socket fallback, on the interface the plan was given, and it carries five
synchronized forward samples taken after two warmups. It carries one blocker --
`two_node_pair_only_no_wider_topology` -- and reports
`scalability_claim_allowed` and `release_gate_allowed` false. The
toy-circuit blocker it used to carry is gone, and it went the way the
others did: the probe's forward circuit is now the narrowest rung of the
workload manifest it names -- twenty-two wires over eight layers, one
hundred and seventy-six bound leaves -- and the artifact records that
count against the count the manifold declares, so the blocker's own
premise no longer holds rather than being dropped from the list.

Two blockers were retired by observation rather than by declaration. The staging
audit now profiles the measured region and finds no explicit host transfer in
it, so each sample is device work end to end. And the probe exports the whole
amplitude vector as the product of a leg of its own: the amplitude shards are
reassembled by the runtime's own gather, the leg records the canonical logical
basis order and compares the assembled vector against the single-device
reference as it is taken, and it times the gather so the vector and its duration
cannot come from different calls. That is a materialization the workload needs,
not a full-object gather taken to check an answer, so
`validation_only_tiny_full_state_gather` is gone. The two blockers that remain
are the pair's own scope and the circuit's own size; no measurement on this
hardware can retract either.

`artifacts/cuda_multinode_mps_a800_jp171_jp172_20260930.json` covers the
site-sharded MPS workload. Six qubits at two ranks gives three owned sites per
rank, so the middle adjacent gate straddles the ownership boundary and an
ordinary gate exercises the exchange:

- the sharded forward pass, gathered for validation and compared with the same
  complex128 reference;
- an exact reverse gradient, with the layer-boundary halo prefetched over the
  inter-node transport rather than reconstructed, and one parameter that both
  ranks own because it is bound on both halves;
- three optimizer legs that checkpoint and resume, compared step for step with
  an uninterrupted run of the same length;
- the optimizer's initial loss checked against the reference expectation, so
  the training objective is tied to the exact statevector rather than to
  itself;
- a full-state export, which rebuilds the whole MPS through the runtime's own
  `gather_distributed_mps` and times that gather. It is a leg rather than a
  check: the exported state is the product, the runtime gather is what produces
  it, and the leg records the canonical site order and a per-iteration
  comparison against the single-device reference so the vector and its duration
  cannot come from different calls. Its durations are never read as a speedup;
  an all-gather has none to report.

Its numerical metrics are at round-off, the two ranks report distinct host and
device identities, and the reverse attributes 32 bytes of layer halo to the
inter-node tier and none to the intra-node tier. The forward boundary figures
in `numerical_metrics` are the workload's totals -- four messages and 192 bytes
-- while each rank record carries the 96 bytes that rank took part in; the two
are separate fields because ownership is rebalanced as the circuit runs, so a
rank can observe a boundary gate spanning two other ranks' sites without
exchanging anything for it. The artifact is checked against that arithmetic. Its
route was read from the NCCL debug log as RoCE, and it carries five
synchronized forward samples taken after two warmups. It carries two blockers --
`two_node_pair_only_no_wider_topology` and `host_staging_in_measured_region`
-- and reports both claim flags false. The toy-circuit blocker is gone: the
measured circuit is now the contract's own accepted rung, eight thousand one
hundred and ninety-two sites at bond sixty-four, read from the ladder file
the manifest names, and the artifact records the thirty-one leaves it bound
against the thirty-one that rung declares. The
staging blocker is the audit's finding that host transfers sit inside the
measured region. The full-state gather blocker is gone, because the probe now
exports the whole MPS as the product of a leg of its own rather than gathering
it to check an answer. Unlike the statevector artifact it belongs to a
capability that also claims single-host and multi-GPU support, so it is the
multi-node leg of a broader entry rather than an entry of its own.

That finding was then taken apart rather than left as a label. Most of the
transfers were the distributed accounting: the per-rank site footprint, the
bond dimensions and the canonicalization counters are host-known integers, and
each was written into a device vector one element at a time, which lifts a
fresh device scalar per element inside whatever region is being timed.
Assembling those vectors from device scalars instead, reading each gathered
table back once rather than once per field per rank, and initialising the
rank-owned `|0>` from an `arange` rather than from the host integer one, takes
the same forward workload at one rank from 57 explicit transfers to 13 -- 43
host-to-device copies down to six. Binding the circuit's angles to the device,
as the statevector probe already does, takes it to nine.
`classify_explicit_host_transfer` is the classifier both audits use, so the
rule is a measured one: a device scalar times a Python int, and a stack of
such products, stage nothing.

Nine is a floor for one configuration, and that configuration is not the one
this lane profiles. The profiled leg is

```python
execute_torch_distributed_mps_forward(circuit, device=device, max_bond=MAX_BOND)
```

with no `rebalance_threshold`, so it runs at the default `1.5`. Every two-site
gate then calls `rebalance_mps_if_needed`, which calls
`global_mps_tensor_bytes`, which ends in `sizes.tolist()` -- one device-to-host
read per two-site gate, on top of the seven reads that remain in any
configuration. The floor of nine is the floor of the *compiled* path, where
`rebalance_threshold=inf` short-circuits that collective because a compiled
run's buckets must not move between ranks mid-layer. The reverse leg of this
same lane uses exactly that configuration, which is why the number and the
compiled path were easy to conflate. The forward leg does not.

That census was a reading of the code, and it was superseded by a measurement of
the running region. Wrapping the host-read paths of `torch.Tensor` so that each
one records its caller attributes every explicit transfer in the profiled
region to the call site that issued it, and the region carries thirty-six of
them: twenty-eight device-to-host and eight host-to-device. The reads are

| Count | Call site | What it reads |
| --- | --- | --- |
| 14 | `distribution.global_mps_tensor_bytes` | the load-balancing cost vector, once per two-site gate |
| 5 | `transport._recv_tensor_p2p` | the point-to-point control descriptor that sizes a receive buffer |
| 4 | `metadata_transport.all_gather_json` | the truncation records the global error budget policy reads |
| 3 | `canonicalization.canonicalize_rank_owned_mps` | the metric table, the centre norms, and the residual |
| 1 | `distribution.global_mps_bond_dimensions` | the global bond dimensions |
| 1 | `forward.execute_torch_distributed_mps_forward` | the per-rank tensor byte vector |

The host-to-device copies are the receive buffers those descriptors size, the
device scalars the counters are scaled onto, and the metadata the collective
encodes. None of the thirty-six stages a rank payload. The one site that did --
the per-site host-to-device lift in `_device_footprint` -- was repaired before
this measurement was taken, which is why it appears in the table only as the
fourteen reads of the vector it now assembles on the device.

That makes the blocker narrower than its text, and the blocker text is what was
corrected rather than the blocker. `classify_explicit_host_transfer` classifies
by event name and count, so a one-scalar readback is one event exactly as a
staged tensor is, and no arrangement of a dynamic-ownership forward can bring
the count to zero. It can be brought to twenty by running with
`rebalance_threshold=inf`, which removes all fourteen of the load-balancing
reads and one host-to-device copy -- and that measurement is the reason not to:
the frozen run issues four boundary gates and eight boundary messages where the
dynamic run issues three and six, so the cost probe pays for itself in
inter-node traffic and trading it away would report a cleaner profile while
moving more bytes over the fabric. `host_staging_in_measured_region` therefore
stands for MPS, and it stands for a control-plane readback profile rather than
for staging. The claim remains a latency claim over a workload whose measured
region contains bounded, disclosed host accounting.

The cut width was swept rather than declared, and the sweep is a leg of its own
rather than part of the declared workload. The declared circuit could not carry
it: its Schmidt profile is flat at rank two across all five cuts, so moving the
owned site boundary along it exchanges the same bond wherever it lands and any
two numbers from it would be the same number. The sweep therefore binds its own
circuit, which rotates each adjacent pair in turn and so has a profile the cuts
can tell apart -- ranks `2, 4, 4, 4, 2`, read from the singular values of the
exact statevector rather than from the run. The probe then re-runs the same
`execute_torch_distributed_mps_reverse` the training loop uses, under every site
plan the launched shape can hold with a boundary moved: at two ranks that is the
five positions of the single boundary, and at a wider shape it walks each rank
boundary across its neighbours in turn. Each leg is compared with the exact
complex128 expectation and gradient and raises on disagreement, so a leg that
cut the wrong circuit fails the run rather than being recorded as a narrow
width. What each leg reports is its own inter-node layer halo, the counter the
transport attributes per tier; the widths that crossed bytes are the ones
`cut_width_claim_blockers` counts, so a placement that exchanged nothing leaves
`inter_node_cut_width_not_swept` standing. In the recorded run the five
placements crossed 576, 1344, 1344, 832 and 320 bytes at widths 2, 4, 4, 4 and
2, each at round-off against the reference.

A plan at more than two ranks holds one boundary per adjacent rank pair, so it
carries several cuts at once and is named by the heaviest bond it has to carry;
naming it by a lighter one would claim a rank-four bond was carried by a leg that
crossed nothing of the sort, and the widths recorded are the ones the legs
actually carried. At the recorded pair every placement has one boundary, so both
heights of the profile are placed; a wider shape would reach only the heaviest
height and would report the one crossed width, which the blocker's own rule
declines to count as a sweep.

`artifacts/cuda_multinode_tn_a800_jp171_jp172_20260930.json` covers the
slice-sharded tensor-network workload. Five qubits are contracted along a cut of
two labels, which the workload declares rather than leaving to the automatic
slicer:

- four amplitudes at four distinct bitstrings, plus a single-amplitude
  projection through the other output path, each compared with the exact
  complex128 statevector;
- the expectation and its gradient, where the two ranks report the same
  reduced expectation but different per-rank gradient contributions, so a
  reduction that dropped a rank would change the answer;
- three optimizer legs that checkpoint and resume, compared step for step with
  an uninterrupted run of the same length;
- the committed checkpoint generation read back from the shared directory, so
  the check covers every rank's file and not just the one that wrote it.

Its numerical metrics are at round-off, every byte of its reduction is
attributed to the inter-node tier, and both ranks report accelerator runtime
evidence rather than a CPU collective. Its route was read from the NCCL debug
log as RoCE, and it carries five synchronized amplitude samples taken after two
warmups. The cut width was swept across both labels of the declared cut, which
retired `inter_node_cut_width_not_swept` and
`slice_count_fixed_at_world_size`. It carries one blocker --
`two_node_pair_only_no_wider_topology` -- and reports both claim flags false.
The toy-circuit blocker is gone: the probe's circuit is now this contract's
own narrowest matched-speed rung, fourteen wires over six layers and
eighty-four bound leaves, and the artifact records those counts against the
rung it names. Two further blockers were retired by observation
rather than by declaration. The staging audit now profiles the measured region
and finds no explicit host transfer in it, so each sample is device work end to
end. And the probe exports the whole distributed state as the product of a leg
of its own: the leg runs `run_distributed_tensor_network`, which all-reduces the
rank partials and reshapes them into the full state, and it asserts the
placement that leg ran under -- two nodes, accelerator evidence, replicated
after the all-reduce -- before it reports anything. It refuses a state it did
not materialize, so the export cannot be satisfied by a summary that merely
claims one, and it re-asserts that no scalability claim was made. That is a
materialization the workload needs, not a full-object gather taken to check an
answer, so `validation_only_tiny_full_state_gather` is gone. The two blockers
that remain are the pair's own scope and the circuit's own size; no measurement
on this hardware can retract either.

The cut is declared because the automatic slicer selects by peak memory, and on
this circuit the cheapest cut is a label carried only by state-copy nodes: a
tensor with fewer than two dimensions above one. Every branch but one already
carries a zero on such a label, so slicing it yields ranks whose partial is
exactly zero and lets a run report sharded execution while one rank does the
arithmetic. The slicer now excludes those labels and this workload passes its
cut explicitly.

#### Why this workload cannot reach `release_certified`

The tensor-network slice contract is different from the statevector one, and the
difference is worth stating before anyone tries to close the gap with another
run. Its release manifest freezes a *matched-speed* protocol at six
configurations, whose acceptance configuration is `matched_speed_18q_l10_s6` --
180 trainable angles over eighteen wires and ten layers at six sliced labels.
That configuration was measured on one A800 before any payload was sealed, and
it exhausts the device. The two narrower rungs of the same frozen ladder
complete: `matched_speed_14q_l6_s4` at 0.79 GiB peak in 15.0 s and
`matched_speed_16q_l8_s4` at 8.53 GiB in 30.4 s. The acceptance configuration
peaks at 66.9 GiB and then asks for a further 16 GiB, which an 80 GiB device
does not have, and it fails in 50.7 s.

Adding the second host does not change that. The pair fails the rung with the
same 58.9 GiB resident and the same 16 GiB refused, because the reverse-mode peak
is not slice-owned -- the same finding the capacity ladder records, where the
per-rank peak is identical at world size 1 and world size 2 to within a few
megabytes. So `missing_statistically_significant_speedup_artifact` cannot be
cleared by running the frozen protocol on this pair: there is no configuration in
the freeze that both ranks can complete, and the only way to produce one would be
to re-freeze the ladder *after* measuring it, which is precisely what a
`frozen_before_release_run` manifest exists to prevent. The measurement is
therefore recorded in the manifest's own `measured_single_device_floor` rather
than quietly corrected, in the same spirit as the capacity ladder's twelve
recorded rungs.

The same structural reason blocks the capacity route.
`capacity_premise_not_established` requires one measured shape that exhausts a
single device *and* that a sharded run completes at the same frozen slicing. The
ladder shows no such shape: the rung that exhausts one device, two sliced labels
at 80930380288 bytes, fails on the pair at the same number, and the rung that
completes on one device at four sliced labels completes on the pair too. Slicing
reduces the forward peak and not the reverse one, so a pair cannot train a
tensor-network workload that one device cannot.

One further blocker is also not closable by a run:
`two_node_pair_only_no_wider_topology` is permanent because this cluster has
two hosts. `toy_circuit_parameters_only` no longer attaches to this artifact,
and it was retired by measurement rather than by argument: the circuit the
probe contracts is the contract's own narrowest released rung, and the
observation that binds it reports both the count the rung declares and the
count the built circuit bound, which is the check whose absence produced the
blocker. The host-staging blocker the statevector lane closed was
never attached to this artifact, and the validation-only-gather blocker is now
closed the same way the statevector lane closed its own: by measuring a
production full-state readout leg. Both closures are recorded above as
observations, not as declarations.

The lane therefore holds at `production_supported` with one blocker and both
claim flags false. That is the honest ceiling for this capability on this pair,
and it is a measured one.

#### The frozen capacity workload does not contract at its own slicing

The paragraph above records the capacity route as blocked by the pair. That is
true, and it is not the whole reason. Re-measuring the frozen grid at the
current revision through the producer's own `capacity-failure` role shows the
rung does not reach a memory measurement at all:

```bash
python -m benchmarks.tensor_network_release_evidence \
  --role capacity-failure --node-count 1 --local-world-size 1 \
  --slice-count 4 --checkpoint-budget-bytes 4294967296
```

On one A800 this raises `RuntimeError: tensor has too many (>25) dims` from
`_canonical_layout_pair`, reached along `checkpointing.py` -> `stages.py` ->
`complex_einsum_pair` -> `_fused_layout_bmm`. The cause is the default
slice-label selection, not the device:
`flagquantum/runtime/executors/tensor_network/training.py` chooses the
*lowest-numbered* contracted internal labels, and for the frozen `[4, 6, 2]`
grid those are the deepest labels in the network. Slicing them leaves both
operands above the twenty-five-dimension ceiling that `torch.einsum` and
PyTorch's copy machinery share, and because the operands must be transposed
before they can be grouped, the copy is not available as a fallback either.

The labels are the whole difference. Holding the circuit, the parameters, the
step count, the optimizer and the tape budget fixed and varying only the
`sliced_labels` argument, the same rung gives:

| Label choice | Labels | Peak | Rematerialized |
| --- | --- | --- | --- |
| default (lowest) | 25, 26, 27, 28 | `RuntimeError` | -- |
| highest | 446, 447, 448, 449 | 55820657664 B | 2512 |
| middle | 223, 224, 250, 251 | 4433595904 B | 0 |
| stride | 25, 125, 250, 350 | 4099927552 B | 0 |

A spread choice therefore completes in 4.10 GiB where the default cannot start,
an order of magnitude below the 51.19 GiB completion the frozen ladder records
for this rung. Two consequences follow, and both matter more than the capacity
verdict.

First, the frozen `measured_ladder` rungs are not reproducible from the
revision the manifest declares. The rung above is recorded as `completed` at
51191433216 bytes, and `_default_sliced_labels` is byte-identical at that
revision, so the recorded completions must have been measured with an explicit
label argument that the manifest does not name. A reader cannot re-derive the ladder
from the freeze, which is a provenance defect in the manifest rather than a
property of the hardware.

Second, the capacity premise was never a property of the workload. With a
spread label choice the frozen grid is a small single-device problem; with the
default choice it is unrunnable on any number of devices. Neither outcome
describes a workload that needs a second host, so
`capacity_premise_not_established` is the right verdict and it stays right until
the frozen workload is replaced with one whose *best-sliced* single-device peak
exceeds one device.

The fix for the default selection is not a change of order within the candidate
list. Choosing slice labels by plan is the correct route --
`TensorNetworkSlicingPlan` exposes `peak_bytes`, so labels can be chosen by the
criterion the capability actually cares about -- but that is an executor change
that has to be measured across the workload family before it lands, and it is
not made here. What is changed here is only the record: the default's behaviour
is named, its consequence is measured, and the frozen manifest is left alone
rather than edited after the measurement.

#### The tape budget is a capacity lever and the ladder does not freeze one

Every rung in `measured_ladder` was measured with an unbounded rematerialization
tape, so the rungs that exhaust a device measure the budget they ran under rather
than the shape. Bounding the reverse tape at 4 GiB and holding shape, slicing,
parameters and steps fixed changes the same rungs:

| Shape | Sliced labels | Unbounded | Bounded at 4294967296 B |
| --- | --- | --- | --- |
| `[4, 6, 2]` | 2 | 80930380288 B OOM | completed, 17160570368 B |
| `[4, 6, 2]` | 4 | 51191433216 B | `RuntimeError` (>25 dims) |
| `[4, 6, 3]` | 4 | 82023653888 B OOM | completed, 30072620032 B |
| `[4, 6, 3]` | 8 | 82478769152 B OOM | completed, 27926225920 B |

The rung the ladder records as the one shape that exhausts a single device --
two sliced labels at 80930380288 bytes -- completes in 17.16 GiB once the tape is
bounded. So the capacity question for this family is what the tape budget is
before it is what the sharding is, and a capacity premise that freezes a shape
without freezing a budget has not frozen the thing it is about. This is a sixth
falsified candidate, recorded in the manifest's `freeze_note` rather than in
`measured_ladder`, because a frozen ladder is a record of what was run and this
is a record of what the record omits.

The `[4, 6, 2]` four-label rung appears in both tables and fails differently in
each: unbounded it completes, bounded it raises. The budget therefore does not
monotonically help, which is the strongest form of the finding -- neither the
label choice nor the budget is a knob the manifest can leave implicit and still
claim to have frozen a capacity premise.

#### Reading the MPS gate's default output

Promoting the five statevector envelopes into `benchmarks/results/scalability`
changed what the MPS gate prints when it is run with no arguments, because that
directory is the gate's own `RESULTS` and the gate reads every `*.json` in it.
The statevector payloads are not MPS evidence, and the MPS gate has an explicit
check for that, so the default invocation now reports one blocker more than the
scoped one:

```console
$ python benchmarks/internal/evidence/mps_release_gate.py
{"capability": "distributed_matrix_product_state", "artifact_count": 6, ...,
 "passed": false, "blockers": ["production_artifact_is_not_mps_evidence",
   "missing_release_world_sizes", "missing_multinode_correctness_artifact",
   "missing_sharded_training_ownership",
   "missing_statistically_significant_speedup_artifact",
   "capacity_premise_not_established",
   "capacity_premise_evidence_not_verifiable"]}
```

The refusal is correct -- the envelopes were sealed for another capability --
but it is a statement about the directory rather than about the MPS lane, and
the MPS lane's own reading is the scoped one:

```console
$ python benchmarks/internal/evidence/mps_release_gate.py \
    --candidate benchmarks/results/smoke/release_candidates/mps_matched_speed_4x2
{"capability": "distributed_matrix_product_state", "artifact_count": 2, ...,
 "passed": false, "blockers": ["missing_release_world_sizes",
   "missing_multinode_correctness_artifact",
   "missing_sharded_training_ownership",
   "missing_statistically_significant_speedup_artifact",
   "capacity_premise_not_established",
   "capacity_premise_evidence_not_verifiable"]}
```

Both invocations need `FQ_EVIDENCE_SIGNING_KEY` in the environment; without it
the gate cannot verify a sealed payload and adds
`invalid_or_unsigned_production_artifact` to either set. The two invocations
differ only by `production_artifact_is_not_mps_evidence`, and neither is one
blocker away from passing: `missing_release_world_sizes` wants a world size of
sixteen carried by a signed envelope, which this pair cannot produce.

#### Reading the tensor-network gate's default output

`benchmarks/internal/evidence/tensor_network_release_gate.py` takes repeatable
`--candidate` directories and falls back to `(RESULTS, baseline)` when none is
given, where `RESULTS` is `benchmarks/results/scalability`. That directory is the
**statevector** capability's declared `release_artifact` and holds the five
promoted statevector envelopes. The tensor-network lane has never promoted
anything: `benchmarks/results/smoke/release_candidates/tensor_network_matched_speed`
and `.../tensor_network_single_gpu_capacity` both exist and are both empty, and
the sealed payloads the lane produced live outside the tree.

Running the tensor-network gate with no `--candidate`, therefore, evaluates the
statevector payloads as tensor-network production evidence and reports:

```console
$ python benchmarks/internal/evidence/tensor_network_release_gate.py
{"capability": "distributed_tensor_network", "artifact_count": 5, ...,
 "passed": false, "blockers": ["capacity_premise_not_established"]}
```

That output reads as "the tensor-network release is one blocker away". It is not
tensor-network evidence. The runtime-evidence envelope has no capability field,
so the gate cannot tell the two apart from the payload alone, and the
statevector envelopes happen to satisfy every tensor-network check that is not
capability-specific. The only reason it is one blocker rather than five is that
a `measured_production_run` is a `measured_production_run` regardless of which
capability sealed it.

The correct invocation names the candidate set, and then the same gate reports
what is actually missing:

```console
$ python benchmarks/internal/evidence/tensor_network_release_gate.py \
    --candidate benchmarks/results/smoke/release_candidates/tensor_network_matched_speed
{"capability": "distributed_tensor_network", "artifact_count": 0, ...,
 "blockers": ["missing_release_world_sizes", "missing_multinode_correctness_artifact",
              "missing_sharded_training_ownership",
              "missing_statistically_significant_speedup_artifact",
              "capacity_premise_not_established"]}
```

A future tensor-network promotion must therefore also pass
`--release-directory`, because `benchmarks/results/scalability` belongs to the
statevector capability and `tools/promote_release_candidates.py` promotes into
the gate's own `RESULTS` unless told otherwise. Two capabilities do not share a
release directory.

#### The tensor-network lane has no sealed payload, in this tree or outside it

The candidates the lane runs are sealed outside the checkout, under
`/nfs/fq-scratch/campaign`, and the directory holds only per-rank measurement
records rather than the release envelopes a promotion would move. A per-rank
record's keys are `commit`, `device_name`, `hostname`, `iterations`,
`local_rank`, `local_world_size`, `measured_peak_memory_bytes`,
`measured_peak_memory_bytes_by_rank`, `measurements`, `node_count`, `rank`,
`ranks`, `role`, `schema`, `software`, `timings`, `warmup`, `workload_sha256`,
and `world_size` -- there is no `artifact_class`, no `integrity`, and no
`provenance` block, so a file in that directory is a measurement input rather
than a runtime-evidence envelope. The tensor-network producer's speed role runs
the frozen ladder and writes those records; the sealing step that would turn them
into a candidate was never reached, because the ladder's acceptance rung does not
complete on either configuration.

That is the empirically checkable form of
`missing_statistically_significant_speedup_artifact`: it is not that the lane
failed to satisfy the frozen protocol, it is that no run of the frozen protocol
produced a file the gate could evaluate. `artifact_count` is `0` for that
candidate directory, and the gate reports five blockers rather than naming a
rejected payload, which is the right fail-closed reading.

### Repeatability of the measured leg

The artifacts as first recorded were produced by two independent invocations of
the lane on the same pair: the recording run and a `multinode-scheduled` tier
run, both at the same source revision, over RoCE, with the same `--measure` flag
and the same revision of the probes. Their medians agree to within a few percent
per workload -- 3.167 against 3.242 ms for the statevector forward, 20.264
against 20.303 ms for the MPS forward, and 4.283 against 4.280 ms for the sliced
amplitudes -- while the slowest of the five samples is in every case the first,
which is why the probe warms up twice before it starts recording. The medians
are therefore a property of the workload and the pair rather than of one
scheduling accident, and the spread is reported alongside them so a reader can
see how wide it is.

A third invocation, run after the claim artifact was published, measures the
same three legs on the same pair at the merged revision with both hosts
otherwise idle. The artifacts currently checked in were then re-recorded by a
fourth invocation -- the `multinode-scheduled` tier run that added the MPS
cut-width sweep -- and that column is the one the checked-in digests describe:

| Measured leg | Recording run | Tier run | Post-merge run | Sweep run | Spread |
| --- | --- | --- | --- | --- | --- |
| Sharded statevector forward | 3.167 ms | 3.242 ms | 3.247 ms | 3.188 ms | 2.5% |
| Sharded MPS forward | 20.264 ms | 20.303 ms | 21.138 ms | 21.579 ms | 6.5% |
| Sliced tensor-network amplitudes | 4.283 ms | 4.280 ms | 4.167 ms | 4.199 ms | 2.8% |

Four invocations across three revisions put every leg within 6.5% of its
smallest recorded median, and the MPS forward is again the widest. The MPS
spread widened from 4.3% to 6.5% when the sweep leg was added, which is expected
rather than incidental: that lane now runs five further reverse passes before it
times anything. It is still the number to compare a future measurement against:
a re-recording outside it is worth investigating rather than publishing.

This is an observation about the lane, not a recorded claim. The tier runs'
reports stay on the host, and the artifacts named above are the evidence. The
first tier run shared both hosts with an unrelated container holding a small GPU
allocation and the post-merge run did not, so the table also brackets what that
external load was worth: these workloads did not measure a difference between a
shared pair and an idle one, which is itself worth knowing before either state
is treated as the baseline. Every recording earlier than the one the artifacts
currently carry is superseded.

A re-recording is not free of consequences: all three artifacts share one
`source_revision`, the claim artifact's `commit` is that revision, and the
capability matrix pins the claim's `artifact_sha256` and `code_version` to it.
Re-record one lane and the other two are re-recorded with it, or the claim
builder refuses the mixed set.


### Publishing the measured leg

The three artifacts above are correctness evidence, and each of them is shaped
by the run that produced it: a training workload, a checkpoint, an optimizer
leg. A capability matrix entry that wants to publish a latency needs an artifact
whose selectors are stable and whose revision is one a reader can check out, so
`benchmarks/build_multinode_performance_claim.py` reads the three recorded
artifacts and writes one much smaller artifact that carries only the measured
leg: `benchmarks/results/local/multinode_two_node_performance_20261001.json`.

The builder copies no number it has not checked. It recomputes each recorded
evidence digest, refuses a run whose performance leg did not measure or was not
synchronized, refuses fewer than three samples or fewer than one warmup, refuses
recorded extremes that disagree with the samples they summarize, refuses a route
that was not read from the NCCL debug log as the fabric or that saw the socket
fallback, refuses any run that still reports
`production_performance_not_measured`, refuses either claim flag set true, and
refuses three artifacts that do not name the same source revision. It then
records that revision as the claim artifact's `commit`, which the capability
matrix requires to equal the claim's `code_version`.

The result is a latency claim, not a scaling claim. The artifact states
`scalability_claim_allowed` and `release_gate_allowed` false, carries the union
of the three blockers that survived, and reports the layout it measured --
`distribution_semantics`, the world, local world and node counts, and one
`rank_shards` entry per rank naming the work that rank owned, the largest local
footprint it recorded, and the traffic it carried on the timed forward leg.
That is what makes it auditable non-release evidence: the checked-in result
audit inspects it instead of skipping it, classifies it
`benchmark_evidence_class = "local_non_release"`, and still refuses to promote
it, because it carries scalability blockers and
`claim_evidence_type = "development_smoke"` rather than a release payload.
Rebuild it with `python benchmarks/build_multinode_performance_claim.py`, and
fail the build when it has drifted from the recorded evidence with the same
command and `--check`. Rebuild it whenever an artifact it reads is re-recorded,
because the digests it copies move with them. The rebuild is byte-identical on
the CUDA environment as well as on a development host, so the committed artifact
is a property of the recorded evidence and not of the interpreter that wrote it.

## Evidence boundary

SSH success, 16 visible GPUs, matching package versions, successful pings, and
an NCCL smoke test establish readiness and transport health only. They do not
show that a FlagQuantum workload scales across nodes.

A multi-node scalability result must still satisfy the repository contracts:

- one logical workload is partitioned across ranks;
- `distribution_semantics="sharded_across_ranks"` is truthful;
- `world_size`, `local_world_size`, `node_count`, rank ownership, memory, and
  communication evidence are recorded;
- blockers and `scalability_claim_allowed` are reported fail-closed;
- the two-node lane in `.github/workflows/scheduled-hardware.yml` passes, and
  the benchmark/release audits pass.

The recorded artifacts above satisfy the first four. None is a scalability
claim; promoting one requires the audited benchmark payload the release gate
validates.
