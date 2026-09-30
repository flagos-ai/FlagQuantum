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
  --staging /nfs/fq-multinode-run \
  --checkpoint-directory /nfs/fq-multinode-run-checkpoints \
  --report-directory hardware-run
```

`--probe` selects the workload and defaults to `statevector`; everything else
is shared.

It takes three shared paths, and the names matter because the lane empties two
of them:

| Flag | Role | How it is treated |
| --- | --- | --- |
| `--staging` | the tree both ranks execute | synchronized into with `--delete`; must be named `fq-multinode*` and sit outside the source tree |
| `--output-directory` | per-rank JSON and NCCL debug logs (defaults to `<staging>-out`) | emptied before the ranks start |
| `--checkpoint-directory` | the rank checkpoints a restarted rank resumes from | emptied before the ranks start; both ranks must see the same directory |

The output and checkpoint directories are emptied for one reason: a stale rank
record left by an earlier run would otherwise be published as the evidence of a
run that failed before writing its own, and a checkpoint left by an earlier run
would be resumed from instead of the one this run wrote. All three names carry
the `fq-multinode` prefix so a mistyped path cannot be wiped.

The command refuses to start when any preflight answer is no, stages the tree,
and supervises both ranks as one unit with `tools/run_multinode_watchdog.py`.
On exit the report directory holds:

```text
plan.json                 the exact commands and environment both ranks were given
preflight.json            the nine answers
watchdog.json             which rank exited with which code, and why the run ended
watchdog-logs/            one log per rank
rank-0.json               the artifact, assembled by the rank that validated it
nccl-rank-{0,1}.log       the transport both ranks negotiated
```

The launch fails when the watchdog wrote no verdict, so a run that could not
report is distinguishable from a run that reported a failure.

Replicated per-rank execution can be used as a transport smoke test, but it
must not be reported as capacity scaling.

## Recorded two-node evidence

Two artifacts have been recorded from this lane, both taken on the eight-A800
pair with two ranks on one device each.

`artifacts/cuda_multinode_statevector_a800_jp171_jp172_20260930.json` covers
the amplitude-sharded statevector workload:

- the sharded forward pass, validated against a single-device complex128
  reference;
- the exact adjoint gradient, compared with the same reference;
- an owner-sharded optimizer step;
- a checkpoint written by one run and resumed by a restarted one, with the
  resumed leg reproducing exactly the steps the uninterrupted run computed
  after the checkpoint.

Every numerical metric in it is at round-off, both ranks own half the
amplitudes, and the communication it reports is inter-node with none
intra-node. It carries six blockers, including `rdma_not_tested` and
`production_performance_not_measured`, and reports `scalability_claim_allowed`
and `release_gate_allowed` false.

`artifacts/cuda_multinode_mps_a800_jp171_jp172_20260930.json` covers the
site-sharded MPS workload. Six wires at two ranks gives three owned sites per
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
  itself.

Its numerical metrics are at round-off, the two ranks report distinct host and
device identities, and the reverse attributes 32 bytes of layer halo to the
inter-node tier and none to the intra-node tier. It carries six blockers,
including `inter_node_cut_width_not_swept` and `rdma_not_tested`, and reports
both claim flags false. Unlike the statevector artifact it belongs to a
capability that also claims single-host and multi-GPU support, so it is the
multi-node leg of a broader entry rather than an entry of its own.

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

The recorded artifact above satisfies the first four. It is not a scalability
claim; promoting one requires the audited benchmark payload the release gate
validates.
