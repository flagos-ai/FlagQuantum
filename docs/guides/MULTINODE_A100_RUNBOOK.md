# Two-Node DGX A100 Runbook

This runbook records the site-local setup used to prepare a two-node,
16-GPU FlagQuantum test. It covers access, environment replication, source
synchronization, and network preflight. Passing these checks establishes that
the machines are ready for multi-node testing; it is not scalability evidence.

## Site inventory

| Role | Hostname | Management address | GPUs |
| --- | --- | --- | --- |
| node 0 / launch host | `gpu-node-0` | `192.0.2.1/23` on `enp226s0` | 8 x NVIDIA A100-SXM4-40GB |
| node 1 / remote host | `gpu-node-1` | `192.0.2.2/23` on `enp226s0` | 8 x NVIDIA A100-SXM4-40GB |

The hosts also expose paired data-plane interfaces:

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

## SSH access

Node 0 uses a dedicated Ed25519 key to reach node 1. Never commit the private
key or an `authorized_keys` file. A suitable local-only SSH configuration is:

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

## Project-local Conda layout

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

## Source synchronization

Node 0 is the single source of truth. Do not edit the same source files on node
1. Preview every synchronization first:

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

Verify management connectivity from node 0:

```bash
ping -c 3 192.0.2.2
nc -vz -w 3 192.0.2.2 22
ssh -o BatchMode=yes fq-node2 hostname
```

Verify every paired data-plane address from node 0:

```bash
for n in $(seq 1 8); do
  subnet=$((191 + n))
  ping -I "ens10${n}" -c 2 -W 1 "10.8.${subnet}.25"
done
ping -I ens201 -c 2 -W 1 192.0.2.21
```

Ping failure alone is inconclusive if ICMP is filtered. Test the intended
rendezvous port in both directions with `nc`. A common rendezvous configuration
for this site is:

```bash
export MASTER_ADDR=192.0.2.1
export MASTER_PORT=29500
```

The port must be free on node 0 and reachable from node 1. Do not assume SSH
reachability proves that arbitrary rendezvous or NCCL ports are allowed.

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
- the `distributed_multinode` tier and benchmark/release audits pass.

Replicated per-rank execution can be used as a transport smoke test, but it
must not be reported as capacity scaling.
