# MC550 CPU runner pool

The FlagQuantum repository uses sixteen CPU runners on each of `mc550-111`,
`mc550-112`, `mc550-121`, and `mc550-122`. Runner names follow
`fq-mc550-111-cpu01`; labels identify the pool, host, and exact slot.
The `flagquantum-cpu-general` label admits a slot to existing trusted portable
CPU jobs after acceptance. Fork pull requests retain the hosted-runner routes.

## Storage and service ownership

`liuwei01.mount` bind-mounts the local `/data/liuwei01` directory at `/liuwei01`.
Applications, workspaces, caches and homes reside under
`/liuwei01/flagquantum-runners`. These files do not consume the root filesystem.
The mount is enabled at boot; each enabled `fqmc-cpuNN.service` requires it.
The independent services restart after listener failure and kill their entire
process group on stop. Existing business services and containers are not changed.

Every slot has a dedicated non-root `fqmcNN` account, private home/workspace,
read-only runner application and read-only shared Python runtimes/wheels.
The installer verifies official SHA256 metadata before extracting archives.
GitHub administrator tokens are never retained on the nodes. Only the runner's
own registration credentials are readable by its account.

## Resource boundary

Each slot receives four physical cores and their SMT siblings within one NUMA
node, with a **four logical-CPU time quota**, 12 GiB memory high watermark,
16 GiB hard memory limit, no swap and at most 1,024 tasks. These are resource
limits rather than a claim that eight logical CPUs equal eight physical cores.
Physical cores 0–7 and 40–47, including their SMT siblings, are never assigned
to CI. The pool slice caps all sixteen slots together at 64 logical-CPU time
units, 256 GiB memory and no swap. CPU weight 1, nice 10 and idle I/O priority
favor existing work during contention. Aggregate data-disk I/O is capped at
300 MB/s read and 150 MB/s write.

Existing unrestricted business containers can still execute on the CPUs assigned
to CI. Affinity and priority therefore reduce interference but cannot guarantee
zero impact or a latency SLA on this shared host. Do not change business affinity
or restart existing containers to make the CI allocation exclusive.

Systemd gives the services private mount/device/tmp namespaces, no capabilities,
no privilege escalation, a read-only host filesystem and writable access only
to their own home, work and diagnostics directories. Business data mounts, peer
homes/credentials and the host Docker socket are unavailable. Accelerator devices
are excluded by `PrivateDevices`. Jobs receive a fresh Python virtual environment
through the existing `isolated-python` action. Only trusted team code may execute;
these services are not an isolation boundary for hostile public submissions.

## Installation and acceptance

The operator creates the dedicated mount and `fqmc.slice`, then runs
`tools/ci/provision_mc550_runners.py downloads.json` as root. The input JSON
contains official runner/Python archive URLs and SHA256 values; it has no secrets.
After separately registering each runner using a short-lived repository token,
enable its service. Disable automatic runner updates because applications are
root-owned; verified updates are an operator responsibility.

Run the existing workflow against the deployment branch:

```bash
gh workflow run local-gpu.yml --repo flagos-ai/FlagQuantum \
  --ref ci/mc550-cpu-runner-pool -f validate_mc550_pool=true
```

The 4-by-16 matrix selects every exact runner. It reuses the workflow's verified
source artifact and checks the restored commit against `GITHUB_SHA`, avoiding
sixty-four simultaneous HTTPS Git fetches on the unreliable Git transport.
It verifies cgroup limits and
filesystem/device boundary from inside the actual service, installs the project
in a private environment and runs CPU circuit/backend tests. It verifies the
checked-out SHA and CPU-only PyTorch. This acceptance does not certify accelerator
execution, scalability or interference-free production performance.

Useful host-side checks:

```bash
systemctl is-enabled liuwei01.mount fqmc-cpu{01..16}.service
systemctl status fqmc-cpu01.service
systemctl show fqmc.slice -p CPUQuotaPerSecUSec -p MemoryMax -p MemorySwapMax
journalctl -u fqmc-cpu01.service -n 40 --no-pager
```

Pause only this pool with `systemctl stop fqmc-cpu{01..16}.service`. Do not reboot
a shared host for acceptance; validate enabled units and dependencies, then test
automatic listener recovery while its runner is idle.
