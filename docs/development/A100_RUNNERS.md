# A100 runner pool

The dedicated A100 node serves this repository with sixteen CPU runners named
`fq-a100-cpu01` through `fq-a100-cpu16`, and one GPU runner named
`fq-a100-gpu08`. The CPU pool has the `flagquantum-a100-cpu` label. The GPU runner
has `gpu`, `flagquantum-local`, `flagquantum-a100`, and `a100-8gpu` labels.

Ordinary CPU CI and pre-commit jobs use the CPU pool for pushes and same-repository
pull requests. Fork pull requests use GitHub-hosted runners. macOS, Windows,
Linux wheel-container builds, and container publishing retain their existing
runners. Existing required check names and hosted-fallback capacity buckets are
unchanged.

## Validate the deployed pool

Run this existing manually dispatched workflow against the revision to validate:

```bash
gh workflow run local-gpu.yml --repo flagos-ai/FlagQuantum \
  --ref main -f validate_pool=true
gh run list --repo flagos-ai/FlagQuantum --workflow local-gpu.yml --limit 5
```

Expected results are sixteen successful CPU validation jobs, each pinned to a
different runner, and one successful eight-GPU validation job. CPU validation
checks the non-root identity, eight logical CPUs in the affinity mask, denied
access to GPU devices and protected file contents, then runs the existing native circuit
and backend regressions. GPU validation exercises all eight devices and runs the
existing eight-rank NCCL sharded forward and gradient differentials against dense
references. These are correctness checks, not a scalability performance claim.

Without `validate_pool=true`, the workflow retains its existing one- and two-GPU
gates. GPU jobs share the `flagquantum-accelerator` concurrency group.

## Host operation

The deployment lives in `/opt/flagquantum-runners`. Each runner uses a separate
Unix account, private home, workspace, temporary directory, and tool cache.
Root-owned framework environments are shared read-only. A pre-job hook rebuilds
the writable Python environment for every job. Landlock denies filesystem access
outside the runner's allowlist; CPU runners cannot open NVIDIA devices. Reading
file contents from personal/project or platform credential mounts is denied.

CPU affinity sets are disjoint: each CPU runner gets four physical cores and
their SMT siblings. The GPU runner gets sixteen different physical cores and
their SMT siblings. The process pool therefore uses 160 logical CPUs within the
container's 180-CPU quota. Thread defaults limit nested numerical libraries.

The container has no writable cgroup delegation or usable systemd. Supervisor
restarts exited listeners and terminates their process groups on stop. A separate
watchdog restarts Supervisor. Root can inspect the pool with:

```bash
/opt/flagquantum-runners/envs/admin/bin/supervisorctl \
  -c /opt/flagquantum-runners/admin/supervisord.conf status
```

Logs rotate under `/opt/flagquantum-runners/logs`. Registration uses a temporary
repository runner token; the host does not need a long-lived GitHub PAT. Runner
application files are root-owned and automatic updates are disabled, so the
operator must install supported runner releases within GitHub's update window.

This is a shared-kernel environment. CPU affinity is not a cgroup quota, memory
limits are shared, and Landlock ABI 1 does not isolate network traffic, hide
directory metadata, or provide all newer filesystem protections. The listener
requires enumeration of its ancestors up to `/`, so directory names can remain
visible even though file contents are denied. Treat same-repository workflow changes as
trusted executable code. Never route arbitrary fork code to this pool or use
`pull_request_target` to execute a fork checkout. A platform instance recreation
requires restoring the deployment and restarting its watchdog; process supervision
does not preserve a deleted container.
