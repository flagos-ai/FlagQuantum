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
the writable Python environment for every job. Private account directories use
Unix permissions to protect runner credentials and workspaces from other runner
accounts. CPU jobs use a CPU-only PyTorch build and an empty
`CUDA_VISIBLE_DEVICES`; this is an execution policy for trusted jobs, not device
isolation against hostile code.

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

This is a shared-kernel environment. CPU affinity is not a cgroup quota and
memory limits are shared. The platform does not permit container mount isolation.
Landlock ABI 1 was excluded because it prevents cross-directory renames needed by
editable Python installations. Jobs can read ordinary world-readable host files,
access the network, and change their own environment. Treat same-repository
workflow changes as trusted executable code. Never route arbitrary fork code to
this pool or use `pull_request_target` to execute a fork checkout. A platform
instance recreation requires restoring the deployment and restarting its
watchdog; process supervision does not preserve a deleted container.

The node's platform DNS intermittently routes `github.com` to an unreachable
address. A runner-local Git build supports `http.curloptResolve`; the launcher
provides tested alternate GitHub addresses while retaining hostname and TLS
certificate verification. The operator should refresh these addresses when the
network route changes. System Git and system hostname resolution are not part
of this runner-specific workaround.

First checkouts are seeded from a root-owned repository object cache readable
only by the dedicated runner group. The Git remote remains GitHub, and checkout
still verifies the requested commit. Keep the cache objects while any workspace
uses them as Git alternates; do not prune or remove the cache underneath jobs.
