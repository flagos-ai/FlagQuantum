# A100 runner pool

The dedicated A100 node serves this repository with sixteen CPU runners named
`fq-a100-cpu01` through `fq-a100-cpu16`, and one GPU runner named
`fq-a100-gpu08`. The CPU pool has the `flagquantum-a100-cpu` label. The GPU runner
has `gpu`, `flagquantum-local`, `flagquantum-a100`, and `a100-8gpu` labels.

Portable CPU coverage, optional-dependency, distributed CPU, dependency-bound
and supply-chain checks use `flagquantum-cpu-general`, registered on all sixteen
A100 and eight A800 CPU runners. GitHub can assign the same PR's ready jobs to
both hosts' pools without an A100-only backlog while A800 slots are idle. Each
job restores the producer's digest-verified source artifact and checks its exact
commit, including when only failed jobs are rerun. This avoids A800's unreliable
HTTPS Git route without adding checkout credentials. Source restoration clears
sparse checkout state, dirty files and files left by preceding jobs.

Quality checks need full Git history for evidence provenance and retain the A100
pool's existing authenticated Git transport. Manual pre-commit and coverage
aggregation also retain that pool. A800 CPU compatibility, smoke/unit and runtime
jobs retain their dedicated label, source transport and configuration. macOS,
Windows, Linux wheel-container builds and container publishing retain their
existing runners. Fork pull requests use GitHub-hosted fallback.

Every self-hosted CPU job uses a distinct concurrency group for each workflow
run, job and matrix index. Physical runner availability bounds parallelism; there
is no self-hosted matrix `max-parallel` or shared capacity lock. The historical
`a100-cpu` group prefix also covers portable jobs and imposes no pool limit.
Sixteen scoped concurrency buckets continue to bound GitHub-hosted Ubuntu
fallback jobs. Physical GPU device locks remain independent of CPU scheduling.

To deploy this routing, add the general label without replacing existing labels:

```bash
for id in $(gh api 'repos/flagos-ai/FlagQuantum/actions/runners?per_page=100' \
  --jq '.runners[] | select(any(.labels[]; .name == "flagquantum-cpu" or .name == "flagquantum-a100-cpu")) | .id'); do
  gh api --method POST "repos/flagos-ai/FlagQuantum/actions/runners/$id/labels" \
    -f 'labels[]=flagquantum-cpu-general'
done
```

A800 portable jobs reuse `/root/liuwei/fq-ci-bootstrap/cpu-wheels` before
trying the network CPU wheel mirrors. A cache miss preserves the existing
mirror and official-index fallback, and each dependency-bound leg retains its
requested Torch version. Other dependencies still resolve through the job PyPI
index.

Only CPU registrations receive this label. Keep A100 validation slot labels and
A800 dedicated labels: those jobs depend on their original host environments.

## Validate the deployed pool

Run this existing manually dispatched workflow against the revision to validate:

```bash
gh workflow run local-gpu.yml --repo flagos-ai/FlagQuantum \
  --ref main -f validate_pool=true
gh run list --repo flagos-ai/FlagQuantum --workflow local-gpu.yml --limit 5
```

Expected results are sixteen successful CPU validation jobs, each pinned to a
different runner, and one successful eight-GPU validation job. CPU validation
checks the non-root identity, eight logical CPUs in the affinity mask, a CPU-only
PyTorch environment, and denied access to other private runner directories, then
runs the existing native circuit and backend regressions. GPU validation exercises all eight devices and runs the
existing eight-rank NCCL sharded forward and gradient differentials against dense
references. These are correctness checks, not a scalability performance claim.

Without `validate_pool=true`, the workflow retains its existing one- and two-GPU
gates. A100 validation uses `flagquantum-accelerator-a100` and can run alongside
A800 GPU work. A800 local and scheduled jobs retain `flagquantum-accelerator`
because the multi-node lane occupies both A800 hosts. Scheduled single-host jobs
are pinned to A800-172 so they cannot use the A100 host under the A800 lock.

## Host operation

The deployment lives in `/opt/flagquantum-runners`. Each runner uses a separate
Unix account, private home, workspace, temporary directory, and tool cache.
Root-owned framework environments provide the package seeds. A pre-job hook
creates a private writable Python environment and copies its framework packages,
using reflinks where supported. Both package files and distribution metadata
belong to the job, so upgrades and dependency audits see one installed version. Private account directories use
Unix permissions to protect runner credentials and workspaces from other runner
accounts. CPU jobs use a CPU-only PyTorch build and an empty
`CUDA_VISIBLE_DEVICES`; this is an execution policy for trusted jobs, not device
isolation against hostile code.

CPU affinity sets are disjoint: each CPU runner gets four physical cores and
their SMT siblings. The GPU runner gets sixteen different physical cores and
their SMT siblings. The process pool therefore uses 160 logical CPUs within the
container's 180-CPU quota. CPU jobs default to two numerical-library threads
so two pytest workers fit the four physical cores assigned to one runner.

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

The node's HTTPS Git route is unreliable. Runner Git redirects this repository's HTTPS URL
and the public pre-commit hooks dependency URL to GitHub's official `ssh.github.com:443` endpoint.
Host keys come from GitHub's official metadata API and strict host verification
is enabled. A repository-only, read-only deploy key is stored under
`tools/git-ssh`, readable only by root and the dedicated runner group. Revoke it
in repository Settings → Deploy keys using the title
`FlagQuantum A100 runner pool (read only)` when retiring the pool. System Git,
system SSH configuration, and system hostname resolution remain unchanged.

First checkouts are seeded from a root-owned repository object cache readable
only by the dedicated runner group. The Git remote remains GitHub, and checkout
still verifies the requested commit. Keep the cache objects while any workspace
uses them as Git alternates; do not prune or remove the cache underneath jobs.

The pool uses the Aliyun PyPI mirror for ordinary dependencies and the SJTU
`pytorch-wheels/cpu` mirror for CPU-only PyTorch wheels. Hosted jobs retain the
official PyTorch CPU index. Official setup-python 3.10.22, 3.11.17 and 3.12.15
runtimes are cached read-only after verification against GitHub asset SHA256
values; large bootstrap downloads may use a public release accelerator with
that same mandatory digest check.

Verified public wheels are cached read-only under `tools/public-wheels`. CPU
PyTorch retains its explicit local seeding and configured index fallback. The
CUDA-Q job recognizes an available self-hosted public cache using
`RUNNER_ENVIRONMENT`, rather than relying on an optional `FQ_RUNNER_KIND` value.
Runner accounts do not modify that public cache.

### CUDA-Q complete wheel cache

The existing CUDA-Q lane still verifies **both 0.15.1 and 0.16.0.post1** with
its contract checker, all export/conformance tests and lazy-import check. It
now fixes `cuda-quantum-cu13` explicitly, matching the binary flavor observed
in the previous CPU-only installation. A constraint refuses the other binary
flavors rather than mixing a cached CUDA 12 SDK with an auto-selected CUDA 13
meta package. This does not claim CUDA-Q GPU execution support in FlagQuantum.

The `prepare-cudaq-wheels` composite action uses pip's resolver to collect the
complete transitive dependency closure for each version, including ordinary
Python packages and the newer logical package dependencies. It does not
maintain a partial prefix list or install with `--no-deps`. All wheels share a
private job directory and an `actions/cache` key scoped to OS, architecture,
Python 3.12, CUDA 13, both SDK versions and the preparation action's content.
A cache is usable only if a fresh offline resolver can satisfy the complete
requirements while ignoring installed packages. SDK installation is then
fully offline, followed by exact distribution-version/flavor checks and
`pip check`. Failure remains failure; cache availability never skips a test.

A miss retains the job's configured package index: TUNA for self-hosted jobs,
official PyPI for hosted jobs. The mirror wheel-building attempt is bounded at
900 seconds; a failed attempt tries official PyPI with the same SDK pins and
CUDA constraints. Only the tiny upstream `cudaq` meta package may build from
source; its wheel is rebuilt during a miss so a wheel previously inferred from
another host's CUDA cannot select a different binary flavor. Other packages
must provide wheels. A complete restored cache takes the offline path and
avoids any index selection. The cache saves only after a successful job;
bump its key epoch or preparation action when deliberately refreshing the
resolved dependency set. Never remove shared public cache files while jobs
are using them.

On Linux with the optional SDK, a user can run:

```bash
python -m examples.cudaq_bell_export
```

Expected output on each certified version:

```text
Bell probabilities: [0.5, 0.0, 0.0, 0.5]
```

The same example is executed by the CUDA-Q job after its existing tests. Local
cache regression tests use real pip and tiny synthetic wheel metadata to
exercise transitive closure and incompatible/missing-cache refusal without
requiring CUDA or downloading SDKs. Those tests are not substitutes for real
SDK verification on Linux.

GitHub cache visibility follows branch scope. A PR warms subsequent runs of
that PR; other PRs can restore a default-branch cache. After this workflow
change is merged, seed the main-branch cache once with the existing full manual
workflow (under the repository's execution authorization):

```bash
gh workflow run ci.yml --ref main
```

Wait for that complete validation to succeed before reporting a shared cache
as warm. This change does not add a main-push trigger or bypass full validation.
Public wheel provisioning retains its official-source SHA256 checks: CPU Torch
against its official index, and CUDA-Q wheels against official PyPI metadata.
