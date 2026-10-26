# A100 runner pool

The dedicated A100 node serves this repository with sixteen CPU runners named
`fq-a100-cpu01` through `fq-a100-cpu16`, and one GPU runner named
`fq-a100-gpu08`. Fourteen ordinary CPU registrations have the `flagquantum-a100-cpu` label.
Slots `cpu15` and `cpu16` instead have `flagquantum-a100-cpu-entry`.
All sixteen retain the existing `a100-cpu` host and individual slot labels
for explicitly requested pool certification. The GPU runner
has `gpu`, `flagquantum-local`, `flagquantum-a100`, and `a100-8gpu` labels.

Portable optional-dependency, dependency-bound and supply-chain checks use `flagquantum-cpu-general`, registered on fourteen ordinary
A100 and sixteen A800 CPU runners after entry-pool deployment, alongside other accepted general-pool hosts. GitHub can assign the same PR's ready jobs to
both hosts' pools without an A100-only backlog while A800 slots are idle. Each
job restores the producer's digest-verified source artifact and checks its exact
commit, including when only failed jobs are rerun. This avoids A800's unreliable
HTTPS Git route without adding checkout credentials. Source restoration clears
sparse checkout state, dirty files and files left by preceding jobs.

Quality checks need full Git history for evidence provenance and retain the A100
pool's existing authenticated Git transport. Coverage aggregation also retains that pool; manual source hygiene uses the entry pool. A800 CPU compatibility, smoke/unit, runtime, coverage and distributed contract
jobs retain their dedicated label and verified source transport. The long suites
use the measured A800 service budget rather than a two-worker general-pool slot;
the eight A800 services per host each expose 12 CPU time units and 64 GiB,
admitting six coverage workers or one eight-rank distributed launcher. The
aggregate CPU ceiling remains 96 units per host. macOS,
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

## Linux prerequisites without hosted capacity

For same-repository pull requests and trusted push/manual runs, the CI source
producer, GPU source producer and source-hygiene checks use
`flagquantum-a100-cpu-entry`. Installed-wheel main smoke, quality and coverage
aggregation continue to use the fourteen ordinary `flagquantum-a100-cpu` slots. This pool already supplies private Unix
accounts, workspaces, temporary directories and tool caches, and the Git
transport needed for checkout. The general pool includes hosts whose HTTPS Git
route is unreliable, so source checkout is not routed to that broader pool.
Fork pull requests retain the existing GitHub-hosted route; no
`pull_request_target` execution or broader runner access is introduced.

Each migrated job clears inherited sparse-checkout state before checkout and
removes the checkout credential afterwards. Python jobs reuse the existing
private-environment action, without the shared setup-python pip cache. Source
producers refuse an existing staging directory, verify `HEAD == GITHUB_SHA`,
and retain full/incremental packs and run-attempt artifact identities. Consumers
still verify digests and exact revisions. Job timeouts, source-hygiene hooks,
installed-wheel/native-extension assertions and required single/two-GPU checks
remain enforced. Main smoke retains its Torch 2.10 floor and runs outside the
source checkout. Trusted concurrency groups include the workflow run identity;
fork fallback retains its hosted capacity buckets.

This removes the hosted source-preparation dependency for trusted CPU/GPU
fan-out. Full CI still includes hosted CUDA-Q preparation, Ubuntu 22.04 GCC 11
package certification, Linux wheel containers, macOS and Windows. Their
platform/dependency contracts are retained, so hosted congestion can still delay
complete CI. Existing queued runs keep the workflow revision with which they
started; changing routing does not migrate or restart those runs.

A current public CPU workflow can be checked after installation by saving this
as `/tmp/flagquantum_cpu_example.py` and running
`python /tmp/flagquantum_cpu_example.py`. Requirements are supported Python and
core FlagQuantum with CPU PyTorch; no optional SDK, GPU or credentials. From the
checkout use `PYTHONPATH=. python /tmp/flagquantum_cpu_example.py`.

```python
import flagquantum as fq

result = fq.run(
    fq.Circuit(2).h(0).cx(0, 1),
    outputs=fq.samples([0, 1]),
    shots=32,
)
assert result.samples is not None and result.samples.shape == (1, 32, 2)
assert bool((result.samples[..., 0] == result.samples[..., 1]).all())
print("Bell sample invariant: True")
```

Expected stdout:

```text
Bell sample invariant: True
```

This user example checks the unchanged CPU workflow. The routing and isolation
claims additionally require exact-head CI, real source-artifact round trips,
installed-wheel smoke and required GPU validation; the example alone does not
certify runner isolation or performance.

## Two reserved entry registrations

The reservation partitions existing registrations; it creates no runner, Unix
account, host, CPU affinity, memory quota, GPU access, or additional capacity.
Keep all existing isolation and `MAX_JOBS=2`. Only the two source producers and
CI/manual source hygiene target the entry label in routine workflows. Ordinary
builds, long tests, main wheel smoke, and general-pool jobs cannot match it.
Forks retain hosted fallback; native macOS/Windows, CUDA-Q and Linux package
certification retain their existing hosted contracts. This is coexistence, not
replacement of the hosted platform gates, and no measured speedup is claimed.

The existing explicit `validate_pool=true` maintenance certification remains an
exception to routine routing: all sixteen slots still run the same account,
affinity, private-directory and native/backend regressions on their neutral
`a100-cpu` host label and individual slot labels. No validation slot or assertion
is removed to reserve capacity. This maintenance run deliberately occupies the
entry slots while certifying them; do not dispatch it as routine PR work.

After reviewing this PR, inspect the repository-scoped label plan with the
existing authenticated `gh` CLI and Python 3.10 or later:

```bash
python tools/configure_a100_entry_pool.py
python tools/configure_a100_entry_pool.py --apply
```

Both reserved registrations must be online and idle. The default command only
prints the plan. Applying replaces only their custom label sets: preserve all
identity/slot labels, add `flagquantum-a100-cpu-entry`, remove
`flagquantum-a100-cpu` and `flagquantum-cpu-general`. The other fourteen
registrations remain unchanged. Each API call is bounded and each mutation
rechecks the full inventory. GitHub does not provide an atomic scheduling lock
for label updates: an old queued job may be assigned between inspection and the
label update. Labels do not interrupt that job; let it finish and verify the
assigned jobs before certifying the entry pool. If a registration is observed
busy or a request fails,
stop and inspect the inventory; a partially applied split is reported as a
failure, not certified as a deployed pool. Reapplying when both are idle is
idempotent. Do not restart services, alter quotas, or interrupt existing jobs.

Publish the concrete PR before changing labels. Then apply the reviewed label
plan once the slots are idle so the PR's automatically triggered source/hygiene
jobs can run. Do not start another CI run to make them pick up labels. Admission
requires those actual jobs on the reserved registrations, all ordinary/platform
checks, required single/two-GPU checks, review, and actual-main installation and
public example acceptance. For rollback, first restore the workflow baseline;
when both slots are idle restore their two ordinary labels and remove only the
entry label. Keep the neutral validation/slot labels throughout.

The executable Bell example and exact expected output above remain the public
user acceptance example; label inspection alone does not certify functionality.

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

The CUDA-Q certification job runs on GitHub-hosted Ubuntu because cold SDK
downloads timed out on the self-hosted CPU pool. It uses official PyPI and has
a twenty-minute job limit, including a ten-minute preparation limit. The
reusable preparation action still honors its configured index: an alternate
mirror gets at most 180 seconds before the 300-second official PyPI attempt.
An already selected official index is attempted only once. SDK pins, CUDA
constraints, both version checks and all tests remain mandatory. Only the tiny upstream `cudaq` meta package may build from
source. On a miss, binary SDK wheels and their full dependency closure are
resolved first. A private temporary installation of the real NVIDIA CUDA 13
NVRTC library is version-checked and supplied to the unmodified upstream
metadata hook through its library search path. Thus host CUDA 12 autodetection
cannot override the chosen variant. The meta-wheel-only build disables pip's
wheel reuse and dependency resolution; a subsequent full resolver (including
logical extras) and the final offline resolution still require every dependency.
No shared public wheel is rewritten. Other packages must provide wheels. A complete restored cache takes the offline path and
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
