# R300 CPU CI runners

Initial qualification used Qiskit on `flagcicd-r300-cpu05`, Braket on
`flagcicd-r300-cpu03`, and Cirq on `flagcicd-r300-cpu04`. All three completed
resource inventory, native CPU preflight and both certified SDK versions.
Current workflow routes come from the qualified registry described below.
Same-repository pull requests and manual CI runs use these targets. Fork pull
requests retain `ubuntu-latest` and its existing hosted concurrency buckets.
The two certified SDK versions in each compatibility job remain unchanged.

## Validation and limits

The [resource inventory](https://github.com/flagos-ai/FlagQuantum/actions/runs/39936953344)
targets all 17 scale sets listed in repository settings, including the exact
`flagcicd-r300-cpu115` spelling. Inventory uses shell tools and reads the current
process's cgroup ancestors without downloading source or third-party actions.
The workload probes separately use the existing `tools/ci_cpu_budget.py` to
measure effective CPU/memory limits and select bounded test workers.

Measured samples expose 128 logical CPUs, 540,647,444,480 bytes (503.5 GiB) of
host memory and more than 370 GiB of free workspace storage. Containers are in
Kubernetes BestEffort cgroups, with no individual CPU or memory limit. The
`kubepods.slice` ancestor limits memory to the host total. These figures describe
shared host visibility; they are not per-job reservations and must not be added
across scale-set names to claim aggregate capacity.

The Ubuntu 24.04 base image has Git but lacks GCC and G++. The
`prepare-r300-cpu` action installs `build-essential`, Python 3.12, its headers
and venv support from the Tsinghua Ubuntu mirror with bounded, non-interactive
sudo commands. A job-local DEB822 source uses the official Ubuntu archive keyring;
signature verification remains enabled. If mirror metadata cannot be fetched,
a warning records a bounded retry against the image's configured Ubuntu sources.
The three SDK jobs use this distribution Python instead of downloading a GitHub
Python tarball; hosted fork jobs retain `setup-python`. Compilation uses at most
two processes. The cross-version manual probes still use `setup-python`.
Python packages use the existing Tsinghua pip index, while CPU PyTorch uses the
existing Aliyun wheel mirror and official fallback.
Each job keeps a private virtual environment through `isolated-python`, CPU-only
PyTorch through `install-cpu-torch`, and a writable temporary pip cache.

Live native-build checks and 296 representative regressions passed on:

- [cpu01 / Python 3.12](https://github.com/flagos-ai/FlagQuantum/actions/runs/39934968491/job/120352160239):
  CPU PyTorch, native extension availability and all representative tests passed.
  The subsequent serial default tier was cancelled when replacing the older
  network-dependent inventory; it is not recorded as a complete tier pass.
- [cpu16 / Python 3.11](https://github.com/flagos-ai/FlagQuantum/actions/runs/39935812731/job/120355109700):
  the complete representative workload job succeeded.

The [PR CI run](https://github.com/flagos-ai/FlagQuantum/actions/runs/39941739834)
passed all required checks with Python 3.12 and signed packages downloaded from
the Tsinghua Ubuntu mirror. Each SDK job passed 182 native preflight tests, then:

- [Qiskit / cpu05](https://github.com/flagos-ai/FlagQuantum/actions/runs/39941739834/job/120374725353):
  both certified versions passed the interoperability contract, 45 adapter tests
  and 241 quantum parity tests per version.
- [Cirq / cpu16](https://github.com/flagos-ai/FlagQuantum/actions/runs/39941739834/job/120374725210):
  both certified versions passed the interoperability contract and 55 tests each.
- [Braket / cpu03](https://github.com/flagos-ai/FlagQuantum/actions/runs/39941739834/job/120374725246):
  both certified versions passed the interoperability contract and 54 tests each.

Direct GitHub checkout and some artifact downloads showed long delays on other
attempts. Source artifact transfer on R300 is therefore bounded to three minutes.
If transfer fails, an explicit warning records the alternate GitHub checkout;
that checkout is bounded to five minutes, uses the same `github.sha`, does not
persist credentials, and verifies HEAD before installing anything. Artifact
checksum failures remain errors for hosted fork jobs. Source transport never
changes the tested commit.

Cirq initially qualified on cpu16, but two attempts on the integrated revision
subsequently exhausted both the three-minute artifact and five-minute checkout
budgets before reaching tests. Cirq therefore moves to inventoried cpu04;
qualification requires the full native and certified SDK tests on that target.
The cpu16 network failure remains unqualified for current routing; increasing
timeouts or repeatedly rerunning it is not treated as a repair.

The three SDK jobs use distinct scale sets so one set's occupied slots do not
serialize unrelated compatibility checks. The native CPU tests execute before
the SDK matrix on every migrated job. Their full SDK tests remain required CI
checks, and every job has a 30-minute timeout. GPU, A800 CPU/runtime, full-history quality and fixed-host
performance jobs retain their existing routes.

## Repeat the checks

After this workflow is merged, run:

```bash
gh workflow run r300-validation.yml --repo flagos-ai/FlagQuantum --ref main
```

The full inventory and cross-version diagnostic are manually dispatched. Pool
policy changes additionally qualify only the explicitly selected candidate batch
on same-repository pull requests; ordinary PRs do not repeat the whole inventory.
For a full diagnostic, inspect all resource inventory jobs, the three Python
workload probes and the
`r300-workload-*` resource artifacts. The Python 3.12 probe also runs the default
PR tier with a measured, bounded number of pytest workers. A new target should
pass both inventory and the relevant native/SDK workload before receiving CI.

## Workflow-managed pool

The workflow-level pool does not require cluster administration or new ARC
labels. `tools/ci/r300_pool.json` owns the qualified member list, blocked targets,
consumer jobs and an explicit candidate batch. `prepare-source` calls the standard
library-only `tools/ci/r300_pool.py` once and passes the JSON plan to job
`runs-on` expressions. Fork pull requests keep their GitHub-hosted fallback.

A plan assigns each consumer to a distinct qualified target, rotating the
starting point by workflow run ID. This spreads work across runs; it is not
live idle-capacity detection, a global reservation, or automatic failover of an
already-running job. Busy selected sets can still queue. No GitHub API polling,
cluster credentials, RBAC changes or runner registration changes are needed.

Inspect the exact planning behavior locally:

```bash
python3 tools/ci/r300_pool.py --seed 123
```

Initially cpu03/cpu04/cpu05 are qualified using successful SDK CI evidence.
The first additional candidate batch is cpu01/cpu02/cpu06/cpu07; remaining
inventoried targets stay candidates, and cpu08/cpu16 remain quarantined.
Candidates and quarantined members cannot receive production jobs. Fewer
qualified members than consumer jobs makes planning fail before fan-out.

Run the candidate batch manually when needed:

```bash
gh workflow run r300-validation.yml --repo flagos-ai/FlagQuantum \
  --ref ci/r300-unified-pool -f qualify_pool=true
```

Pool-definition PRs run this batch automatically only when the PR originates
in the same repository. At most four qualification jobs run concurrently.
Each target must transfer and verify the exact source commit, install the
signed Ubuntu Python/compiler prerequisites, install CPU-only PyTorch, compile
and assert the native extension, run representative native/backend/smoke tests
and routing policies, and retain measured resource/timing reports. The CPU-wheel
installation is bounded to eight minutes, source paths to three/five minutes,
and the entire qualification job to 25 minutes.

After inspecting actual successful job logs, promote only passing targets by
recording their qualification job URL and changing state to `qualified`.
Failed targets retain candidate state or become `quarantined` with their reason.
Promotion does not silently begin a different batch: the batch is explicit, and
completed members are not requalified. Further expansion requires selecting
another bounded batch in the registry.

The first consumers are the three existing SDK jobs. After additional members
qualify and pooled SDK jobs pass, migrate a small portable Python 3.12 CPU job,
then validate a limited coverage subset before any broad coverage migration.
Heavy distributed suites and fixed-host performance checks retain dedicated
measured services. There is no claim that unqualified candidates or additional
CPU jobs have already migrated.

## Cluster-level shared selector (optional operator path)

A cluster-level alternative is `[linux, x64, flagquantum-r300-cpu]`.
That shared ARC selector is not deployed. The workflow-managed pool above uses
existing scale-set names and can proceed independently of operator access.

The runner-side audit found Kubernetes `v1.31.14` in namespace `arc-runners`.
The cpu04 runner uses service account
`flagcicd-r300-cpu04-gha-rs-kube-mode`; its existing permissions allow reading
its own Pod, but cluster-wide deployment and autoscaling-runner-set listing
returned HTTP 403. Its runner and Docker sidecar have empty resource settings.
The observed images are `harbor.baai.ac.cn/baai-infra/actions-runner:latest` and
`harbor.baai.ac.cn/baai-infra/docker:dind-ubuntu-v2.0`. Kubernetes version and
runner binary version do not establish the ARC controller/chart version.

No local kubeconfig, R300 management SSH alias, or authoritative ARC deployment
repository is available in this workspace. The runner's permissions must not be
expanded to obtain cluster administration. An existing cluster operator must
provide a supported management context or apply reviewed deployment changes.

The initial read-only audit evidence is retained in
[the completed cpu04 probe](https://github.com/flagos-ai/FlagQuantum/actions/runs/39958916351/job/120431654590).
The temporary cluster-audit implementation is not part of the workflow-managed
pool: production planning and candidate qualification never read service-account
credentials or contact Kubernetes APIs.

A cluster operator can collect the missing non-secret deployment evidence with:

```bash
helm list --all-namespaces --output json
kubectl get deployments --all-namespaces \
  -l app.kubernetes.io/name=gha-runner-scale-set-controller \
  -o 'custom-columns=NAMESPACE:.metadata.namespace,NAME:.metadata.name,IMAGE:.spec.template.spec.containers[*].image'
kubectl get autoscalingrunnersets.actions.github.com --all-namespaces \
  -o 'custom-columns=NAMESPACE:.metadata.namespace,NAME:.metadata.name,CHART:.metadata.labels.helm\.sh/chart,LABELS:.spec.runnerScaleSetLabels,MIN:.spec.minRunners,MAX:.spec.maxRunners'
```

Admission sequence:

1. Verify controller and chart versions. ARC 0.14.0 introduced multiple scale-set
   labels ([release notes](https://github.blog/changelog/2026-03-19-actions-runner-controller-release-0-14-0/)).
   Do not assume the installed version supports them or upgrade a shared
   controller without inspecting its other consumers.
2. Use the authoritative Helm/GitOps values to add `linux`, `x64`, and
   `flagquantum-r300-cpu` to qualified R300 sets. Preserve each individual selector,
   GitHub scope, runner-group restrictions, node placement and existing capacity
   limits. Do not add the generic A800/MC550 `flagquantum-cpu` selector.
3. Pin the runner image, provide Python/compiler prerequisites and the existing
   CPU-wheel mirror/cache policy, and set measured CPU/memory requests and limits
   for both runner and sidecar. Derive pool concurrency from node allocatable
   resources and per-Pod budgets; shared host totals are not per-job capacity.
4. Run concurrent smoke/native and SDK probes using only the shared selector.
   Record actual selected scale sets, Pod limits, download source, queue delay,
   test results and cleanup. Accept only healthy, qualified members; exclude
   cpu16 while its repeated source-transfer failure remains unresolved.
5. Move the three SDK jobs to the shared selector first. Then qualify additional
   ordinary CPU jobs and a small coverage subset before broader migration.
   Performance and distributed jobs retain their dedicated measured services.

Rollback removes the shared selector from the affected deployment values and
restores individually qualified job targets. Until operator access and the
admission workflow are verified, no shared pool or new CPU migration is claimed.
