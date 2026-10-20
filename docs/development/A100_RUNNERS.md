# Shared A100 CPU CI

## Verified runner labels

On 2026-10-20, the refreshed repository runner settings listed
`a100-jiuding-0` and `a100-jiuding-1` as Idle in the organization Default group.
Both expose `self-hosted`, `Linux`, `X64`, `a100`, `jiuding`, and `gpu`.
CI targets that complete label set; either runner can accept a job.

The earlier local migration targeted `flagquantum-a100-cpu` and described
sixteen dedicated CPU registrations. That label and those registrations were
absent during the initial settings inspection. Later API checks found the
dedicated registrations partially online. They remain separate from the shared
Jiuding runners and are not prerequisites of this change.
The existing local GPU workflow retains its original hardware gates.

## First migration batch

`quality`, `distributed-cpu`, and `pre-commit` use the shared Jiuding runners
on main pushes and same-repository pull requests. The core Python and runtime
jobs retain the dedicated A800 CPU fast path merged in #654. The Python 3.12
required-check aggregator runs on GitHub-hosted Ubuntu.

Fork pull requests use `ubuntu-latest`. Other jobs retain their existing runners.
Required check names, test commands, and hosted capacity buckets are unchanged.
No `pull_request_target` trigger executes fork code on a shared machine.

Each Python job creates a fresh venv without system site packages or inherited
Python search paths. Native library paths inside inherited Python package
directories are also removed so a host Torch library cannot override the CPU
wheel. Test jobs install CPU PyTorch and the declared setuptools/wheel build
tooling before installing the project. CUDA devices are hidden and
numerical library threads are limited to one on shared hosts. These settings do not impose host
resource isolation or prevent other repositories from competing for the same
machines. The two registrations do not prove two independent hosts or a CPU
quota, and jobs can queue when both runners are busy.

The original Jiuding full-unit run spent about sixteen minutes preparing the
interpreter and package before testing. Core and runtime jobs now retain the
A800 thirty-minute budget; distributed CPU has sixty minutes and shorter shared
jobs forty-five. Queue time does not consume the execution budget.

Shared jobs disable setup-python's remote pip-cache upload: the persistent host
already retains its downloads, and its global pip cache can include several
GB of unrelated packages. Fork jobs retain the GitHub-hosted pip cache.

## Package mirrors

Self-hosted jobs use the TUNA PyPI mirror. CPU PyTorch wheels are selected
separately from the Aliyun CPU wheel directory with `--no-index --no-deps`.
The mirror download has a 180-second budget and uses a job-owned temporary
directory, so a timeout leaves no partially installed Torch package;
ordinary dependencies are then resolved through the job's PyPI mirror. This
keeps CUDA wheels out of candidate selection. If the mirror attempt fails or times out,
the action retries the official CPU index and reports the fallback in the log.
An import check requires a CPU-only Torch build after installation.

GitHub-hosted fork jobs retain the official PyPI and CPU PyTorch indexes.
The mirror settings are scoped to the job; no host pip configuration is changed.

The SJTU CPU index had matching official wheel hashes during local inspection,
but the shared host's outbound proxy rejected its CONNECT requests with HTTP
500 during CI. The production route therefore uses the Aliyun wheel directory
with the official CPU fallback instead.

Mirror references:

- https://developer.aliyun.com/mirror/pytorch-wheels
- https://mirrors.tuna.tsinghua.edu.cn/help/pypi/

## Validation

Local configuration checks:

```bash
python -m pytest tests/unit/test_cpu_runner_routing.py \
  tests/unit/test_github_hosted_capacity_policy.py \
  tests/unit/test_shared_runner_checkout_policy.py -q
python tools/validate_required_checks.py
```

The isolation test executes the composite action twice, verifies distinct venvs,
and checks that an inherited host-only module cannot be imported.

After publishing this branch, open a same-repository pull request and inspect the
runner assigned to a first-batch job: it must be one of the two Jiuding runners.
The same job on a fork PR must use GitHub-hosted Ubuntu. Core jobs must pass
their existing assertions that NumPy, JAX, and external quantum frameworks are
absent. Configuration checks alone do not prove setup-python downloads, compiler
availability, network access, or host policy. Only a completed Actions run verifies
those conditions.

## CPU budgets and timing evidence

`tools/ci_cpu_budget.py` records affinity, every visible ancestor cgroup v1/v2
CPU quota, the memory limit, hostname, and load average before package installs.
The effective CPU count is the smaller of affinity and quota. Trusted shared
jobs allow at most eight pytest workers, half of the effective CPU budget, and
one worker per 4 GiB of total host/cgroup memory. This reserves half the resources
for the other runner registration; it is a conservative scheduling budget, not
an enforced resource partition. Both registrations can share one host. Each
worker uses one numerical-library thread. Hosted fork and merge-group jobs
retain at most two workers. No marker or test selection is removed.

The preceding two-worker run at `bc6115c28` selected 13,901 tests and took
1,871.11 seconds in pytest. Its slowest calls were the legacy API source scan
(110.76 s), a distance-five ZXXZ code enumeration (107.99 s), the published
bicycle code instance (84.27 s), and the QEC alignment mutation check (77.20 s).
This baseline includes failures; it is timing evidence, not a passing gate.
Source: https://github.com/flagos-ai/FlagQuantum/actions/runs/39421153835/job/118622882940

Every CI first-batch Python job uploads its measured resources and, where pytest
runs with xdist, JUnit timing XML in a seven-day `cpu-report-*` artifact even
when tests fail. Compare pytest execution separately from package setup and
runner queue time. The required-check aggregator and fork routing stay intact.

The dedicated A800 services pass `--dedicated` to the same resource probe. Their
worker budget uses the service's own measured cgroup quota rather than halving
it again: at most sixteen workers and one per 2 GiB. The declared 24 CPU / 64 GiB
service therefore retains sixteen workers. A smaller actual quota reduces that
number. Fork and merge-group CPU jobs use Ubuntu with at most two workers and
o shared-host package index. A800 source artifacts and verified local wheel
caches are retained unchanged.
