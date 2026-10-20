# Measured CPU CI runner placement

## Production routing

| Jobs | Trusted same-repository PRs and main pushes | Fork PRs and merge groups |
| --- | --- | --- |
| Python 3.10/3.11 core, full Python 3.12 unit, runtime integration | Existing dedicated `flagquantum-cpu` A800 services | GitHub-hosted Ubuntu |
| Coverage and dependency bounds | Existing dedicated `flagquantum-cpu` A800 services | GitHub-hosted Ubuntu |
| Quality, distributed CPU, required-check aggregator | GitHub-hosted Ubuntu | GitHub-hosted Ubuntu |
| Pre-commit source hygiene | Shared `a100-jiuding-0` or `a100-jiuding-1` | GitHub-hosted Ubuntu |

The shared label set is `self-hosted`, `Linux`, `X64`, `a100`, `jiuding`, `gpu`.
CUDA devices are hidden for these CPU tasks. No runner registrations, host
quotas, required-check names, test marker selections, or per-test deadlines are
changed. No `pull_request_target` trigger executes fork code on shared hardware.
The separately registered A100 CPU pool is not a prerequisite of this routing.
GPU gates retain the configuration merged from main.

## Actual resource limits

The resource probe reads affinity and every visible ancestor cgroup v1/v2 CPU
and memory limit. Affinity alone is not a CPU allocation.

| Observed environment | Visible affinity CPUs | Effective CPU quota | Memory limit |
| --- | --- | --- | --- |
| Each Jiuding registration, in distinct container hostnames | 256 | 31 | 125 GiB |
| Observed A800 CPU service | 128 | 24 | 64 GiB |

These are measured limits, not a claim that physical cores are exclusive.
`tools/ci_cpu_budget.py` refreshes the values in each job and records hostname,
load average, inherited rank topology, and the selected worker budget.
Dedicated A800 services use at most sixteen pytest workers, bounded by their
actual service CPU quota and 2 GiB per worker. Hosted fallbacks use at most two.
Shared diagnostic budgets reserve roughly half the visible resources and cap
ordinary workers at eight. Launcher diagnostics budget up to eight child ranks
per worker and cap at two launcher workers; hosted launcher jobs use one.
Numerical libraries use one thread on self-hosted jobs. These budgets are not
new cgroup resource partitions.

## Timing and placement evidence

The original Jiuding two-worker full-unit run selected 13,901 tests and took
1,871.11 seconds in pytest. It had failures and is a timing baseline, not a
passing gate. Its slowest calls included the legacy API source scan (110.76 s),
ZXXZ enumeration (107.99 s), bicycle-code instance (84.27 s), and QEC mutation
check (77.20 s).
Source: https://github.com/flagos-ai/FlagQuantum/actions/runs/39421153835/job/118622882940

Retaining the A800 fast path from #654, the rank-isolation revision selected
13,932 items and passed with 13,804 passed and 184 skipped in 226.73 seconds.
The unchanged skip count and additional selected tests do not represent reduced
coverage. This comparison spans hardware and source revisions; it is not an
isolated claim about worker count.
Source: https://github.com/flagos-ai/FlagQuantum/actions/runs/39447907872/job/118717495331

An eight-worker Jiuding distributed experiment completed its first phase in
857.97 seconds but failed 26 cases: nested launchers exceeded existing deadlines,
and inherited platform rank topology caused a false single-device assertion
and a rendezvous port collision. That configuration was rejected. A subsequent
two-launcher attempt passed previously timed-out cases but exceeded fourteen
minutes including setup before completing its first phase. The passing hosted
comparison completed the entire distributed job in 9m43s, including setup; its
test tier took 8m38s. Hosted quality completed in 6m31s, while shared quality also
waited over ten minutes for a runner. Heavy gates therefore retain the faster
hosted backend rather than holding a concurrency bucket in the shared queue.
Hosted comparison: https://github.com/flagos-ai/FlagQuantum/actions/runs/39429575465/job/118651666421

## Environment and coverage

Coverage and dependency-bound jobs retain main's A800 migration with the same
trusted-event allowlist and hosted fallback. Coverage uses eight workers on
A800 and two on hosted runners, while the full seven-version dependency matrix
retains its three-job parallel limit.

The A800 services retain main's verified source artifacts, isolated per-job
venvs, and local CPU-only PyTorch wheel cache. Shared source-hygiene and hosted
quality/distributed jobs use the reusable isolated-Python action, which excludes
inherited Python package and native-library paths. Persistent hosts do not
upload their global pip cache through setup-python.

Ordinary dependencies on self-hosted jobs use TUNA PyPI; hosted fallbacks use
official indexes. The reusable CPU Torch action selects only CPU wheels, bounds
its optional Aliyun mirror download to 180 seconds, falls back to the official
CPU index, and checks the imported CPU build. A800 jobs retain their existing
local-wheel installation. Mirror reachability and speed must be measured on the
runner: the SJTU proxy path failed there, and the Aliyun trial was slower than
the cached official source.

Jiuding inherited `RANK=0` and `WORLD_SIZE=1` from its container platform. The
ordinary distributed tier clears inherited rank/rendezvous variables before
pytest; explicit torchrun subprocesses define their own topology. The complete
`pr-distributed` tier still runs both original selections (`distributed_cpu`,
then `benchmark_contract or release_gate`). All original test deadlines remain
unchanged. The separate two-rank A800 launched lane remains, and the JAX-installed
distributed job additionally runs the JAX gradient contract under two real ranks.
This restores intended collection semantics and strengthens launched coverage.

CI Python jobs retain resource JSON and available core/runtime JUnit timing XML
in seven-day `cpu-report-*` artifacts, even on failure. Distributed logs identify
individual cases and retain both phase-specific slow-test summaries. Compare
runner queue time, dependency setup, and pytest execution separately.
