# R300 CPU CI runners

Qiskit, Cirq and Braket compatibility use the accepted fixed CPU pool for
latency and source-transport reliability. Same-repository pull requests and
manual CI runs use this pool; fork pull requests retain `ubuntu-latest` and the
existing hosted concurrency buckets. The scale-set diagnostics remain available
for future qualification.

Qiskit previously qualified on `flagcicd-r300-cpu05`. Its two certified versions
now use parallel `qiskit-compat` jobs on the accepted fixed CPU pool, because a
R300 capacity wait exceeded ten minutes before certification started. The fixed
pool completed both versions and the stable gate in 4.2 minutes during validation.
Each lane retains the native CPU preflight and all SDK checks;
`qiskit-optional` requires both lanes to succeed. The R300 validation workflow
continues to inventory cpu05; this routing change does not remove its diagnostics.
Cirq's latest cpu04 attempt exhausted both the three-minute artifact budget and
five-minute checkout budget before any tests started. Its fixed-pool route keeps
native CPU preflight and both complete 1.6.1/1.7.0 certification selections.
Braket subsequently exhausted the same three-minute artifact and five-minute
checkout budgets on cpu03 before any tests started. It now uses the fixed pool
with both 1.117.0/1.127.1 SDK selections and native CPU preflight intact.
Fixed-host jobs reuse verified incremental Git objects.

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
The R300 SDK jobs use this distribution Python instead of downloading a GitHub
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
budgets before reaching tests. Cirq then moved to inventoried cpu04 for qualification. After the later
integrated cpu04 source-transport failure, the normal CI lane returned to the
accepted fixed pool with all native and SDK tests intact.
The cpu16 network failure remains unqualified for current routing; increasing
timeouts or repeatedly rerunning it is not treated as a repair.

All three SDK lanes now use the wide fixed CPU pool. R300 remains available
through manual diagnostics until its source transport and capacity are qualified
again. Native CPU tests execute before SDK certification on every lane. Their
full SDK tests remain required CI checks, and every job has a 30-minute timeout.
GPU, A800 CPU/runtime, full-history quality and fixed-host performance jobs retain
their existing routes.

## Repeat the checks

After this workflow is merged, run:

```bash
gh workflow run r300-validation.yml --repo flagos-ai/FlagQuantum --ref main
```

This diagnostic workflow is manually dispatched; ordinary pull requests use the
real SDK CI checks without repeating the entire pool inventory. Inspect all
resource inventory jobs, the three Python workload probes and the
`r300-workload-*` resource artifacts. The Python 3.12 probe also runs the default
PR tier with a measured, bounded number of pytest workers. A new target should
pass both inventory and the relevant native/SDK workload before receiving CI.
