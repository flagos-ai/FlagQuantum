# R300 CPU CI runners

The initial R300 routing spreads Qiskit compatibility onto
`flagcicd-r300-cpu05`, Braket onto `flagcicd-r300-cpu03`, and Cirq onto
`flagcicd-r300-cpu04`. All three targets have completed resource inventory.
The cpu04 replacement must pass native CPU preflight and both certified Cirq
versions before integration; earlier SDK qualification on cpu16 is recorded below.
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

This diagnostic workflow is manually dispatched; ordinary pull requests use the
real SDK CI checks without repeating the entire pool inventory. Inspect all
resource inventory jobs, the three Python workload probes and the
`r300-workload-*` resource artifacts. The Python 3.12 probe also runs the default
PR tier with a measured, bounded number of pytest workers. A new target should
pass both inventory and the relevant native/SDK workload before receiving CI.
