# R300 CPU CI runners

The initial R300 routing moves the Qiskit and Braket compatibility jobs to
`flagcicd-r300-cpu01`, and Cirq compatibility to `flagcicd-r300-cpu16`.
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
`prepare-r300-cpu` action installs `build-essential` with bounded, non-interactive
sudo commands before Python setup. Compilation uses at most two processes.
Each job keeps a private virtual environment through `isolated-python`, CPU-only
PyTorch through `install-cpu-torch`, and a writable temporary pip cache.

Live native-build checks and 296 representative regressions passed on:

- [cpu01 / Python 3.12](https://github.com/flagos-ai/FlagQuantum/actions/runs/39934968491/job/120352160239):
  CPU PyTorch, native extension availability and all representative tests passed.
  The subsequent serial default tier was cancelled when replacing the older
  network-dependent inventory; it is not recorded as a complete tier pass.
- [cpu16 / Python 3.11](https://github.com/flagos-ai/FlagQuantum/actions/runs/39935812731/job/120355109700):
  the complete representative workload job succeeded.

Direct GitHub checkout and some artifact downloads showed long delays on other
attempts. The first migration therefore uses only cpu01 and cpu16. In particular,
cpu08 is not admitted by a successful resource inventory alone. The migrated
jobs retain the existing exact-commit artifact transfer, SHA verification and
source restoration; their full SDK tests remain required CI checks. Each job
has a 30-minute timeout. GPU, A800 CPU/runtime, full-history quality and fixed-host
performance jobs retain their existing routes.

## Repeat the checks

After this workflow is merged, run:

```bash
gh workflow run r300-validation.yml --repo flagos-ai/FlagQuantum --ref main
```

Inspect all resource inventory jobs, the three Python workload probes and the
`r300-workload-*` resource artifacts. The Python 3.12 probe also runs the default
PR tier with a measured, bounded number of pytest workers. A new target should
pass both inventory and the relevant native/SDK workload before receiving CI.
