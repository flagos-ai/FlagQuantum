# Version and release policy

FlagQuantum supports CPython 3.10 through 3.12.
`flagquantum.version.__version__` is the only package-version source; build
metadata, wheel names and runtime `flagquantum.__version__` derive from it.

FlagQuantum uses semantic versioning. While the project is below 1.0, minor
versions may contain explicitly documented API migrations; patch versions are
backward-compatible fixes. Stable public API removals require the documented
deprecation window and removal version. Runtime evidence and benchmark schemas
carry independent schema versions and migration rules.

Version `0.1.0` is the historical FlagOS 2.1 development snapshot. Version
`0.2.0` is the first release governed by the stable API manifest and the
capability-maturity matrix. Alpha, beta and release-candidate builds must use a
PEP 440 prerelease suffix before a GitHub release is published. Accelerator or
multi-node claims additionally require scheduled hardware evidence and
benchmark audit; CPU contract tests alone are not scalability evidence.

Dependency groups are intentionally separated:

- core: bounded PyTorch and local FlagQuantum execution;
- `dev`: tests, formatting, typing and package verification;
- `jax`: optional bounded JAX quantum kernels;
- `cuda`: optional Triton kernels without changing native PyTorch CUDA support;
- `braket`, `cirq`, `pennylane`, `quafu`, `qiskit`: isolated provider and interop SDKs;
- `interop-all`: the exact aggregate of those five interop groups;
- `cudaq`: an isolated heterogeneous compiler toolchain, excluded from the
  portable `interop-all` aggregate;
- `azure`: the optional Microsoft QDK Azure Quantum client (`qdk[azure]`),
  excluded from `all` and `interop-all`; the core install and the core import
  path stay without it, and only a real remote run reaches an Azure workspace;
- `examples`: datasets and transformer examples;
- `viz`: Matplotlib circuit rendering;
- `all`: the historical development/runtime aggregate, excluding providers;
- Torch-FL/FlagOS: externally managed and never installed by FlagQuantum.

A local wheel/sdist build is verification, not publication. Publishing is
allowed only from an intentional release commit after supported Python and
dependency bounds, clean installs, artifact contents, licenses, reproducible
builds, checksummed manifests and release contracts pass. The current
development branch must not publish unfinished builds.

## A red `main` blocks the release

A release candidate is a revision of `main` whose required checks all completed
successfully. The names are recorded in `.github/required-checks.json` and every
one of them is enforced on `main`, including for the maintainer.

While `main` is red the release waits, and the remedy is a fix that passes on
`main` rather than a release cut beside it. A revision whose required checks
failed is a revision with a known defect; a revision whose checks never reported
is a revision with no reading at all, and the second is the more dangerous of the
two because nothing in the repository reports the difference. That is not
hypothetical: on 2026-10-08 two merges to `main` landed 17 seconds apart and all
23 check runs attached to the first of them were `cancelled`, `quality` among
them, so a revision of `main` existed that no gate had read
([issue #579](https://github.com/flagos-ai/FlagQuantum/issues/579)). The
workflows no longer cancel a superseded run of `main` for that reason, and this
paragraph is what makes the reading a release condition rather than a courtesy.
