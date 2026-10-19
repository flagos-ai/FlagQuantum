# Python and dependency policy

The executable support matrix is `dependency-policy.toml`. FlagQuantum core
supports Python 3.10–3.12 and accepts PyTorch 2.10.x–2.14.x. The native extension is
built once against the PyTorch 2.10 Stable ABI and shipped as a CPython 3.10+
limited-API wheel. CI then loads that same binary with every declared PyTorch
minor instead of assuming source compatibility implies binary compatibility.
Ordinary upstream CPU/CUDA installations continue to recommend PyTorch
2.13.x–2.14.x. The lower floor preserves externally managed accelerator builds
such as `2.10.0+metax3.8.1.0`; pip's normal installed-package preference retains
these compatible versions without `--no-deps`. Do not use eager dependency
upgrades or overwrite a vendor runtime with an upstream CPU/CUDA build.
Compatibility is not a security certification: the platform must maintain its
vendor runtime and patches. The `invertible` extra retains its separate 2.13.x
requirement.

CI runs core smoke checks on each declared minor and loads the same 2.10-built
wheel with 2.10, 2.11 and 2.12; the existing package/release checks cover 2.13 and
2.14. On 2026-10-11, public main `03427de55ddb50e796971aa95557c82c987a23c3`
passed source single-card autograd plus two/eight-card statevector forward,
sharded Adam training and checkpoint recovery on MetaX C550 using
`cr.metax-tech.com/public-library/maca-pytorch:3.8.1.2-torch2.10-py312-ubuntu24.04-amd64`.
The rc3 manylinux wheel also passed a device Bell-circuit check with dependency
installation disabled for that pre-fix metadata. Native extension loading on
that vendor runtime, multi-node execution, performance and capacity expansion
were not certified by those checks.

Source builds require setuptools 77.0.3 or newer for SPDX license metadata.
Do not upgrade build tools in a managed vendor runtime if another package pins
them (the C550 image's xformers pins setuptools 69.5.1). Build a wheel in a
separate environment that can reuse the vendor torch, then install that wheel
normally in the original runtime. For an already downloaded source checkout:

```console
python -m venv --system-site-packages /tmp/fq-build
/tmp/fq-build/bin/python -m pip install 'setuptools>=77.0.3' wheel
/tmp/fq-build/bin/python -m pip wheel . --no-build-isolation --no-deps -w /tmp/fq-dist
python -m pip install /tmp/fq-dist/*.whl
python -m pip check
```

Here `--no-deps` applies only to wheel construction, never to the final runtime
installation. Use fresh build/output directories for each build. Published wheel
users need none of these source-build steps.

Every optional dependency
group in `pyproject.toml` must have an exact, classified entry in that matrix;
`python tools/check_dependency_policy.py` rejects missing groups, requirement
drift, unclassified extras and aggregate extras that no longer equal their
components.

JAX, Triton, cotengra, stim, pymatching, visualization, examples and provider
SDKs remain separate extras. `pymatching` is a cross-check and never the
authority: the minimum-weight matcher is implemented in-tree, and the extra
exists so the correctness claim has one witness that does not share the detector
error model both decoders read. It is imported lazily, so the core install and
the decoder neither need nor load it.
`interop-all` is the explicit aggregate for the Braket, Cirq, PennyLane, Quafu and Qiskit
adapters; it is not part of the historical `all` development/runtime bundle.
Installing core FlagQuantum therefore never installs an external quantum
framework.

CUDA-Q is a separate `cudaq` heterogeneous-toolchain extra. It is deliberately
outside `interop-all`: its platform-specific compiler and simulator distribution
is materially heavier than the portable circuit-model adapters. The first
public adapter provides kernel export only and permits no CUDA-Q import outside
`flagquantum.ecosystem.cudaq`. CUDA-Q is verified in isolated Linux lanes and
remains outside portable aggregate installations.

External framework imports are also namespace-governed. Cirq, CUDA-Q, Qiskit,
and PennyLane imports belong only under their matching `flagquantum.ecosystem`
namespaces. Braket imports are restricted to its planned Ecosystem adapter and
the existing Remote provider. The architecture check rejects these dependencies
in core IR, compilers, kernels, and distributed workers. Existing experimental
Aer entry points are compatibility wrappers over the Qiskit adapter. The common
registry stores only module paths and adapter metadata; listing or resolving an
adapter must not import its external framework. The Braket SDK requires Python
3.11 or newer, so its optional and aggregate requirements carry an environment
marker while core FlagQuantum retains Python 3.10 support.

A collected module must not name an optional root outside a guard. `cpu-core`
installs `.[dev]` and PyTorch only, then collects `tests` and runs `tools`, so a
module-level `import numpy` — or `import stim`, `import qiskit`, `import jax` —
is a collection error on that lane and a passing suite on every developer
machine that happens to have the extra installed. That asymmetry is the defect
two suites shipped once each. `tools/check_dependency_policy.py` now reads the
`tests` and `tools` sources and reports any reference to a
`core_forbidden_imports` root that is not inside a `try` that absorbs the import
error, a `pytest.importorskip` of that root, or a skip on a `find_spec` of that
root; a `find_spec` that only inspects the result is not a guard. A guard on a
root that *requires* the referenced root also counts — `pytest.importorskip
("stim")` makes numpy importable, because stim requires it — and those
implications are declared in `[import_policy.guard_implications]` with the
distribution's own requirements as the citation, rather than exempted per file.
`benchmarks` is out of scope deliberately: it holds standalone comparison
scripts whose module-level imports are entry points, not modules any marker
selects.

`numpy` is itself on `core_forbidden_imports`. It appears only in the `kaiwu`
remote-provider extra; `torch` declares no numpy requirement, and
`import flagquantum` loads it nowhere, so every `cpu-core` leg reports
`Failed to initialize NumPy` while the suite still passes. The `kaiwu` extra
contains only the open adapter's NumPy boundary and deliberately does not
redistribute the separately reviewed Kaiwu SDK. Recording NumPy as forbidden
in core makes the existing absence proofs — the `cpu-core` assertion,
`tests/unit/test_dependency_policy.py`, and the `check_import_time.py` probe —
cover it without a second scanner, while the coverage lane installs the extra
and executes the adapter tests.

A standard-library module added in a later supported Python needs the
`try`/`except ModuleNotFoundError` fallback, not a `sys.version_info` test.
`tomllib` arrived in 3.11, so the 3.10 leg has no such module; the version test
is a statement about the interpreter rather than a fallback, it cannot be
simulated on a supported interpreter the way a missing import can, and it does
not survive a stripped standard library. Every `tomllib` reference in
`flagquantum`, `tests`, `tools`, and `benchmarks` is held to the `try` form, and
the `cpu-core` job asserts that `tomllib` is present exactly where the
interpreter provides it and that `tomli` is installed where it does not.

Torch-FL is different from an interop SDK: it owns the FlagOS platform and
vendor-runtime boundary. It is deliberately recorded as
`managed_outside_flagquantum`, may not appear in core dependencies or a
FlagQuantum extra, and is imported only when the FlagOS platform is explicitly
activated. Native CPU and CUDA paths remain independent of Torch-FL.

The CI matrix tests the oldest and newest supported Python lines. Dependency
lower bounds are exercised by a dedicated compatibility lane and current local
acceptance exercises the upper supported environment. The native wheel is
compiled against the oldest Stable ABI target, then tested without rebuilding
on the oldest and newest supported PyTorch runtimes. Widening the range still
requires that cross-version binary check. Ranges describe tested support, not
aspirational compatibility.

The `dev` extra is bounded like the runtime extras, because it is the toolchain
this repository installs to check itself: an unbounded entry lets an upstream
release change that toolchain and break `main` with no commit and no diff.
`python tools/check_dependency_policy.py` rejects an unbounded development
requirement, so the bound is a checked property rather than a convention. The
static gates record their oldest and newest verified release in `[tested]`, and
the `dependency-bounds` lane installs exactly that floor and re-runs `ruff`,
`black`, and `mypy` on it. A declared floor that cannot pass its own check
therefore fails CI instead of being trusted: `ruff` before 0.15.0 flags the
`_fields_` of a `ctypes.Structure` as a mutable class attribute, `black` before
24.1.0 reformats 73 files, and mypy before 2.0.0 reports errors the current
release does not.

Dependency updates are reviewed monthly. Security updates are expedited.
Lower-bound changes require a clean-install test; upper-bound or major-version
changes require a dedicated compatibility pull request, runtime correctness and
package checks. Torch and JAX claims remain separate.

Source distributions contain runtime package sources, licenses, README and
build metadata only. Examples, tests, benchmarks, generated caches, datasets
and model snapshots are forbidden. Versioned example assets live outside Git
and are described by `examples/assets/manifest.json` with byte sizes and
SHA-256 checksums.
