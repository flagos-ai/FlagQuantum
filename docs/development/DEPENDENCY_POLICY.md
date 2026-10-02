# Python and dependency policy

The executable support matrix is `dependency-policy.toml`. FlagQuantum core
supports Python 3.10–3.12 and installs only PyTorch. Every optional dependency
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

Torch-FL is different from an interop SDK: it owns the FlagOS platform and
vendor-runtime boundary. It is deliberately recorded as
`managed_outside_flagquantum`, may not appear in core dependencies or a
FlagQuantum extra, and is imported only when the FlagOS platform is explicitly
activated. Native CPU and CUDA paths remain independent of Torch-FL.

The CI matrix tests the oldest and newest supported Python lines. Dependency
lower bounds are exercised by a dedicated compatibility lane and current local
acceptance exercises the upper supported environment. Ranges describe tested
support, not aspirational compatibility.

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
