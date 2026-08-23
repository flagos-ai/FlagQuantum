# Python and dependency policy

The executable support matrix is `dependency-policy.toml`. FlagQuantum core
supports Python 3.10–3.12 and installs only PyTorch. Every optional dependency
group in `pyproject.toml` must have an exact, classified entry in that matrix;
`python tools/check_dependency_policy.py` rejects missing groups, requirement
drift, unclassified extras and aggregate extras that no longer equal their
components.

JAX, Triton, visualization, examples and provider SDKs remain separate extras.
`interop-all` is the explicit aggregate for the Braket, Quafu and Qiskit
adapters; it is not part of the historical `all` development/runtime bundle.
Installing core FlagQuantum therefore never installs an external quantum
framework.

External framework imports are also namespace-governed. Qiskit imports belong
only under `flagquantum.interop.qiskit`; the architecture check rejects direct
Qiskit dependencies in core IR, compilers, runtimes, kernels, and distributed
workers. Existing experimental Aer entry points are compatibility wrappers over
that adapter.

Torch-FL is different from an interop SDK: it owns the FlagOS platform and
vendor-runtime boundary. It is deliberately recorded as
`managed_outside_flagquantum`, may not appear in core dependencies or a
FlagQuantum extra, and is imported only when the FlagOS platform is explicitly
activated. Native CPU and CUDA paths remain independent of Torch-FL.

The CI matrix tests the oldest and newest supported Python lines. Dependency
lower bounds are exercised by a dedicated compatibility lane and current local
acceptance exercises the upper supported environment. Ranges describe tested
support, not aspirational compatibility.

Dependency updates are reviewed monthly. Security updates are expedited.
Lower-bound changes require a clean-install test; upper-bound or major-version
changes require a dedicated compatibility pull request, runtime correctness and
package checks. Torch and JAX claims remain separate.

Source distributions contain runtime package sources, licenses, README and
build metadata only. Examples, tests, benchmarks, generated caches, datasets
and model snapshots are forbidden. Versioned example assets live outside Git
and are described by `examples/assets/manifest.json` with byte sizes and
SHA-256 checksums.
