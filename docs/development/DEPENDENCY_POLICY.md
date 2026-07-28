# Python and dependency policy

The executable support matrix is `dependency-policy.toml`. FlagQuantum core
supports Python 3.10–3.12 and installs only PyTorch. JAX, visualization,
datasets/transformers and provider integrations are separate extras; providers
currently use the Python standard library and add no dependency.

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
