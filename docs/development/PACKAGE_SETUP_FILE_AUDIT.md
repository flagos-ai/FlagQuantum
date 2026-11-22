# Complete setup metadata entry

Q-0942 covers the complete `setup.py` at Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`, copyright Google LLC,
Apache-2.0, Git blob `4eb99f1178be2d037f10e2716c2d564a1e04b6ce`.
The actual script executes unchanged with setuptools.setup intercepted before
distribution construction. Its four direct input blobs are checked.

| Complete source responsibility | Native owner and evidence |
| --- | --- |
| version_number(path) | `flagquantum/version.py` and the existing PEP 621 dynamic attribute provide the release value. The source regex is exercised independently, including its actual failure behavior. Native version/public metadata consistency uses the same authoritative owner. |
| main version lookup | Actual source main reads `qualtran/_version.py`. Native setuptools reads `flagquantum.version.__version__` through `pyproject.toml`. Both real paths are executed. |
| README long description | Actual source main reads the frozen README bytes. Native PEP 517 metadata contains the complete native README through the existing readme configuration. |
| Runtime requirements | Actual source main retains every non-comment line after strip, including empty strings. Native dependencies and optional features remain in `pyproject.toml` under the existing dependency policy. The generated native METADATA retains the configured mandatory torch and conditional tomli requirements. |
| Nine setup keyword arguments | The captured source call checks name, version, author, author_email, description, long_description, install_requires, license and packages. Native metadata validates its own project identity, version, author, description, license, README and requirements. |
| find_packages() | All 73 frozen initializer blobs match the immutable inventory. Parent-initializer traversal predicts exactly the 69 packages found by setuptools. Native package discovery uses its existing FlagQuantum include and exclusion configuration. |
| Import-time main() | The actual source script invokes main once. Missing version, README and runtime files fail before setup. The native entry is exercised through the real prepare_metadata_for_build_wheel hook. |

## Source failures and ownership decisions

The original parser accepts `0.7.0`, `1.2.3`, `0.7.0.dev1`, `0.7.0.dev` and
`0.7.0d`. Alpha, rc and missing matches raise AttributeError from group before
the intended main ValueError branch. This source failure remains in the
reference tests. Native version ownership uses its existing literal and
setuptools dynamic attribute. The parser bug does not require a duplicate
native release parser.

The source eagerly installs analysis, drawing and external-framework
dependencies. FlagQuantum's existing mandatory/optional dependency ownership
is the integration disposition. Runtime, Compiler, Simulation, Ecosystem and
development tools retain their current responsibility boundaries. Existing
distribution-policy tests retain package-root quarantine, runtime dependency,
native-extension and release-matrix requirements.

The original file delegates distribution construction to setuptools. Its
complete local responsibility is metadata assembly and entry invocation.
The source call interception and native real metadata hook prove that boundary.
This batch compiles no native extension and builds no wheel or sdist. Platform
release artifacts remain governed by existing release and distribution gates.
The metadata output and egg-info are created in a temporary directory.

No quantum operation, measurement history, workspace or resource model is
defined by this file. Gate counts, complex-amplitude and measured-branch
comparisons are inapplicable. Normal installed workflows continue using
`import flagquantum as fq` and `circuit.append(op)`.

Source intake receipts are `ftqc-reference-env/g2-setup-source-intake-20261122.json`
and `g2-native-metadata-intake-20261122.json`. The latter retains all frozen
initializer identities, the complete package list and generated native metadata.
The version regression also retains Python 3.10's existing tomli fallback.
Q-0134 remains open for complete documentation-build configuration.

## Batch verification

All 39 concentrated packaging/export checks and 29 local integration checks
pass. Strict typing covers both touched Python files. Black, Ruff, Integration
scope, architecture, multilevel IR, protected API and whitespace checks pass.
Logs use `ftqc-reference-env/g2-setup-*-20261122.log`. The previous complete
root/version batch's Core NumPy import-policy failure retains its unchanged-input
receipt and original log. This packaging batch has no Core implementation edits.
