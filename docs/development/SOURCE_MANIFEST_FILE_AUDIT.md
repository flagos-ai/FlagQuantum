# Complete source distribution manifest

Q-0020 covers `MANIFEST.in` at Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`, copyright Google LLC,
Apache-2.0, Git blob `8f97b4d84b80065568b51b21fb6472250fc49da2`.
Its entire content is the single include directive for
`dev_tools/requirements/deps/runtime.txt`. The reference test verifies the
actual blob and executes the directive through setuptools FileList. It includes
the runtime input while leaving unrelated candidate paths unselected.

The native source distribution retains dependency authority in its existing
`pyproject.toml`. `MANIFEST.in` owns source-tree inclusion/exclusion, including
native C++ sources. The real PEP 517 build_sdist entry executes existing setup.py
with temporary egg-info and output locations. The resulting archive is opened
and its manifest, pyproject, README, version and package initializer are compared
byte-for-byte with their current source owners. Native C++ sources are present.
Tests, examples and benchmarks retain their existing quarantine.

Archive PKG-INFO retains the native project name, release version and mandatory
torch requirement. This is the dependency-input replacement for the original
runtime.txt include. The existing metadata-hook and distribution-policy suites
continue to cover mandatory/optional dependency and package-root ownership.
Q-0102 retains its separate full scientific dependency declaration scope.

The standalone intake produced a real native sdist containing 1965 archive
entries. Its membership and selected build inputs are retained in
`ftqc-reference-env/g2-packaging-sdist-intake-20261122.json`. The build's
transient release tree is removed by setuptools. This source distribution
acceptance leaves compiled wheel and installed native-extension verification
under the existing release gates.

The manifest defines no gates, measurements, work registers or resource model.
Quantum branch and gate-count comparisons are inapplicable. Existing installed
user entry points retain `import flagquantum as fq` and `circuit.append(op)`.

## Batch verification

All 84 concentrated source/native packaging, formatter and pytest-command
checks pass. All 29 local integration checks pass. Strict typing covers both
new Python files. Black, Ruff, Integration scope, architecture, multilevel IR,
protected API and whitespace checks pass. Logs use
ftqc-reference-env/g2-packaging-config-*-20261122.log. The earlier Core NumPy
import-policy failure retains its independent unchanged-input receipt.
