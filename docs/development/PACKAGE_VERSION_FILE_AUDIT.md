# Complete package version file

Q-0169 covers `qualtran/_version.py` at locked Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`, copyright Google LLC,
Apache-2.0, Git blob `a21f8f56cc9e68b4e36f36665fdeba877ced3630`.
After the license comments, its entire executable body is the assignment
`__version__ = "0.7.0"`. The frozen-source test verifies the actual file hash,
single assignment, target and imported value.

FlagQuantum's existing `flagquantum/version.py` owns its independent release
version. The native regression verifies public `fq.__version__`, get_version,
VERSION_INFO and the PEP 621/setuptools dynamic-version attribute agree.
The source library version identifies the frozen reference. The native
version identifies the FlagQuantum distribution. Release numbers follow
their respective release histories.

This file has no gate, measurement, workspace or resource-model behavior.
Primitive counts, estimated gate costs and numerical branch comparisons are
inapplicable. Its full responsibility is covered through the existing native
release owner, with no production implementation required by this audit.
The setup.py packaging parent Q-0942 remains a separate whole-file task.
The regression reads build metadata without invoking a native extension build.

Batch results are retained in `ftqc-reference-env/g2-root-version-*.log`.

## Batch verification

The concentrated run passes 2107 checks and retains one pre-existing failure
in test_core_import_boundary.py. Seven NumPy imports in Core value_math.py
violate its dependency-policy test. The tested file, policy and failure test
are byte-identical to base cbf07e5fcbabc00ee0f36e5055b1fff79f5c090c.
The dedicated three root/version checks and 29 local integration checks pass.
Strict typing of both added Python files, Black, Ruff, team scope, architecture,
IR and public API checks pass. Logs and the unchanged-input blob certificate
are retained under ftqc-reference-env/g2-root-version-. The broader regression
retains that failure in its original log.
