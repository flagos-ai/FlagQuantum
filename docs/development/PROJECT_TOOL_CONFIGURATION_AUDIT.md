# Complete project tool configuration

Q-0137 covers the whole `pyproject.toml` at Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`, copyright Google LLC,
Apache-2.0, Git blob `7b1548955775f38d5f36cae38cc0346dbf9b46d8`.
Its complete executable configuration comprises three tool tables. It contains
no build-system, project metadata or dependency declaration. Those source
responsibilities belong to setup.py and the requirement files.

| Source configuration | Native owner and disposition |
| --- | --- |
| Black line-length=100 | Root pyproject Black configuration owns the existing native width of 88. Actual source/native formatter check/apply regression retains this policy adaptation. |
| Black Python 3.9/3.10/3.11/3.12 targets | Native Black targets the supported native Python 3.10/3.11/3.12 range. Distribution metadata owns interpreter support. |
| Black skip-string-normalization and skip-magic-trailing-comma | Native Black uses its existing normalization and trailing-comma policy. The accepted real formatter roundtrip executes both profiles and compares the resulting program values. |
| Black generated-proto exclusion | Original generated files stay under their frozen source audit. Native serialization owns its existing descriptors and format checks. There is no native generated qualtran/protos namespace. |
| isort Black profile, order_by_type=False, line_length=100, remove_redundant_aliases=True and generated-proto skip_glob | Native Ruff import sorting and root Black policy retain their existing configuration. `tools/development/format_command.py` owns formatter orchestration and source path selection. The full formatter suite executes the original isort/Black chain and the native Ruff/Black chain. |
| Pytest two dependency-warning filters | The frozen profile suppresses the spectral and platformdirs warnings. Native pytest retains its existing warning visibility. Both profiles execute the same isolated fixture with those two warnings and an unrelated control warning. Only the source suppresses the named pair. |
| Pytest python_classes='*TestSuite' | The frozen profile collects ProbeTestSuite. Native pytest.ini collects TestProbe through its existing Test* convention. Native component documentation uses ComponentExample rather than source TestBloq class names. |
| Pytest addopts='--strict-markers' | Both actual profiles reject an undeclared marker during collection. |
| Pytest slow/notebook markers | Frozen markers are checked with the complete source table. Native pytest.ini owns its broader current lane vocabulary, including slow and ftqc_reference. Existing notebook execution owns source notebook acceptance. |

The complete source dictionary and hash are checked. The isolated pytest
fixture runs actual subprocesses with each byte-preserved profile, checking
successful class collection, warning visibility and strict-marker refusal.
Optional plugin autoload is disabled in the fixture processes so installed
unrelated plugins cannot change this configuration comparison.

Existing `test_development_format_commands.py` supplies the complete source
configuration's actual Black/isort consumer and real native formatting roundtrip.
`test_development_pytest_commands.py` covers existing native full/developer/changed
command routing. These accepted implementation owners are reused.

The integration retains FlagQuantum's current tool policy. Source formatting
and collection conventions are recorded explicitly. Public API, dependency
and architecture policy authority stays with its existing repository owners.
The source configuration defines no quantum primitives, measured branches,
workspace or resource formula. Those comparisons are inapplicable here.
Q-0134 retains the separate documentation-site configuration scope.

## Batch verification

All 84 concentrated source/native packaging, formatter and pytest-command
checks pass. All 29 local integration checks pass. Strict typing covers both
new Python files. Black, Ruff, Integration scope, architecture, multilevel IR,
protected API and whitespace checks pass. Logs use
ftqc-reference-env/g2-packaging-config-*-20261122.log. The earlier Core NumPy
import-policy failure retains its independent unchanged-input receipt.
