# CI diagnostic annotations

The quality job registers these matcher files before running its existing
formatting and type checks. GitHub scans tool output and associates recognized
messages with repository paths. Matcher registration adds annotations while the
existing tool exit status continues to determine success or failure.

| File | Diagnostic consumer |
| --- | --- |
| black.json | Black check-mode would-reformat messages |
| mypy.json | Line-numbered mypy errors and warnings |
| pylint.json | Optional retained-source Pylint error/fatal and convention/refactor/warning messages |

Ordinary native lint uses Ruff's GitHub output format. The Pylint matcher supports
the existing optional retained-source lint command and adds no installation or
ordinary Pylint gate. The CI quality job keeps its existing execution conditions.
These matchers also work in another workflow after explicit registration.

The complete declarations are adapted unchanged from Qualtran v0.7.0 at
bcb17bb25360ed68ce3bc084645db3e929799389. Original mypy and Pylint patterns were
copied from Home Assistant 2025.1.2 under Apache-2.0. The original Black pattern
came from the Sumanth077 MLflow fork dated 2022-05-29 under Apache-2.0. The frozen
upstream README contains the detailed provenance URLs. Its relative workflows
link and pypy.json URL contain source path mistakes. Native ownership and actual
workflow paths are documented here, with complete source attribution retained
in docs/development/CI_PROBLEM_MATCHER_AUDIT.md.

Run tests/unit/test_ci_problem_matchers.py in the locked G2 environment to verify
all original blobs, actual tool diagnostics, JavaScript regular-expression
captures, native registration and unchanged failure exit statuses. Local checks
exercise captures and workflow declarations. Hosted GitHub annotation rendering
is checked when the workflow runs.
