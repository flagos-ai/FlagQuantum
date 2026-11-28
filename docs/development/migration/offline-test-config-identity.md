# Keep checkout test configuration outside locked dependency identity

Owner: Integration. Commit this declaration before implementation.

At PR #1360 head21c339dab, Main46154774875 CPU preparation fails with
"The checkout manifest differs from the locked CPU image manifest." The root
migration correctly added the original pytest/Coverage values to pyproject.toml,
but the offline comparator still compares those newly present tool tables with
an older image that obtained the identical values from pytest.ini/.coveragerc.
The image profiles, build requirements, project dependencies and numerical
configuration did not change. This missing consumer causes preparation failure,
not a scientific-test failure. Retain the original CI and local reproduction.

Change only .github/actions/offline-cpu/check_manifest.py and its focused original
unit module, plus this declaration. Extend the existing non-install tool-metadata
normalization to the tool.pytest and tool.coverage tables. Their configuration
continues to be read from the checkout by pytest/Coverage and checked by the
existing selection, marker and measurement policy tests. Do not rewrite, remove
or default any of the six pytest values or original Coverage values.

Keep complete project and build-system comparison, existing Ruff/release-wheel
normalization, unknown tool tables and every other manifest field fail-closed.
Do not alter requirements, extras, Python constraints, backend settings, lock
profiles/digests, caches, runner admission, source transport, coverage floors,
scientific cases/inputs/budgets/precision or public contracts. No image rebuild,
network installation, fallback or broad ignore-all-tools behavior is authorized.

Regression tests must invoke the actual checker CLI against an old-image fixture
and new checkout: accept pytest-only, Coverage-only and their combination; still
reject runtime/optional dependencies, build requirements/backend, Python-version
constraints and an unknown tool table while test configuration is present. Keep
all original cases, functions and markers. Independently reproduce the whole
pre-migration versus current repository pyproject comparison before and after the
repair, proving that only test-tool tables explain the old mismatch.

Run the complete original offline/routing/source/installation modules, the new
boundaries, source quality and static gates. The resulting exact head requires
its own full CI/source/node/scientific/resource/install/elapsed acceptance. Prior
passes remain bound to their original heads. No merge, force push, favourable
CI trial rerun, runner administration or Rust.
