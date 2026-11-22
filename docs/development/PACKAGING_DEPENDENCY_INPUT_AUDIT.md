# Packaging dependency input audit

## Complete source and native ownership

Q-0098 covers all five entries of the unchanged packaging dependency input from
Qualtran v0.7.0, revision `bcb17bb25360ed68ce3bc084645db3e929799389`.
The source blob is `b6ad665bf1ecec33e4af47c5097e25833011e0c5`.
Root `pyproject.toml`, `MANIFEST.in` and the existing protobuf generator retain
build, distribution and code-generation ownership. These optional external
acceptance tools add no production dependency or public entry point.

| Source input | Existing consumer and executed evidence |
| --- | --- |
| virtualenv | An actual isolated interpreter created with downloads and seeding disabled, followed by an isolated-mode prefix probe |
| grpcio-tools | Existing complete eight-schema protobuf command and sixteen generated artifact comparison |
| setuptools | Actual native source distribution built through the root build backend |
| wheel | Actual pure-Python fixture wheel packed and inspected for payload and RECORD |
| twine | Actual offline strict validation of both fixture wheel and native source distribution |

The existing user workflow remains `python -m build --sdist` from the repository
root with its declared build environment. Acceptance calls the same setuptools
backend directly with temporary output and egg-info directories. It checks that
the native README, manifest and project metadata survive distribution unchanged.

## Optional environment and license evidence

The isolated `packaging-command-deps` environment contains 32 distributions.
Thirty-one pins match the unchanged complete development lock. Setuptools 80.9.0
is an explicit bootstrap choice because the source unsafe-package lock omits its
pin. All active base requirement edges are checked against the installed versions.
The test also verifies the five original declarations in their original order.

The constraint derivative removes extras syntax from four constraint entries,
`bleach[css]`, `coverage[toml]`, `jsonschema[format-nongpl]` and
`openfermion[resources]`. Exact versions and environment markers are preserved.
The original reference files remain unchanged. This derivative exists only in
the external acceptance environment.

Thirty-one distributions include license or COPYING files. The grpcio-tools
1.76.0 wheel contains an Apache License 2.0 metadata declaration but no bundled
license file. The test records this exact packaging boundary explicitly.
Dependency acceptance does not certify redistribution notice completeness.
No dependency is vendored into the native distribution by this batch.

These tools remain replaceable at their existing subprocess and build-backend
boundaries. Their removal requires removing only the optional acceptance
environment and the corresponding development checks. Root packaging policy
continues to own release dependencies.

## Executed scope and remaining boundaries

`tests/unit/test_packaging_dependency_input.py` runs the actual interpreter,
wheel packer, native sdist backend and strict Twine checks. The concentrated
regression also runs existing manifest, setup, CI package policy, requirement-lock
compiler and protobuf consumers. Each subprocess has an explicit timeout.

The fixture wheel is pure Python. The native artifact is an sdist. Compiled
native wheels, target Torch ABI compatibility and publication remain separate
release acceptance work. Twine 6.2 skips Markdown rendering, so its success
establishes its actual metadata/archive checks. Existing documentation owners
retain rendered-content evidence. No upload is performed by these tests.

The full scientific development lock Q-0103 remains open independently of this
complete five-entry declaration. Original source defects, platform boundaries
and pending protected-contract proposals retain their existing ledger entries.

This input emits no quantum primitives and owns no state, measurement branches
or work registers. Quantum amplitude, workspace and independent model-versus-
primitive evidence remains with the algorithm owners.

Final local acceptance passes 93 concentrated cases with zero skips and 29
integration cases. Strict typing for the new test, Black, Ruff, Integration scope,
architecture, internal IR, protected API baseline and whitespace checks pass.
