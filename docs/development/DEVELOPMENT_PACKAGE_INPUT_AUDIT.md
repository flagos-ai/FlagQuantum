# Developer package build declaration audit

## Complete declaration and ownership

Q-0053 covers the complete `dev_tools/pyproject.toml` from Qualtran v0.7.0,
revision `bcb17bb25360ed68ce3bc084645db3e929799389`, original blob
`46f067567919a8176e943f1f9e9fa2d08e17d427`.
All build-system, project and setuptools keys are checked against the unchanged
source. The original project declares setuptools >=61.0, Python >=3.10 and one
explicit package. It declares no installation dependencies.

FlagQuantum's root `pyproject.toml` owns distribution identity and dependencies.
Reusable developer functions and module commands live in the existing checkout
namespace `tools.development`. This is the native installation-layout adaptation
already documented in that domain's README. An ordinary contributor uses
`from tools.development.process import output_of` from the source checkout.
Documentation helpers consume the optional documentation environment described
by their existing owner audits.

## Actual build and source boundary

`tests/unit/test_development_package_input.py` copies the unchanged declaration
and developer package into a temporary project and executes the actual
setuptools wheel backend. Every top-level Python module is byte-identical in
the resulting wheel. Distribution name, version, Python constraint and absence
of dependency metadata match the original declaration. An isolated interpreter
loads the shell helper directly from the wheel and executes a bounded command.

The explicit source package list omits the nested `make_reference_docs` package.
That directory exists in the original checkout but is absent from the wheel.
An isolated import-spec probe confirms the omission. This retained packaging
counterexample keeps Q-0053 open under the current source-defect acceptance
policy. The original declaration and package source are unchanged.

A separate isolated native subprocess receives the explicit optional dependency
paths, imports the existing reference renderer and file/process helpers, writes
and reads a report, and executes the same bounded command. Existing complete
reference template/rendering and development helper acceptance is reused for
those modules' behavior. The native checkout layout remains its documented
contract.

## Scope

The actual builder uses the previously audited optional setuptools 80.9.0 and
wheel 0.45.1 environment. All temporary copies, wheels and reports stay inside
the test-owned temporary directory. Every child process has an explicit bound.
The reference package's Apache notices remain in its unchanged module bytes.
No new production package or dependency is introduced.

This build declaration owns no quantum state, primitive gates, measurement
branches, work registers or resource formulas. Quantum execution evidence stays
with the algorithm owners. Full dependency locks, standalone installed native
development tools and release artifacts retain separate scope.

The concentrated regression passes 58 cases with 892 retained parser and package
walk warnings. All 29 integration cases pass. Strict typing for the new file,
Black, Ruff, Integration scope, architecture, internal IR, protected API baseline
and whitespace checks pass. These results retain the source wheel omission and
the existing API-renderer setter refusal as explicit counterexamples.
