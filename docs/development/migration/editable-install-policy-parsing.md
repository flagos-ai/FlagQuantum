# Recognize editable pip installs without treating shell file checks as installs

Owner: Integration. This declaration precedes the policy-test implementation.

Current PR #1360 head `230fb4aa4fc2c9b4000c7252e8d0091599cd46ef`, Main run
`46148655898`, unit shard 0 job `140990293107` failed the installation guard.
The guard selects every entire run block containing the substring ` -e `.
Bounded source promotion legitimately uses Bash file-existence tests, so the
guard classified that source block as an editable install in native-fermion.
The actual native editable pip commands still carry `--no-build-isolation`.
Retain the original failing test and run; do not rerun them to obtain a pass.

Change only `tests/unit/test_installation_contract.py` and this declaration.
Identify literal pip-install invocations on logical shell lines, respecting
quoted words, comments and command separators. Select actual `-e`, `--editable`
or `--editable=...` arguments rather than unrelated Bash flags. Check build
isolation for every selected invocation, so a compliant install in one line or
command cannot hide a second noncompliant one in the same run block. Preserve
CPU Torch index presence and installation-before-project ordering.

Regression fixtures must exercise the real installation guard using workflow
commands: Bash existence checks without an install, compliant and noncompliant
editable forms, mixed valid/invalid commands, multiline continuations and Torch
ordering. No pip or native build is executed by those fixtures. Retain all other
existing installation/native assertions, signatures, markers and source inputs.
No workflow, product/native implementation, dependencies, public API, numerical
case, seed, width, tolerance, resource or test timeout changes are authorized by
this slice. Only newly declared regression nodes may be added; old nodes remain.

Run the complete installation-contract module, affected CI/source policy cases,
source quality and existing static gates. Independently qualify the next head
with all required CI and original scientific/resource/timing evidence. Previous
local passes and current failed/partial CI retain their exact input scope.
No merge, force push, runner administration, scientific reduction or Rust.
