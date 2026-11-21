# Development requirement lock compiler audit

## Complete source and ownership

Q-0099 and Q-0107 cover the complete pip-tools input declaration and its
seven-package frozen environment. Q-0112 is the complete `dev_tools/requirements/re-pip-compile.sh` from Qualtran
v0.7.0 `bcb17bb25360ed68ce3bc084645db3e929799389`. The test checks the unmodified
file against its original inventory blob. The native orchestration owner is
`tools/development/requirement_locks.py`. Existing process and equal-content
file helpers own subprocess execution and output writing.

The command accepts an explicit owner-supplied directory containing `deps/`
requirement inputs and an explicit output directory. It compiles the full dev
environment first. Runtime, format, pylint, pytest, mypy, docs and pip-tools
then compile in original order with the dev lock as a constraint. All eight
input lists and arguments match the complete original shell invocation.
The supplied directory defines that environment's dependency policy. The root
`pyproject.toml` retains native build, runtime and optional-extra authority.

Run `python -m tools.development.requirement_locks --source-root /path/to/requirements --output /path/to/locks --pip-compile /path/to/installed/pip-compile`.
The supplied `deps` directory must contain the original eight profile filenames.
Relative requirement/constraint includes within that copied tree retain their
meaning. Absolute external includes remain the author's responsibility. The
staged invocation uses explicit profile paths and does not inherit configuration
from the source directory's ancestors. This is an explicit native scope choice.

## Actual compiler and dependency boundary

The isolated optional acceptance environment uses pip-tools 7.5.2, build 1.3.0,
click 8.3.1, packaging 25.0, pyproject-hooks 1.2.0, tomli 2.3.0 and wheel 0.45.1
from the complete frozen pip-tools lock. Pip 25.3 and setuptools 80.9.0 are
explicit bootstrap choices because the original lock omits these unsafe pins.
Python is 3.12.14; the original generated lock header describes Python 3.10.
This acceptance does not claim platform-independent lock resolution.

These are external development tools installed into an isolated directory.
No framework runtime dependency or second production manifest is introduced.
The external dependency receipt records exact metadata, bundled license-file
hashes, frozen lock hash and the fixture wheel hash. Pip-tools and click retain
BSD notices. Build, pyproject-hooks, tomli, wheel, pip and setuptools retain
MIT notices. Packaging retains its Apache-2.0/BSD-2-Clause terms. Pip's bundled
third-party license files remain present and hash-recorded. Tool replacement
uses the installed executable argument and the same command/output contract.
Removal requires deleting this optional compiler and its documentation.

Both unchanged source and native module CLI execute actual pip-compile over
an offline local wheel with the same exact pins. A nested relative include is
resolved by both. All eight generated lock files are byte-identical. A second
native invocation seeds existing output pins into resolution and preserves
unchanged output timestamps. Only the fixture dependency is resolved in this
comparison. The full scientific dependency lock parents remain separately open.

## Failure and publication semantics

The source stops at the first failing pip-compile and can leave earlier output
locks updated. Every one of its eight failure positions is exercised unchanged.
Native compilation occurs in a temporary input/output tree. All results are
read before publication, so compiler failures, timeouts and missing generated
outputs leave the destination untouched. The temporary directory is cleaned.
Each compiler subprocess has an explicit 1,200-second bound.

Existing destination profile locks seed each staged output. This preserves the
source resolver's reuse of existing pins. Unrelated destination files remain
present. The accepted WriteIfDifferent helper publishes each changed profile
and preserves equal-file timestamps. It uses ordinary writes. A filesystem
write failure can leave a prefix of updated outputs; callers may rerun the
complete command. This is not a multi-file filesystem transaction.

## Acceptance and remaining scope

Source/default arguments, eight actual compiler outputs, nested includes,
all source/native failure positions, timeout cleanup, missing inputs/output,
existing-pin seeding, equal-content timestamps and actual CLI are covered by
`tests/unit/test_development_requirement_locks.py`. The concentrated batch
passes 350 cases with zero skips and 36 existing warnings. Final focused
acceptance passes 23 cases, including the later full-tool metadata and
patch-version counterexample. Strict typing passes both changed Python files;
Black, Ruff, Integration scope, architecture, IR, API and diff checks pass.

The compiler emits no quantum primitives and owns no state, measurements or
work registers. These resource categories do not apply. Classical resolution
and output-file counts remain separate from quantum resource estimates.

Q-0099/Q-0107 are accepted through the complete seven-pin installed environment,
all package requirement checks for Python 3.10.19/3.12.14 marker environments and actual
pip-compile execution. The marker checks are metadata checks; actual execution
uses Python 3.12.14. Build 1.3.0 requires importlib-metadata on Python
patch versions before 3.10.2; the original pip-tools lock omits that dependency.
The test preserves this exact patch-level counterexample and makes no claim
for Python 3.10.0/3.10.1 lock completeness. Q-0093 Dockerfile, Q-0111 container wrapper and the remaining
Q-0094 through Q-0110 dependency declarations/locks remain open for their own
full ownership and platform acceptance. This command's working fixture does not close those
parents. Current full-suite timeout, Algorithms NumPy ownership, copyright
order interaction and pending Proposals 116/117/118 remain tracked separately.
Q-0039 remains open pending actual-browser review after the local-URL policy
refusal.
