# Documentation command source closure

## Complete source group and owners

This batch closes Q-0042 build-reference-docs-2.py, Q-0043
build-reference-docs.py, Q-0045 clean-notebooks.py and Q-0050
execute-notebooks.py at Qualtran v0.7.0
bcb17bb25360ed68ce3bc084645db3e929799389. Each complete original script's Git
blob is checked against the immutable inventory. Original Google Apache-2.0
notices remain in their source-owned native modules.

The commands live with their existing Tools owners. Module entry points in
reference_docs and reference_templates call the already accepted static and
runtime-template generators. Module entry points in notebook_cleaning and
notebook_execution parse arguments and call their existing cleanup and execution
owners. The commands introduce no parallel generator, notebook executor,
configuration layer or public runtime API.

The native commands run with ordinary Python module invocation.

```bash
python -m tools.development.notebook_cleaning
python -m tools.development.notebook_cleaning --apply
python -m tools.development.notebook_execution --n-workers 1 --output-md
python -m tools.development.reference_templates
python -m tools.development.reference_docs
```

## Argument, failure and source comparison

Unmodified original scripts run through their real __main__ paths with replaced
imported callbacks. Native defaults and all eight export/freshness boolean
combinations match exact keyword arguments. Positive, zero and negative worker
values are passed to the existing owner unchanged. Invalid integers, missing
values and unknown options fail with parser exit status two before invoking the
owner. Source callback errors propagate.

Cleanup defaults to checking and writes only with --apply. Both --apply and
--no-apply preserve the source's dirty-notebook count as the exit status,
including after applying cleanup. The source package directory becomes the
native flagquantum directory. The existing native execution owner's package and
examples/notebooks roots stay authoritative. Static reference generation obtains
the current checkout root, while template generation retains its no-argument
root discovery. Imports alone do not execute any command.

## Actual workflows and reused acceptance

Real subprocesses in a temporary Git checkout check a tracked dirty notebook,
verify that checking preserves its bytes, apply cleanup, then verify a clean
second check. Untracked notebooks are excluded through the existing tracked-file
owner. A real module execution with no tracked inputs succeeds and prints its
empty summary. Both parsers' --help paths succeed.

A real CLI invocation executes a notebook containing circuit.append(op) and an
exact complex statevector assertion, and produces both notebook and Markdown
exports. It uses the accepted notebook owner's controlled kernel environment.
The test helper is shared with that owner's existing tests. Its environment now
also accepts an absent inherited JUPYTER_PATH. Kernel connection files are removed
after the command. Existing notebook tests retain failure aggregation, startup
failure, all export combinations, freshness, source ordering, actual spawn
execution and shared numerical execution evidence.

The complete static generator's existing inherited-setter refusal remains
observable at its CLI. The loaded native template namespace and source-profile
byte comparisons remain accepted generator evidence. Q-0036 notebook catalog
orchestration and Q-0080's one-to-one native mappings remain open. No source
notebook catalog is silently replaced by automatic discovery.

These are classical orchestration commands. Accepted complete-amplitude,
measurement-branch, work-register and separate primitive/model resource evidence
is reused from the owning implementations. The CLI adds no quantum gates.

## Final checks and broader findings

The six command workflow tests and the concentrated owner regression pass all
240 selected tests, with 901 visible upstream warnings and zero skips. Strict
mypy covers all six changed Python files. Black, Ruff, Integration scope,
architecture, multi-level IR, protected API and diff checks pass.

The bounded repository-wide smoke/unit probe selected 80,494 tests from 98,026
collected items, with 17,532 deselections and 51 collection skips. It reached
48 percent before its 1,200-second deadline. The runner exited and a process
inventory found no remaining owned runner. The log already contains an
Algorithms NumPy boundary failure and a copyright original-suite failure in the
full collection order. The Algorithms failure reproduces independently. The
copyright source suite passes in this concentrated regression, so its broader
order/environment interaction remains open. The complete default tier is
unverified. Retained evidence is in g2-template-docs-default-tier-20261121.log
and g2-template-docs-default-failure-isolated-20261121.log.
