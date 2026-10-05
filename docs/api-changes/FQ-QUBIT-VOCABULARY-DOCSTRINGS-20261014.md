# The docstring and comment slice of the qubit vocabulary migration

**Date:** 2026-10-14
**Slice:** `WD-2`
**Surface:** wire-named tokens in Python docstrings and comments inside the package
**Contract:** `contracts/qubit-vocabulary-contract.toml`, section `[docstring]`
**Gate:** `tools/check_qubit_vocabulary.py`, function `_docstring_errors`
**Census:** `tools/census_wire_vocabulary.py`, function `docstring_census`
**Predecessor:** [the documentation slice](FQ-QUBIT-VOCABULARY-DOCUMENTATION-20261013.md)

## What this slice decides

Every wire-named token in a docstring or a comment inside `flagquantum/` is either
reworded into the qubit vocabulary or recorded in `[docstring]` with a reason, and
the gate reconciles the recorded multiset against a live scan on every run.

337 prose tokens in 146 containers across 66 files became 43 tokens in 31 containers
across 22 files. 294 tokens were reworded, on 259 lines across 54 files. The 43 that
remain are listed, one row per container, with the reason each one keeps its spelling.

Two further tokens live inside `>>>` blocks, which are code rather than prose and are
counted by `[docstring_example]`. They did not move, and they must not: see
[The example that had no legal spelling](#the-example-that-had-no-legal-spelling).
The first three commits of this slice reported the surface as 339 tokens becoming 44,
because the scan could not tell those two lines from a sentence about them. The fourth
commit is the correction, and it is also the repair of the one line those first three
commits broke.

This is the last user-facing surface the migration had left inside the package. A
docstring is what `help()` prints and what an editor shows on hover; a comment is
what the next reader of the module is taught. Both reach a user exactly as a
signature does, and both were invisible to every check the migration already had,
because those read an AST signature and a docstring is not one.

## Why it needed a surface of its own

The five checks that existed before this slice read, in order: parameter names,
attribute names, module-level definition names, documented keyword arguments, and
call-site keywords. A docstring is none of those. That is not a hypothetical gap:
`flagquantum/_api.py` opened with

```python
>>> fq.compile(fq.Circuit(2).h(0).cx(0, 1)).n_wires
```

which names a frozen payload key and is correct, while
`flagquantum/algorithms/kmedians.py` explained its own width policy as

```
The register width in wires: ``1`` for two centroids, and ``3`` for the eight
```

which is the migration's own noun. Both are prose attached to a name. Neither is a
name, and only one of them was right.

## The rule

An occurrence is a **token**, not a line and not a scalar count:

* A token is an identifier-shaped word containing `wire`. `n_wires - 1 - wire`
  inside a code span is two tokens; `test_multi_wire_routing_locality.py` is one.
* A token **keeps** its spelling when it names something the package still has: a
  frozen payload key, a live attribute, a private parameter (which `[boundary]`
  places outside the migration), a retired spelling the package still accepts as an
  alias, a live private switch, or the file name of a test the package ships.
* A token **moves** when it is the English common noun for a qubit.
* Nothing inside a backticked code span is ever rewritten, because a backticked
  token is naming a code entity by construction, and every code entity that still
  exists keeps its spelling.

That last rule is what keeps the migration honest about the code it describes.
`flagquantum/simulation/statevector/local.py` said

```
# The sign of a wire is read from bit ``n_wires - 1 - wire``.  A wire outside
```

and becomes

```
# The sign of a qubit is read from bit ``n_wires - 1 - wire``.  A qubit outside
```

The prose moved and the expression did not, because the expression is the code.

## What the 43 kept tokens are

The gate groups them into nine reasons, one `[[docstring.keep]]` row per reason. A row
names the containers and the exact multiset of tokens they may show.

| Reason | Tokens | Why it keeps its spelling |
|---|---:|---|
| a private parameter | 21 | `[boundary]` is the public function surface, so a private parameter is outside the migration and the docstring that explains it keeps its spelling |
| a frozen payload key | 7 | the spelling of a serialized key — `CircuitIR.n_wires`, the `n_wires` field of `EvolutionPlan.n_qubits` and `TEBDResult.n_qubits`, and the twin evidence envelope that serializes it — so renaming it is a payload-schema change rather than a rename |
| a retired spelling the package still accepts | 4 | `sharded_wires`, `wire_options`, `show_wire_labels`, `active_wire_notches`; the package still reads each and warns, so the docstring that explains the compatibility promise has to say the name |
| the field or parameter of the definition the docstring explains | 3 | `_Mechanism.wire` twice and `_checked_wire`'s own `wire` parameter; a docstring that documents a name has to use that name |
| a live public attribute | 2 | `HamiltonianTerm.max_wire`, `ExecutionPlan.shardable_wires` |
| a third-party object's own spelling | 2 | a legacy device object's `wires` in `drawer/ir_adapter.py`; the object is the caller's, and the package re-expresses the spelling as `qubits` at that boundary |
| the word itself, quoted | 2 | the sentence is about the retired spelling rather than using it |
| a live private switch that names its own spelling | 1 | `_cpu_single_wire_elementwise_enabled` |
| a shipped test's file name | 1 | `tests/team/compiler/test_multi_wire_routing_locality.py`, named where the table it feeds is declared |

The gate does not take a total. It compares the multiset of tokens **per container**,
so a second occurrence inside an already-exempt docstring needs its own row rather
than being absorbed by the first. The key is `relative::Qualname::token`, with the
qualname dotted: `A._prepare` and `B._prepare` are different docstrings that reach
different readers, and keyed by the bare `_prepare` they would be one container whose
single row covers an occurrence nobody had looked at. A docstring is keyed by its own
definition; a comment is keyed by the innermost definition containing it, which keeps
the key stable when lines move inside that definition.

A `>>>` block inside a docstring is not prose, and this slice learned that the hard
way — see [The example that had no legal spelling](#the-example-that-had-no-legal-spelling).
Those lines are counted by a separate `[docstring_example]` section, and the two
surfaces are disjoint: the union of `docstring_census()` and
`docstring_example_tokens()` is every wire-named token the package's docstrings and
comments carry, with nothing counted twice and nothing dropped.

Two definitions that share a bare name were in fact the one thing this slice got wrong
on its first attempt. The key was the bare `node.name` for a docstring and the dotted
path for a comment, so five recorded rows named containers the live scan no longer
produced and the gate called them stale. The fix is not a wider exemption but a
narrower key, and the two tests that pin it — `A._prepare` against `B._prepare`, and a
closure against the method that holds it — are the regression.

## Evidence

Baseline, at `f44909ef` — the same script below, run against that revision:

```text
docstrings 285 over 131 docstrings in 59 files
comments   54 over 44 comments in 18 files
union      66 files
```

After this slice, from the repository root:

```python
import ast, io, pathlib, tokenize

docs = com = doc_nodes = com_lines = 0
doc_files, com_files = set(), set()
for path in sorted(pathlib.Path("flagquantum").rglob("*.py")):
    source = path.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            text = ast.get_docstring(node, clean=False)
            if text and "wire" in text.lower():
                docs += text.lower().count("wire")
                doc_nodes += 1
                doc_files.add(str(path))
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT and "wire" in token.string.lower():
            com += token.string.lower().count("wire")
            com_lines += 1
            com_files.add(str(path))
print(f"docstrings {docs} over {doc_nodes} docstrings in {len(doc_files)} files")
print(f"comments   {com} over {com_lines} comments in {len(com_files)} files")
print(f"union      {len(doc_files | com_files)} files")
```

```console
$ python the-snippet-above.py
docstrings 36 over 26 docstrings in 17 files
comments   9 over 8 comments in 7 files
union      22 files
```

The snippet counts *substring* occurrences, which is the metric the baseline above
was measured with, and it counts every `.py` under `flagquantum/`. It also counts the
two executable example lines, because a substring walk cannot tell a `>>>` line from a
sentence about it. Its docstring figures are one higher here than the three-commit tree
read — 36 over 26, against 35 over 25 — because restoring `.wires` on the
`Circuit.compose` example is what makes that docstring match `wire` again. The census
below counts *tokens*, splits the prose from the examples, and is what the gate
reconciles; the two agree at 43 prose tokens here because every remaining prose
occurrence is identifier-shaped.

The census:

```console
$ python -c "import sys; sys.path.insert(0, 'tools'); \
    from census_wire_vocabulary import docstring_census as f; print(len(f()))"
43
```

```console
$ python -c "import sys; sys.path.insert(0, 'tools'); \
    from census_wire_vocabulary import docstring_census as f; \
    print(len({c for c, _ in f()}))"
31
```

```console
$ python -c "import sys; sys.path.insert(0, 'tools'); \
    from census_wire_vocabulary import docstring_example_tokens as f; print(len(f()))"
2
```

The same two functions run against the baseline revision `f44909ef`, by pointing them
at that worktree's package root, read `337` prose tokens over `146` containers in `66`
files and `2` executable tokens over `2` containers in `2` files. The figure this
record carried before the split — 339 prose tokens — was those 337 plus the 2 example
lines, and the 295 tokens it reported as reworded were 294 reworded prose tokens plus
the one example line this slice rewrote by mistake.

The line and file figures move with it. The first commit's own `--shortstat` is 55
files and 260 replaced lines in each direction; one of those lines is the executable
example, which the fourth commit puts back, so the slice's net effect on
`flagquantum/` is **54 files and 259 replaced lines**. Measured on the finished tree:

```console
$ git diff -U0 d24a1b0e HEAD -- flagquantum/ | grep -cE '^[-+][^-+]'
518
$ git diff --name-only d24a1b0e HEAD -- flagquantum/ | wc -l
54
```

518 is 259 lines on each side, and `flagquantum/circuit.py` is not in the file list at
all: the only thing this slice ever changed in it was the example line, and the fourth
commit undoes that.

The gate:

```console
$ python tools/check_qubit_vocabulary.py
...
Qubit vocabulary documentation: 4 of 4 documented wire-named keywords exempt as recorded (0 to reword)
Qubit vocabulary docstrings and comments: 43 of 43 wire-named tokens kept as recorded (0 to reword)
Qubit vocabulary docstring examples: 2 of 2 executable wire-named tokens pinned by name rather than owed as debt
Qubit vocabulary call sites: 0 keywords passed that their callee rejects (invariant: 0)
$ echo $?
0
```

The record's table is generated from the census rather than typed, and the generator
is the same walk the gate uses, so the two cannot disagree.

## The example that had no legal spelling

This is the defect this slice shipped and then found, and it is the reason the
executable example is now a surface of its own.

The `Circuit.compose` docstring ends with an example:

```python
>>> fq.Circuit(4).x(0).compose(bell, qubits=(1, 2)).to_ir().instructions[-1].wires
(1, 2)
```

The second commit of this slice reworded the prose in that docstring and, in the same
pass, reworded that line to `.qubits`. The vocabulary census then read one fewer token
and the record booked it as a successful migration. It is not a successful migration:
`Instruction` carries no `qubits` attribute, so the example raises

```text
AttributeError: 'Instruction' object has no attribute 'qubits'
```

and `tests/api_contract/test_public_docstring_examples.py` fails.

**That test is wired into the pull-request lane, and this record said it was not.** The
first three commits of this slice carried the claim that the file is marked
`api_contract`, that the lane's marker expression does not select that marker, and that
the broken example therefore "passed CI green". All three clauses are false, and the
correction is worth more than the original claim:

* the file carries `pytestmark = pytest.mark.unit` (`tests/api_contract/
  test_public_docstring_examples.py:24`), not an `api_contract` marker — no test file in
  this repository contains the string `api_contract`, and `pytest.ini` does not declare
  it;
* the pull-request lane's tier command is `python -m pytest -m "smoke or unit" -q`
  (`tools/ci_tier.py:52`), run with `testpaths = tests` (`pytest.ini:2`), so it collects
  this module. Verified rather than reasoned:
  ```console
  $ python -m pytest -m "smoke or unit" --collect-only -p no:cacheprovider \
      | grep test_public_docstring_examples
        <Module test_public_docstring_examples.py>
          <Function test_primary_public_docstring_examples>
          <Function test_every_module_with_examples_is_covered>
  $ python -m pytest -m "smoke or unit" tests/api_contract/test_public_docstring_examples.py -q
  2 passed in 2.70s
  ```
* the expression this record quoted,
  `(smoke or unit or integration or jax) and not slow and not qiskit and not cudaq and
  not triton`, is not the tier command at all.

So the gate exists and is wired, and the real reason the defect would have reached the
base branch is a different one:

```yaml
on:
  pull_request:
    branches: [main]
```

`ci.yml` fires for pull requests **targeted at `main`**. Every slice of this stack
targets another feature branch — this one's base is
`feat/qubit-vocabulary-algorithms` — so **no CI ran on it at all**:

```console
$ gh pr checks 493
no checks reported on the 'feat/qubit-vocabulary-docstrings' branch
$ gh pr checks 492      # a pull request whose base is main
coverage  fail  34m16s  ...  cpu-core (3.10)  fail  22m17s  ...
```

The defect was found by running that file by hand, and the reason it had to be is
narrower and more embarrassing than "no lane runs it": **the broad sweep this slice ran
to call itself verified was `pytest tests/unit -m "smoke or unit"`, and a command that
names a path is not a lane.** The tier command names no path; restricting it to
`tests/unit` excluded `tests/api_contract/` entirely, which is precisely the directory
this file lives in. A green sweep proves something about the selection it ran, and
nothing about the selection CI runs.

Two things are true at once here, and the second one is the design lesson:

* The census was **right** to demand that the token move. `Instruction.wires` is one of
  the seven stable-core-reachable owners of the spelling, it has no `qubits` alias, and
  a docstring that says `.wires` while a slice is retiring the vocabulary is exactly
  what the surface exists to flag. The token cannot be reworded, because there is no
  other spelling of it.
* A prose scan **cannot** tell a `>>>` line from a sentence about one, because both are
  the same `ast.Constant`. So the surface has to be split, and the split has to be
  reconciled apart, because the two halves obey different rules: prose obeys the
  vocabulary rule and is debt the migration owes; an example obeys
  "the doctest must keep working" and is not debt at all.

So the fix has three parts, all in one commit:

1. `docstring_census()` now skips the lines of a `>>>` block, and
   `docstring_example_tokens()` reads exactly those lines. `_example_lines()` is the
   shared definition of a block: a line whose stripped form starts with `>>>` opens one,
   a blank line closes it, and an output line keeps it open.
2. The example line is restored to `.wires`, which is what makes the doctest pass.
3. `[docstring_example]` is added to the contract, reconciled per container with the
   same rigor as the prose surface but with a different `condition`: the two tokens
   there are *pinned*, not owed. A future pass that touches one gets a gate failure
   instead of a green run.

The prose count fell from 44 to 43 as a side effect, and that fall is correct: one of
the 44 was never prose. The one kept token that the restore put back —
`flagquantum/_api.py::compile::n_wires`, the other executable example, which was
already written in the deprecated spelling before this slice and which the old reading
had mis-filed under "a frozen payload key" — moved to the example surface with it.
`flagquantum/circuit.py` leaves the diff entirely, because the example was the only
thing this slice ever changed in it.

**The rule the first commit stated, and the hole in it.** That commit's message says
"nothing inside a backticked code span was touched, because a backticked token names a
code entity and every code entity that still exists keeps its spelling." The rule is
right and it was applied correctly — ` ``n_wires - 1 - wire`` ` in
`simulation/statevector/local.py` survives a sentence that moved to qubits around it.
The hole is that the rule enumerated *one* class of code entity, the backticked span,
and a `>>>` line is a second class it never considered. A rewriter that carries a rule
listing the cases it must not touch will still touch every case the list forgot, and
this one forgot the only class that a tool actually executes.

The honest generalisation, after the correction above: **a vocabulary census cannot audit
a doctest, and a sweep that names a path cannot audit the lane CI runs.** The first half
is why this slice needed a new surface. The second half is why a defect the lane would
have caught still reached the record: the repository does run these doctests in the
pull-request lane, and the lane does not run on a pull request whose base is a feature
branch. Both remaining questions are recorded in the migration reference's open-work
table rather than resolved here.

## Negative tests

A gate that reports zero findings is worth nothing until it has been shown to fail.
All eight of these were run against the finished tree and then reverted; the gate
returns to 0 afterwards. The first four exercise the prose surface, the last four the
executable-example surface.

1. **A new wire token in a docstring.** Adding `Every wire carries one variable.` to
   `qubo_to_ising`:

   ```text
   qubit vocabulary docstring publishes a wire-named token: flagquantum/algorithms/qubo.py::qubo_to_ising ['wire']
   exit 1
   ```

2. **An exemption that permits more than the tree shows.** Adding a phantom token to
   one row:

   ```text
   qubit vocabulary docstring in flagquantum/algorithms/qubo.py::_constant_wire shows ['max_wire', 'n_wires'], but ['max_wire', 'n_wires', 'phantom_wire'] is exempt
   exit 1
   ```

3. **A stale exemption container.** Renaming a key to a file that does not exist:

   ```text
   qubit vocabulary docstring exemption flagquantum/algorithms/nowhere.py::gone is stale
   exit 1
   ```

4. **A measurement that drifts.** Changing `measured_tokens` from 43 to 44:

   ```text
   qubit vocabulary docstring measured_tokens must equal the live scan
   exit 1
   ```

5. **An executable token with no row.** Dropping the `[docstring_example]` row, which
   is what a pass that reworded an example line would effectively do:

   ```text
   qubit vocabulary docstring example shows a wire-named token: flagquantum/_api.py::compile ['n_wires']
   qubit vocabulary docstring example shows a wire-named token: flagquantum/circuit.py::Circuit.compose ['wires']
   exit 1
   ```

6. **An executable exemption that permits more than the tree shows.** Adding a phantom
   token to the example row:

   ```text
   qubit vocabulary docstring example in flagquantum/_api.py::compile shows ['n_wires'], but ['n_wires', 'phantom_wire'] is exempt
   exit 1
   ```

7. **A stale executable exemption.** Renaming the example key to a definition that does
   not exist:

   ```text
   qubit vocabulary docstring example shows a wire-named token: flagquantum/_api.py::compile ['n_wires']
   qubit vocabulary docstring example exemption flagquantum/_api.py::a_definition_that_does_not_exist is stale
   exit 1
   ```

8. **An executable measurement that drifts.** Changing
   `[docstring_example].measured_tokens` from 2 to 3:

   ```text
   qubit vocabulary docstring example measured_tokens must equal the live scan
   exit 1
   ```

The sixth and seventh are separate cases rather than one, because a container that
exists but shows the wrong multiset reaches the *over-broad* branch while a container
that does not exist reaches the *stale* branch; a fixture that fabricates a container
therefore only ever proves the second one.

## Why there is no API change proposal

`AGENTS.md` rule 8 protects exports, signatures, defaults, result fields and
serialized schemas. This slice changes none of them: no name is renamed, no signature
moves, and no payload key changes its spelling. What it changes is the prose the
package prints as `help()`, and `[docstring]` is where that is authorized and
reconciled — exactly as `[documentation]` authorized the previous slice, which is why
this slice adds `docstring_authorization` and `docstring_example_authorization` to
`[verification]` (both naming this record) and no new
`docs/development/API_CHANGE_PROPOSAL_*.md`.

The one place this slice touched something a proposal does cover is the `.wires` it
*restored*, and the restore is the point: `Instruction.wires` was never renamed, so
putting the example back is not an API change but the removal of one. Rule 8 is why
the example could not be migrated in the first place — the attribute it names is
canonical, and this slice renamed no attribute.

The distinction that decides it is the one the previous slice drew: a documented
keyword argument is an *instruction* a reader copies into a script, and a docstring
is a *description* of code that already exists. The first is a signature under a
different spelling and needs the same authorization a signature does. The second is
not a signature at all.

## The measurement this slice moved that it did not intend to

`[other_surfaces].message_strings` was 632 and is now 538.

`message_strings()` walks every `ast.Constant` string under the public modules and
reports the ones containing `wire`. A docstring **is** an `ast.Constant` string, so
every docstring this slice reworded far enough to stop naming a wire spelling left
that surface with it. The drop is the migration working. The last one back is the
slice correcting itself: restoring `.wires` on the `Circuit.compose` example line puts
one wire-bearing literal back, so the honest figure is `537 + 1 = 538`. The reading
moved twice in one slice, in opposite directions, for two different and both correct
reasons.

This surface is deliberately reported and never reconciled — the contract's
`message_strings_condition` says so, and says the number has drifted four times now:
`634` recorded against `632` measured, `641` after the drawer compatibility fix, `537`
after this slice, and `538` once this slice's own regression was repaired. Each drift
had a different and correct cause, which is the argument for keeping the surface
reported. A ledger reconciled to this number would have blocked the drawer fix that
made legacy device objects drawable, would have blocked this slice, and would then have
blocked the correction of the defect this slice introduced — three separate pieces of
right work.

## What the example surface cost, and what it cost to give that back

A new surface is not free, and the price of this one was paid by a lane that is not
this slice's. The tier command CI runs for a pull request is `tools/ci_tier.py
pr-default`, whose selection is `pytest -m "smoke or unit" -q` over `pytest.ini`'s
`testpaths = tests`. Four runs of it, all on this machine in one sitting, two at each
revision, with the second pair ignoring the two modules this slice owns:

| revision | selection | result |
|---|---|---|
| base `f44909ef` | the lane | `3 failed, 6126 passed, 249 skipped, 2138 deselected, 79 warnings in 636.02s` |
| base `f44909ef` | the lane, two vocabulary modules ignored | `3 failed, 6004 passed, 249 skipped, 2138 deselected, 79 warnings in 221.28s` |
| tip `8fd6a910` | the lane | `3 failed, 6176 passed, 249 skipped, 2138 deselected, 79 warnings in 904.08s` |
| tip `8fd6a910` | the lane, two vocabulary modules ignored | `3 failed, 6004 passed, 249 skipped, 2138 deselected, 79 warnings in 390.58s` |

Four things in that table are worth more than the wall times, and the wall times are
worth the least.

**The counts are exact and they add up.** The two modules hold 122 tests at the base
and 172 at the tip; every other test in the lane is `6004` at both revisions, and
`6126 + 50 = 6176` is the whole difference between the two full runs. So this slice
changed nothing in the lane outside its own two test modules, and the fifty new tests
are the forty-three-token ledger being reconciled plus the nine negative tests that
keep it honest.

**The lane's cost is dominated by those two modules.** At the tip, 904.08 s of lane
against 390.58 s of the same lane without them: the two modules are **513.50 s**, or
57% of the lane, for 172 of its 6176 tests.

**And that is the number the fix moved.** Measured isolated, the same two modules:

    before   tests/unit/test_qubit_vocabulary_contract.py + tests/unit/test_census_wire_vocabulary.py
             169 passed in 1128.89 s
    after    172 passed in  621.83 s

about 6.9 s per test down to 3.0 s per test, with fifty more tests in the module than
the base carried and three of them new. What is left is not the docstring scan:
`attribute_census`, `_call_site_errors`, `_payload_index` and `_member_literal_index`
re-derive every surface on every call and none of them is memoized, which was equally
true at the base. This record does not fix that and does not claim to.

**Wall clock on this machine is not a measurement.** The base's own full lane read
`367.12 s` in one run and `636.02 s` in another — same commit, same command, same
laptop, 1.7× apart — and the tip's two full runs differ the same way. So the table is
evidence for the *ratios* and for the test counts, and for nothing that would need the
two full-lane columns to be comparable across rows. A per-test cost quoted from a
single run would be a number with no error bar on a machine whose load this record
does not control.

Neither reading is a lane the stack has run in CI. `ci.yml` triggers `pull_request`
only for `branches: [main]` and this pull request's base is a feature branch, so all
four are local runs of the selection expression rather than CI results — the same
distinction this record's Evidence section was corrected for once already.

The fix is in the seventh and eighth of the nine commits on the head branch — the first
six are the surface and the ledger, the seventh reads the docstring's two halves once
and memoizes them, the eighth fixes the loop-variable reuse that made both vocabulary
tools type-check for the first time, and the ninth is this record. It is described where
it lives rather than here: the two halves of a docstring are now read in one traversal
and memoized under a content digest of the tree, with three tests that fail if the
digest is weakened to a size-and-mtime stamp, if the memo is deleted, or if the
`_example_lines` early return stops matching the loop it replaces.

Which shape the content has *after* landing is worth stating, because the head branch
and the stack branch no longer agree about it and the stack branch alone does not record
the difference. The first eight commits were squash-merged onto
`feat/qubit-vocabulary-algorithms` as `0e617c1d`, whose tree is byte-identical to the
eighth commit's; the ninth is this record, committed separately precisely because its
own file changes in none of the preceding eight, so it can be read against either shape.
The stack branch therefore carries eight commits' content as one commit and this record
as a second, while the head branch carries all nine separately — the same content, two
histories, and `git rev-parse <ref>^{tree}` at the two tips is what shows it.

The lane readings above were taken after the eighth, so they are the cost of the slice
as it will land rather than the cost it had when the surface was added. That
distinction is the whole reason the seventh commit exists, and it is the reason this
section reports both the lane and the two modules: a slice that adds a surface owes its
own cost back to the lane it shares, and the evidence that it did is a pair of readings
rather than an assurance.

## The lane's failures are the same three at both revisions

Every one of the four runs above reports exactly three failures, and they are the same
three names at the base and at the tip:

* `tests/team/runtime/test_target_capability_matching.py::test_decision_identity_is_independent_of_python_hash_seed`
* `tests/team/simulation/test_statevector_ops_numerics.py::test_native_fused_rotation_layer_only_promotes_terminal_regions`
* `tests/unit/test_native_cpu_adjoint.py::test_compact_cx_runtime_threshold_and_rollback`

None of them is this slice's, and the third is the one an earlier reading of this
branch already reproduced at the base. A *fourth* of them is worth naming because
naming it wrongly would have been easy: earlier full-lane runs at the tip reported
`tests/test_simulator_advisor.py::test_advisor_live_calibrates_an_unmeasured_circuit`
as well, with `assert 'flagquantum_native' == 'qiskit_aer'`. That test compares which
backend it decided was faster, so it is a load-sensitive reading rather than a defect:
it passed standalone, it passed under `-n 8` on its own file, it also fails at the base
under load, and it did not appear in either of the two tip runs tabulated above. The
honest description is that the lane has three stable failures at both revisions and one
intermittent test that this slice neither introduced nor fixed — and that a lane whose
failure list depends on how busy the machine is cannot be read as a pass/fail signal
without the list being written down, which is the reason this section exists.

## What the slice does not reach

* `benchmarks/`, `tools/` and `examples/` are outside `[boundary]`, which stops at the
  package. Their docstrings and comments are not scanned here.
* A docstring in a *test* is not scanned either: the census walks `flagquantum/`, and
  a test's prose is not what `help()` prints.
* The executable-example surface is now reconciled, and the test that would *catch* a
  broken example — `tests/api_contract/test_public_docstring_examples.py` — **is** in the
  pull-request lane (see "the correction" above; the earlier claim in this record that it
  is not was wrong). It did not run here because `ci.yml` triggers `pull_request` only for
  `branches: [main]`, and this pull request's base is a feature branch. So the repository's
  coverage of this file is real but unreached from this stack, and the gap is the trigger
  rather than the marker expression. Whether this stack should reach `main` as one pull
  request, and whether the stack's slices should be checked before then, are the two open
  questions; neither is answered here.
* The command-line flag surface, the `benchmarks/` helper parameters, the capture
  keyword and the payload keys in the migration reference's table are untouched and
  remain the open work.
