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

339 tokens in 148 containers across 66 files became 44 tokens in 32 containers across
22 files. 295 tokens were reworded, on 260 lines. The 44 that remain are listed, one row per
container, with the reason each one keeps its spelling.

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

## What the 44 kept tokens are

The gate groups them into nine reasons, one `[[docstring.keep]]` row per reason. A row
names the containers and the exact multiset of tokens they may show.

| Reason | Tokens | Why it keeps its spelling |
|---|---:|---|
| a private parameter | 21 | `[boundary]` is the public function surface, so a private parameter is outside the migration and the docstring that explains it keeps its spelling |
| a frozen payload key | 8 | the spelling of a serialized key — `CircuitIR.n_wires`, `Instruction.wires`, and the `n_wires` field of the two simulation records that serialize it — so renaming it is a payload-schema change rather than a rename |
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
docstrings 35 over 25 docstrings in 17 files
comments   9 over 8 comments in 7 files
union      22 files
```

The snippet counts *substring* occurrences, which is the metric the baseline above
was measured with, and it counts every `.py` under `flagquantum/`. The census below
counts *tokens* and is what the gate reconciles; the two agree at 44 here because
every remaining occurrence is identifier-shaped.

The census:

```console
$ python -c "import sys; sys.path.insert(0, 'tools'); \
    from census_wire_vocabulary import docstring_census as f; print(len(f()))"
44
```

```console
$ python -c "import sys; sys.path.insert(0, 'tools'); \
    from census_wire_vocabulary import docstring_census as f; \
    print(len({c for c, _ in f()}))"
32
```

The gate:

```console
$ python tools/check_qubit_vocabulary.py
...
Qubit vocabulary documentation: 4 of 4 documented wire-named keywords exempt as recorded (0 to reword)
Qubit vocabulary docstrings and comments: 44 of 44 wire-named tokens kept as recorded (0 to reword)
Qubit vocabulary call sites: 0 keywords passed that their callee rejects (invariant: 0)
$ echo $?
0
```

The record's table is generated from the census rather than typed, and the generator
is the same walk the gate uses, so the two cannot disagree.

## Negative tests

A gate that reports zero findings is worth nothing until it has been shown to fail.
All four of these were run against the finished tree and then reverted; the gate
returns to 0 afterwards.

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

4. **A measurement that drifts.** Changing `measured_tokens` from 44 to 43:

   ```text
   qubit vocabulary docstring measured_tokens must equal the live scan
   exit 1
   ```

## Why there is no API change proposal

`AGENTS.md` rule 8 protects exports, signatures, defaults, result fields and
serialized schemas. This slice changes none of them: no name is renamed, no signature
moves, and no payload key changes its spelling. What it changes is the prose the
package prints as `help()`, and `[docstring]` is where that is authorized and
reconciled — exactly as `[documentation]` authorized the previous slice, which is why
this slice adds a `docstring_authorization` key to `[verification]` and no new
`docs/development/API_CHANGE_PROPOSAL_*.md`.

The distinction that decides it is the one the previous slice drew: a documented
keyword argument is an *instruction* a reader copies into a script, and a docstring
is a *description* of code that already exists. The first is a signature under a
different spelling and needs the same authorization a signature does. The second is
not a signature at all.

## The measurement this slice moved that it did not intend to

`[other_surfaces].message_strings` was 632 and is now 537.

`message_strings()` walks every `ast.Constant` string under the public modules and
reports the ones containing `wire`. A docstring **is** an `ast.Constant` string, so
every docstring this slice reworded far enough to stop naming a wire spelling left
that surface with it. The drop is the migration working.

This surface is deliberately reported and never reconciled — the contract's
`message_strings_condition` says so, and says the number has drifted three times now:
`634` recorded against `632` measured, `641` after the drawer compatibility fix, and
`537` after this slice. Each drift had a different and correct cause, which is the
argument for keeping the surface reported. A ledger reconciled to this number would
have blocked the drawer fix that made legacy device objects drawable and would have
blocked this slice.

## What the slice does not reach

* `benchmarks/`, `tools/` and `examples/` are outside `[boundary]`, which stops at the
  package. Their docstrings and comments are not scanned here.
* A docstring in a *test* is not scanned either: the census walks `flagquantum/`, and
  a test's prose is not what `help()` prints.
* The command-line flag surface, the `benchmarks/` helper parameters, the capture
  keyword and the payload keys in the migration reference's table are untouched and
  remain the open work.
