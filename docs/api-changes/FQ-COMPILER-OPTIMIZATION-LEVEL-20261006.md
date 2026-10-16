# The optimization-level parameter on `flagquantum.compiler.compile` and `optimize`

## Status

Proposed on 2026-10-06 for review with the implementation on the same branch.
This is a compatible, additive signature change: it adds one keyword-only
parameter with a default value to two public compiler entry points, and removes,
renames, reorders, or re-defaults nothing. It adds no stable export, no root-level
function, no `ExecutionOptions` field, no `IR_VERSION` change, and no serialized
schema change. It is recorded here because
[Public API Protection](../development/PUBLIC_API_PROTECTION.md) protects
parameter names, order, and defaults, and because `compile` is a Stable Core
export name and a `production_supported` capability row.

The change is implemented and under review. It is not approved until an API owner
records approval; the entries below that claim verification are the runs that were
executed on this branch, not an expectation.

## Decision and authorization

Two public signatures change, in the same way.

Before:

```python
flagquantum.compiler.optimize(circuit_or_ir) -> CircuitIR
flagquantum.compiler.compile(
    circuit_or_ir, *, coupling_map=None, routing_strategy=..., optimize=True, config=None
) -> CircuitIR
```

After:

```python
flagquantum.compiler.optimize(
    circuit_or_ir, *, optimization_level=2
) -> CircuitIR
flagquantum.compiler.compile(
    circuit_or_ir, *,
    coupling_map=None,
    routing_strategy=...,
    optimize=True,
    optimization_level=2,
    config=None,
) -> CircuitIR
```

`optimization_level` is keyword-only, has a default, and is inserted after
`optimize` and before `config`. Inserting a parameter is only source-compatible
when every existing parameter is keyword-only, which holds here: both functions
already take their options by keyword, and every call site in this repository
passes the program positionally and the rest by keyword, which was checked before
the parameter was added rather than assumed.

This implements Alternative 2 of
[FQ-PASS-MANAGER-CONTRACT-20261001.md](FQ-PASS-MANAGER-CONTRACT-20261001.md),
which is the alternative that document recommends and the parity backlog names
W6-08. It deviates in one respect, deliberately: that document places the level
tables in `pipeline.py`, and they are in `flagquantum/compiler/optimization_levels.py`
instead. The tables are private either way -- the module is not re-exported from
`flagquantum/compiler/__init__.py` and is reached only by the loop and its tests --
and a separate file keeps the declaration, its ordering rationale, and the Qiskit
pass correspondence out of a driver that is already the package's largest module.

The **owner of the affected surface** is the compiler team, because
`flagquantum/compiler/**` is `[teams.compiler]`-owned and the implementation,
documentation, and tests of this change are all inside the paths that team may
write. The **API owner** is the separate approval this document exists to request,
for the reason above: `compile` is a Stable Core export name and
`flagquantum.compiler.compile` is a declared `public_apis` entry of the
`program_compilation` capability, which is `production_supported`.

`python tools/check_team_scope.py --team compiler --files` over the complete
change set is the check that the first of those two claims holds.

## Problem and affected user journey

FlagQuantum had exactly one optimization behaviour and no way to ask for less of
it. `optimize` ran a fixed sequence of eleven pass calls to a fixed point, and
`compile` ran it twice -- once before routing and once after, because routing
inserts SWAPs and the second pass is what cancels them. A user who wanted a
result that a target's own lowering would handle better, or who wanted to know how
much of a compiled program's size came from optimization, had to either take all
of it or bypass `optimize`/`compile` and call the pass modules directly. The pass
modules are reachable, but they are not the documented path and their composition
is not stated anywhere a user can read it.

Qiskit solves this with `transpile(..., optimization_level=...)`, and the two
libraries' pass sets correspond closely enough that a user moving between them
should not have to learn a second scale. The asymmetry was the whole gap: the
passes existed, their composition existed, and the published parameter did not.

The affected journey is small and concrete. The three programs below are the same
circuit at the three levels, and each level's extra reach is visible in the count:

```python
import flagquantum as fq
from flagquantum.compiler import optimize

program = fq.Circuit(2).x(0).x(0).rz(0, 0.3).cz(0, 1).rz(0, 0.4).to_ir()
print(len(optimize(program, optimization_level=0)))   # 5, unchanged
print(len(optimize(program, optimization_level=1)))   # 3, the x pair cancelled
print(len(optimize(program, optimization_level=2)))   # 2, the two rz merged across cz
```

Level 1 cancels the `x` pair and level 2 then reaches the second `rz` across the
`cz`, which is a rotation commuting with a diagonal gate. Level 0 returns the five
instructions the caller submitted.

## Alternatives considered

**A boolean `optimize=False`-style flag is already there, so add nothing.** The
existing flag turns optimization off entirely and has no middle. The gap is
between "all of it" and "none of it", and neither existing value covers the level
a target-lowering user wants.

**A string or enum level name** (`"light"`, `"canonical"`, `"full"`). This would
be a second vocabulary beside Qiskit's, and
[rule 9](../../AGENTS.md) rejects relative or maturity-flavoured labels where a
precise domain term exists. An integer level is a *quantity of optimization*,
which is what the parameter measures; a name would have to be re-explained
against Qiskit's numbers anyway.

**An `OptimizationLevel` enum exported from Stable Core.** This adds a new stable
export and therefore a long-term maintenance obligation, for a value domain of
four small integers that is already fully described by the parameter's
documentation. The enum can be added later without breaking anything, because an
integer is what the parameter accepts.

**A pass-manager object as the parameter**, mirroring a Qiskit
`PassManager`. This is the shape [FQ-PASS-MANAGER-CONTRACT-20261001.md](FQ-PASS-MANAGER-CONTRACT-20261001.md)
covers and it is the intended long-term direction (backlog items W11-08 through
W11-10). It is not the smallest implementation of the current requirement:
`optimize` still runs one fixed sequence, and a pass-manager parameter would
publish a composition contract this release cannot yet honour, because the
unitary-synthesis stage level 3 needs does not exist as a pass over `CircuitIR`.

**Naming nothing and documenting the composition only.** A user cannot select a
composition they cannot name, and `compile`'s second optimization point would
remain silently at the full level.

## Compatibility impact

**Source compatibility.** Unchanged for every existing caller. The parameter is
keyword-only with a default, so a call that does not name it behaves as before.
This was verified rather than assumed: the repository has more than sixteen call
sites that pass only a program, and they were left untouched and still pass.

**Behavioral compatibility.** The default is `2`, which is the composition
`optimize` and `compile` already ran. The claim "a caller that names no level
receives the program it used to receive" is a checked postcondition, not a
consequence of the default's value: a new test compares the default against
explicit level `2` instruction for instruction and measurement for measurement
over more than one program, so a future change that made level `2` narrower would
fail there rather than silently changing what a default caller gets.

**Level 3 fails closed.** Level 3 is *declared and reserved*: it is in the ladder,
so a user asking for it can be told what is missing, and it raises
`CompilationError` at the earliest knowable stage rather than quietly returning a
level-2 program. This is the
[fail-closed rule](../../AGENTS.md) applied to a number: Qiskit's level 3 differs
from its level 2 by moving `UnitarySynthesis` into the optimization loop, and this
package's synthesis entry points are not passes over a `CircuitIR` -- they answer
about one gate or one amplitude vector and are reached by module path. Narrowing a
declared ladder later is a breaking change; widening it is not, so the refusal is
the version that can be corrected cheaply. A value that is not an integer is
refused the same way, because `True` and `2.0` both compare equal to an
implemented level and would otherwise reach the ladder unvalidated.

**A new metadata key.** The level that ran is recorded on the result under
`metadata["optimization"] = {"level": <int>}`. `CircuitIR.metadata` is an open
mapping and carries keys this compiler and the runtime already write
(`"layout"`, `"routing"`, `"runtime_config"`), so this is an addition inside an
unversioned mapping rather than a schema change, and no validator constrains the
key set. Without it a level-0 program that had nothing to optimize and a level-2
program with nothing to optimize would be the same artifact, and the request would
not be recoverable from the result.

**Serialization compatibility.** None affected. `IR_VERSION` is unchanged, no
`ExecutionPlan` identity input changes, and no result field is added or reordered.

**Snapshot and contract impact.** `contracts/public-api-v0.2-baseline.json`
records no `flagquantum.compiler.compile` entry, so the protected baseline
signature is not the surface this change touches; the root `fq.compile`, which is
the Stable Core export, does not forward the new parameter and its signature is
unchanged. `python tools/public_api_snapshot.py` was run on this branch with the
change in place and reported `public API migration baseline passed`. That is
evidence about this snapshot, not a licence to regenerate one: no snapshot, and no
file under `contracts/`, is modified by this change.

## Migration example

No migration is required. A caller that wants the earlier behaviour explicitly can
name it:

```python
from flagquantum.compiler import optimize

optimize(program, optimization_level=2)   # the default, spelled out
```

## Documentation and tooling impact

- `flagquantum/compiler/README.md` gains a section stating the ladder, the default,
  the level-3 reservation, and the per-level stage table.
- `docs/reference/API.md` states the parameter where it already documents
  `optimize` and `compile`.
- `docs/reference/RELEASE_NOTES.md` gains an entry, because a new parameter on a
  public entry point is user-visible.
- The capability rows for `program_compilation` already name
  `flagquantum.compiler.optimize` and `flagquantum.compiler.compile` as their
  `public_apis`, so no capability label changes and no maturity level is promoted
  by this change.
- No tool, contract, CI step, or generated document changes.

## Evidence

Executable behavior contracts (the change's own test file, run and green on this
branch):

- `tests/team/compiler/test_optimization_levels.py` asserts the declared domain,
  resolves every declared pass name through the loop's own roster, asserts level 2
  is the whole library and level 1 a subsequence of it, and asserts per-level
  behavior on programs chosen so that one property is decidable per level --
  including the two levels' difference on a diagonal gate before a measurement and
  on rotations separated by a commuting gate.
- The same file asserts each level preserves the state and never grows the
  program over forty seeded programs, and that each level is strictly narrower
  than the one below on a measured count of them, so the ladder is measured rather
  than declared.
- `tests/team/compiler/test_optimization_pass_conformance.py` now reads the round
  body from the declaration and asserts the declaration is level 2, so the fixed
  point's composition and the level table cannot drift apart.
- The existing optimization suite, which drives more than sixteen call sites that
  name no level, is the source-compatibility evidence.

## Owner and approvals

- **Owner:** compiler team, 2026-10-06.
- **Affected surfaces:** `flagquantum/compiler/{pipeline,optimization_levels}.py`,
  `flagquantum/compiler/README.md`, `tests/team/compiler/**`,
  `docs/reference/{API,RELEASE_NOTES}.md`.
- **Team scope:** `python tools/check_team_scope.py --team compiler --files` over
  the complete change set reports `team ownership policy passed`.
- **Approval required from:** the API owner, for adding a parameter to a
  `production_supported` capability's declared public entry points; the integration
  maintainer, for the documentation surfaces that live under `docs/`.
