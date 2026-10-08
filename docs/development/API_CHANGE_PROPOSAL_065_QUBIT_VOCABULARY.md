# API Change Proposal 065: One word for a qubit, across the whole package

## Status

**Proposed.** The user authorized the rename for the Python surface; this
proposal is the decision record, and
[`FQ-QUBIT-VOCABULARY-INTEGRAL-20261005.md`](../api-changes/FQ-QUBIT-VOCABULARY-INTEGRAL-20261005.md)
is the authorization record it answers to.

The change is a **rename**, and it is deliberately the only kind of change it
contains. This proposal adds no capability, no export, no result field, no plan
property, and no serialized schema. It is written because `AGENTS.md` rule 8
protects Stable Core signatures and this rename touches them, not because a new
mechanism is being introduced — which is the first of the four questions below.

The library it changes is not yet frozen (`0.3.x`), so the rename is performed
now, while the cost is a mechanical edit, rather than after a release makes every
old spelling a compatibility obligation.

## 1. Why this is a vocabulary migration and not a new capability

A new capability has to satisfy `AGENTS.md` clause 2 of the Current Strategic
Priority: it must **replace** an existing implementation behind an existing
boundary, demonstrated by swapping one implementation without modifying its
consumers. A rename replaces nothing and is demonstrated by nothing of the sort.
What it must demonstrate instead is the opposite kind of proof: that the set of
callable behaviour is **identical** before and after, with only the names changed.

That is a checkable claim, and this program makes it in three ways:

- **The gate compares a frozen baseline against a live scan** and requires
  equality. A rename program cannot hide a capability change inside itself,
  because the baseline is a list of *names at known locations*, and every location
  that is not renamed must still be present.
- **The lexicon is closed.** The naming rule is root substitution, and four
  plausible alternative spellings are forbidden outright. A migration that started
  inventing better names for individual sites would be a redesign wearing a
  rename's clothes; the contract forbids that.
- **The work is sliced by subpackage, not by concept.** Each slice is a mechanical
  edit to one domain. A slice that found itself needing a new abstraction would be
  evidence that the boundary was wrong, and would be split out.

There is also a product reason to state plainly. The program this belongs to
aligns FlagQuantum with PennyLane semantically. PennyLane's own noun is `wires`.
Adopting the *word* while aligning the *semantics* would be alignment in the
wrong direction: the semantics are what should converge, and the vocabulary should
stay FlagQuantum's own. Keeping `wire` in the user's face makes the library read as
a re-spelling of another project.

## 2. Relationship to `AGENTS.md` rule 6 (keep the user API simple)

Rule 6 requires normal usage to stay centred on `fq.Circuit`, `fq.Module`, `run`,
`plan`, training loops, and deployment packages. This proposal **serves** rule 6 by
removing a decision the user currently has to make on every call.

Measured today, one screen of ordinary code needs both words:

```python
import flagquantum as fq

circuit = fq.Circuit(2).h(0).cx(0, 1)             # "qubit"
fq.Circuit(1).rx(qubit=0, theta=0.3)              # "qubit"
fq.Z(qubit=0)                                     # "qubit"
result = fq.run(circuit, outputs=fq.probabilities(qubits=(0,)))
result.n_wires                                    # "wire"
fq.from_engine_qir(payload)                       # n_wires="wire"
```

The user's mental model does not contain two kinds of index. Under this proposal
the same screen is one vocabulary, and the rule for guessing a keyword argument
becomes "it is called what it is".

A second, quieter cost is removed as well: today a reader who greps the package for
`n_qubits` and does not find a site has learned nothing, because the site may be
spelled `n_wires`. After the migration a grep is exhaustive.

## 3. Relationship to `AGENTS.md` rule 8 (Stable Core protection)

Rule 8 forbids renaming a stable signature without explicit user authorization and
an approved API change proposal. **Both are present**, and the boundary between
what they cover and what they do not is the substantive decision in this document.

| Tier | Definition | Signatures | Compatibility obligation |
|---|---|---:|---|
| T1 | reachable from `fq.*` | measured by the gate | yes — new name plus deprecated alias |
| T2 | reachable from a sub-namespace of `fq.*` | measured by the gate | yes, treated as T1 |
| T3 | public modules imported by path, not by `fq.*` | the remainder | no — renamed in place |
| T4 | public modules needing an optional extra to import | the remainder | no — renamed in place, verified with the extra installed |

The measured fact that makes T3 and T4 large: `fq.compiler`, `fq.algorithms`,
`fq.simulation`, `fq.noise`, `fq.qec`, `fq.core`, `fq.deployment`, and
`fq.services` are **not attributes of `fq`**. Measured,
`fq.deployment` raises `AttributeError` — the root module's `__getattr__` is an
allow-list over observables, remote jobs, and the `_api` entry points, and it
raises for anything else. A parameter in one of those modules therefore is not a
Stable Core name, and giving it an alias would preserve two vocabularies in
perpetuity for a caller who cannot legitimately exist. `AGENTS.md` engineering
decision principle 1 forbids exactly that debt.

Rule 8 also requires that a contract or snapshot is **never** updated merely to
make tests pass. That risk is real here, because
`contracts/public-api-v0.2-baseline.json` records constructor parameter names and
contains six occurrences of `n_wires`. The rule this program adopts: the baseline
file is regenerated **once, by the slice that performs the rename, with the
proposal cited in the commit message**, and a baseline diff that appears in any
other slice is a defect regardless of whether the tests pass.

## 4. Why `IR_VERSION` does not change

`flagquantum/core/ir/__init__.py` pins `IR_VERSION = "1.0"`, and the serialized payload
uses `n_wires` and `wires` as **keys**:

```python
fq.Circuit(2).h(0).to_ir().to_dict()
# {'n_wires': 2, 'instructions': [{'name': 'h', 'wires': (0,), ...}], ...}
```

`n_wires` here is a string in a document, not a parameter name. Renaming it is a
schema change, and a schema change requires a version bump, a migration path for
readers of the old schema, and a compatibility window — a different and much
larger proposal than this one, whose entire justification is that it is
mechanical.

So the serialization keys are **excluded by name** in the gate's exclusion table,
along with three other classes of value that are not FlagQuantum names:

| Excluded | Example | Why the gate must pass it |
|---|---|---|
| serialized payload key | `CircuitIR.n_wires`, `Instruction.wires` | an exact-match version pin |
| environment variable | `FQ_STATEVECTOR_PERSISTENT_WIRE_LAYOUT` | a published deployment switch |
| vendor adapter payload key | a provider schema's field | the adapter translates at the boundary (`AGENTS.md` rule 5) |
| benchmark parameter | `run_case(n_wires=...)` | renamed with the package, but with no alias obligation because it is not reachable from `fq.*` |

The exclusions are entries in the contract, not a filter inside the gate. A filter
would be invisible at review time and could be widened in a later change without
anyone seeing the widening; an exclusion table is read in the diff.

## 5. Alias lifecycle and removal version

Two populations exist and they must not be added together when reporting progress:

| Population | Count | Meaning |
|---|---:|---|
| canonical sites still spelled `wire` | 341 | the work remaining |
| deprecated aliases whose canonical spelling already exists | 11 | renamed; awaiting deletion |

Conflating them produces the false reading that eight renames constitute progress,
and it makes the gate unable to distinguish "this is a declared legacy alias" from
"this is a new violation". The gate keeps them separate and the alias test is
machine-decidable: **a `wire`-named parameter is an alias exactly when the same
signature also declares its qubit-named replacement**, which is how
`flagquantum/core/_qubit_aliases.py` implements it.

Lifecycle, unchanged from the earlier proposal:

1. A T1 or T2 site gains the qubit-named parameter and keeps the wire-named one as
   a keyword-only parameter.
2. Passing the old name calls `warn_qubit_alias`, which emits a
   `DeprecationWarning` naming both spellings.
3. Passing both raises `TypeError` rather than silently preferring one.
4. The alias is deleted at **0.4.0**, which is the version
   `warn_qubit_alias` already publishes for the eleven shipped aliases. New aliases
   inherit it; a second removal window would require a reason this proposal does not
   have.

A T3 or T4 site skips all four steps and is renamed in place, in one commit, with
no alias.

## Deviation from the proposed surface

The proposal's scope of "the user-facing surface" turned out to cover three
surfaces of very different measurability, and this document ledgers only the
first:

| Surface | Measured | Ledgered |
|---|---:|---|
| parameter names on the public function surface | 341 + 11 | yes — the gate's baseline |
| public attribute and property names | 102 | **no** — recorded as an owned gap, `WQ-2` must extend the gate first |
| string literals | 1291 | **no** — reported by the census; a literal scan cannot separate a serialized key from a refusal sentence |

This is stated here rather than left implicit because the alternative — gating the
attribute surface with a per-item judgement write-up — would have produced a
second, unreviewed copy of the serialization exclusion table inside one PR. The
gap is narrow, owned, and blocking: the owning slice cannot land until the gate
covers it.

## Verification

1. `python tools/check_qubit_vocabulary.py` exits 0 and reports `0 of 341 baseline
   sites retired`.
2. `python tools/census_wire_vocabulary.py` reproduces 341 canonical, 11 aliases,
   397 private, 102 public attribute names, and 1291 string literals.
3. `tests/unit/test_qubit_vocabulary_contract.py` proves the gate fails on: a
   newly added `wire` parameter, a fabricated retirement entry, a baseline line
   deleted without a retirement entry, an alias forwarding to the wrong name, and a
   slice whose declared count is wrong.
4. `python tools/validate_required_checks.py` still passes — the gate is added as
   a CI **step** inside the existing `quality` job, not as a new job.
5. No Stable Core export, default, or serialized key changes: `fq.__all__`
   membership and `docs/public_api_v1.json` are untouched by WQ-1.
