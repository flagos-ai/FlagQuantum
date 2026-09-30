# Pauli algebra and quantum_info ownership boundary

## Decision and authorization

Status: **proposed, not approved.** This document records the boundary defects, the
evidence, and the intended end state so that the integration owner, the API owner,
and the core domain owner can approve or reject it. No code change is included, and
no approval is claimed. Implementation must not land before that approval.

The change is proposed under `docs/development/PUBLIC_API_PROTECTION.md`, because
the intended end state relocates a Stable Core name and changes the semantics of
another. It exists to answer one question before the 42-PR operator-algebra
workstream starts: **where does the Pauli algebra live, and which of the
representations already in the repository is canonical?** The parity backlog
schedules all 42 of those PRs as `feat(observables):`, and nothing in the
repository currently establishes that `observables/` is the right owner, that it
is a domain at all, or that the type it exports is the one the rest of the
repository uses.

## Problem and affected user journey

### `observables/` owns 29% of the stable public API while not being a domain

`observables/` is one of the 20 entries in `architecture.toml`
`allowed_top_level_directories`, and 10 of the 34 names in
`docs/public_api_v1.json` `stable_exports` come from it:

| Stable Core name | Defined at |
| --- | --- |
| `Observable` | `flagquantum/observables/__init__.py:25` |
| `OutputRequest` | `flagquantum/observables/__init__.py:88` |
| `I`, `X`, `Y`, `Z` | `flagquantum/observables/__init__.py:182-206` |
| `expectation`, `probabilities`, `samples`, `counts` | `flagquantum/observables/__init__.py:212-277` |

That is 10 of 34 stable names, or 29% of the documented public API, from one
package. The same package is absent from every architectural record that would give
it an owner or a direction:

| Record | Contains `observables`? |
| --- | --- |
| `architecture.toml` `allowed_top_level_directories` (20) | yes |
| `architecture.toml` `[layers] order` (9 layers) | **no** |
| `contracts/long-horizon-architecture-v1.json` `domains` (11) | **no** |
| `tools/check_architecture.py` boundary checks (7 prefixes) | **no** |

The 11 long-horizon domains are `core`, `compiler`, `runtime`, `simulation`,
`noise`, `compute`, `remote`, `twin`, `ecosystem`, `services`, `gateways`. A
package that owns 10 Stable Core names and appears in none of them is governed by
its own README and nothing else.

Nor is it a leaf. Fifteen files import it, at 17 static import sites plus 3 dynamic
`import_module(".observables", ...)` lookups, across ten directories: the root
package (`__init__.py:89` resolves ten names lazily; `_api.py:13,121,261`;
`circuit.py:33`), `ecosystem/cirq` (`simulator.py:14`), `ecosystem/pennylane`
(`lightning.py:13`), `ecosystem/qiskit` (`aer.py:26`), `lindblad/__init__.py:13`,
`remote/jobs.py:29`, `remote/compute` (`_native_job.py:18`,
`_program_submission.py:15,59`, `execution.py:13`, `jiuding.py:40,64`),
`remote/qpu/execution.py:17-18`, `services/managed_simulator.py:84`, and
`simulation/lindblad.py:12`.

### The module is two responsibilities with a 17-to-1 difference in reach

`flagquantum/observables/__init__.py` is 400 lines with one `__all__` of ten names,
and it contains two unrelated things:

| Responsibility | Lines | Stable Core names | Import sites needing it |
| --- | --- | --- | --- |
| Output-request contract: what `fq.run` / `fq.plan` accept and how it lowers to IR | 88-113, 212-285, 297-388 | `OutputRequest`, `expectation`, `probabilities`, `samples`, `counts` | **16 static + 2 dynamic** |
| Pauli observable algebra | 15-86, 178-211, 286 | `Observable`, `I`, `X`, `Y`, `Z` | **1** |

The single external consumer of the algebra is
`flagquantum/simulation/lindblad.py:12`, which imports `Observable` for the
`observable` parameter of one function (`lindblad.py:432`). Every one of the other
16 static import sites wants `OutputRequest`, `lower_outputs`, or `counts`, and the
remaining dynamic lookups are the root lazy-export block (both halves) and
`_api.py:121,261` (`lower_outputs`).

The package README states both responsibilities in one sentence, which is how the
mixture survived: "This package owns the public mathematical observable model and
the output requests accepted by `fq.run` and `fq.plan`." Those two things have no
reason to share a module, and the 42-PR workstream is scheduled to land entirely in
the half that one file uses.

### Eight Pauli representations already exist

Before adding a single class, the repository already represents a weighted Pauli
product eight different ways:

| # | Representation | Definition | Wire identity | Phase | Coefficient type |
| --- | --- | --- | --- | --- | --- |
| 1 | `observables._PauliTerm` | `observables/__init__.py:19` | `factors: tuple[tuple[int, str], ...]` | no | `float` |
| 2 | `statevector._PauliTerm` | `runtime/executors/statevector/split_real_imag.py:232` | `ops: tuple[tuple[int, str], ...]` | no | `Any` |
| 3 | `algorithms.HamiltonianTerm` | `algorithms/core.py:86`, built by `pauli_term()` at `:342` | `pauli: str \| Mapping[int, str]` plus `wires`, with derived `.ops` | no | `float \| complex \| torch.Tensor` |
| 4 | `qec.Pauli` | `qec/pauli.py:35`, exported via `qec/__init__.py:59` | symplectic `x_wires` / `z_wires` | **explicitly not tracked** | none (operator only) |
| 5 | `services.PauliSum` / `PauliSumTerm` | `services/ground_state.py:31-40` | `pauli: str` plus `n_qubits` | no | `float`, unit `hartree` |
| 6 | `noise._pauli_basis` | `noise/channels.py:35` | position in a `torch.Tensor` tuple | no | `torch.Tensor` |
| 7 | `simulation.pauli` functions | `simulation/pauli.py:25,50,71` | arguments only; no type | no | `torch.Tensor` |
| 8 | `deployment.PauliMeasurementPlan` | `deployment/cloud.py:154` | delegates to #3 and `runtime.ObservableGroup` | no | delegates |

Two of these are **classes with the same name in different packages and
incompatible fields**: `observables/__init__.py:19` declares
`_PauliTerm(coefficient: float, factors: ...)` and
`runtime/executors/statevector/split_real_imag.py:232` declares
`_PauliTerm(coefficient: Any, ops: ...)`. The field named `factors` in one is named
`ops` in the other, and the coefficient is a `float` in one and unconstrained in
the other. Both are private, so no gate sees the collision, and a grep for
`_PauliTerm` returns two unrelated types.

Representation 4 is the only one that is public and exported, and its docstring
records the deliberate omission that separates it from Qiskit's `Pauli`: "Global
phase is not tracked, so an operator is described only up to a factor of `+/-1` or
`+/-i`." Any future Pauli algebra has to decide whether it inherits that omission.

Representation 5 carries a serialized contract
(`schema = "flagquantum.pauli_sum.v1"`, `coefficient_unit = "hartree"`,
`basis_convention = "wire_0_is_leftmost"`), so its wire order is a wire-protocol
fact, not an internal detail.

### The two worlds that carry Pauli data never meet

The repository has two parallel Pauli universes and no bridge between them:

- **The public algebra.** `observables.Observable`, Stable Core, one production
  consumer.
- **The internal Hamiltonian algebra.** `algorithms.HamiltonianTerm` flows
  `algorithms` -> `runtime/parallel.py:82 group_observables()` ->
  `deployment/cloud.py:154 PauliMeasurementPlan` -> `remote`.

A search for any import of `observables` from `flagquantum/algorithms/` or
`flagquantum/runtime/` returns nothing. `runtime/parallel.py:13 ObservableGroup`
does not even hold terms: it holds `term_indices: tuple[int, ...]`, indices into a
caller-owned `HamiltonianTerm` sequence, and `basis: tuple[tuple[int, str], ...]` —
a third spelling of the same wire/name pairs as column 1 and column 3 above.
`deployment/cloud.py:154` then binds an `algorithms.Hamiltonian` to
`runtime.ObservableGroup`s, so the deployment layer depends on both worlds
simultaneously and converts between neither.

This is the ownership defect in one sentence: **the type the repository calls
public is not the type the repository uses.** `group_observables` is the only
commuting-group implementation (`runtime/parallel.py:82`) and it is qubit-wise
only, which the backlog correctly notes; but it also operates on the internal type,
which the backlog does not mention. A `group_commuting` added to `observables`
would be a second implementation over a second type.

### W12 would add a fourth algebra to the one-consumer half

The parity backlog schedules 42 PRs over 9-10 person-months for operator algebra
and quantum_info, every one of them titled `feat(observables):`. Qiskit's
counterpart is `qiskit/quantum_info/` at 60 source files and 18,526 lines with 26
public names, of which the symplectic subpackage alone is 9 source files
(`pauli.py`, `pauli_list.py`, `sparse_pauli_op.py`, `base_pauli.py`,
`pauli_utils.py`, `clifford.py`, `clifford_circuits.py`, `random.py`,
`__init__.py`).

Two of those PRs are the problem:

- **W12-01** is "support general Pauli products on overlapping wires". Today
  `observables/__init__.py:47` raises `ValueError("Pauli tensor products require
  disjoint wires")`. Changing that raise changes the accepted inputs of a **Stable
  Core** name, so it needs the API-owner path, not a feature branch.
- **W12-04** is "add Pauli commuting groups", which would duplicate
  `runtime/parallel.py:82` for the internal type unless the two are unified first.

If the 42 PRs land in `observables/` as written, the repository ends with a
quantum_info-scale algebra on the branch that has one consumer, while the branch
that actually carries Hamiltonian data from `algorithms` to `deployment` to
`remote` remains on `HamiltonianTerm`. Deciding that now costs one document;
deciding it after W12-03 costs a rewrite of the container types.

### Affected user journeys

1. A user who follows the `observables` README to
   `fq.expectation(fq.X(0) @ fq.X(1))` and then tries `fq.X(0) @ fq.Y(0)` gets a
   `ValueError` about disjoint wires, with no pointer to a general Pauli type or to
   `qec.Pauli`, which already supports overlapping wires.
2. A user who computes a Hamiltonian in `algorithms`, groups it with
   `group_observables`, and then looks for the public `Observable` class finds two
   unrelated shapes and no documented conversion.
3. A contributor who needs a weighted Pauli product must choose between eight
   representations with no guidance; the choice they make determines whether their
   type reaches `deployment` and `remote`.
4. A maintainer planning W12 cannot tell whether to extend `Observable`, add a
   `Pauli`/`PauliList`/`SparsePauliOp` family beside it, or unify on
   `HamiltonianTerm` first.

## Evidence

| Claim | Evidence |
| --- | --- |
| `observables/` owns 10 of 34 Stable Core names | `docs/public_api_v1.json` `stable_exports`; definitions at `observables/__init__.py:25,88,182-206,212-277` |
| `lower_outputs` is not Stable Core | absent from `docs/public_api_v1.json`; defined at `observables/__init__.py:297` |
| `observables` is not a long-horizon domain | `contracts/long-horizon-architecture-v1.json` `domains`: 11 entries, none named `observables` |
| `observables` is not in the layer order | `architecture.toml` `[layers] order` has 9 layers, none named `observables` |
| `observables` has no boundary rule | `tools/check_architecture.py:524-593` tests 7 prefixes; `observables` is not one |
| It is imported by 15 files | 17 static import sites plus 3 dynamic lookups, enumerated in the first section |
| The algebra has one production consumer | `simulation/lindblad.py:12`, used at `lindblad.py:432` |
| The request contract has 16 static sites | AST-resolved `ImportFrom` of `flagquantum.observables` selecting `OutputRequest`, `lower_outputs`, or `counts` |
| Duplicate `_PauliTerm` with incompatible fields | `observables/__init__.py:19` (`factors: tuple[tuple[int, str], ...]`, `float`) versus `runtime/executors/statevector/split_real_imag.py:232` (`ops`, `Any`) |
| `algorithms.HamiltonianTerm` shape | `algorithms/core.py:86-103`; `pauli_term()` at `:342` |
| `qec.Pauli` is phase-free and symplectic | `qec/pauli.py:1-18,35-44`; exported at `qec/__init__.py:59` |
| `services.PauliSum` is a wire contract | `services/ground_state.py:33-40` (`schema`, `coefficient_unit`, `basis_convention`) |
| `noise` owns a Pauli basis | `noise/channels.py:35-47` |
| The overlapping-wire raise | `observables/__init__.py:47` |
| No bridge between the two worlds | zero imports of `observables` under `flagquantum/algorithms/` or `flagquantum/runtime/` |
| `ObservableGroup` is index-based | `runtime/parallel.py:13-16` (`term_indices`, `basis`) |
| Deployment binds both worlds | `deployment/cloud.py:154-156` (`hamiltonian: Hamiltonian`, `groups: tuple[ObservableGroup, ...]`) |
| W12 is 42 PRs, all `feat(observables):` | parity backlog W12 table; W12-01 and W12-04 rows |
| Qiskit counterpart size | `qiskit/quantum_info/` 60 source files / 18,526 lines / 26 public names; 9 symplectic source files |

## Alternatives considered

**A. Leave `observables/` as it is and build the algebra there.** This is what the
backlog assumes, and it is defensible: the package already owns the public
mathematical model, and the output-request half is small. Rejected as the default
because it leaves the two-worlds split intact. W12-03's `PauliList` and W12-05's
`SparsePauliOp` would have no defined relationship to `HamiltonianTerm`, and
`group_commuting` would be duplicated. If the owners choose this, the document
asks that they record the `HamiltonianTerm` relationship explicitly rather than by
omission, and that W12-01 be routed through the API-change path.

**B. Split the module first, then build the algebra.** Proposed below. The
output-request contract moves to Core, where `shared_contract_owner` says
cross-domain contracts live, and `observables/` becomes the algebra package that
W12's 42 PR titles already assume. The risk is churn: 16 static import sites move,
three of them dynamic and therefore invisible to a static rewrite, and one of them
is the root `__init__.py` public surface.

**C. Make `algorithms.HamiltonianTerm` canonical and build the algebra on it.**
Attractive because it is the type that actually flows to `runtime` and
`deployment`, and because it already accepts `torch.Tensor` coefficients, which the
differentiable framework needs and `observables._PauliTerm`'s `float` does not.
Rejected as the primary proposal for two reasons: `algorithms` is an application
domain, not an infrastructure one, and putting a Stable Core algebra there inverts
the dependency direction; and `observables.Observable` is already Stable Core, so
this alternative changes what an existing public name means. It is the strongest
alternative and the one the owners should weigh against B.

**D. Unify on `qec.Pauli`.** Rejected. It is the only general-purpose Pauli in the
repository and it already supports overlapping wires, but it is phase-free by
design, it carries no coefficient, and its consumer is a domain that has never
exported a stable contract.

**E. Add a new `quantum_info` package and leave `observables/` untouched.**
Rejected as the first step only, because it creates a ninth representation and a
fourth public algebra entry point while the other three remain. It may be the right
*end state* for the quantum_info-scale surface; it should not precede unification.

## Old and proposed behavior

This document changes no behavior. It proposes a split, a canonical type, and an
approval path.

| Surface | Today | Proposed |
| --- | --- | --- |
| Output-request contract (`OutputRequest`, `expectation`, `probabilities`, `samples`, `counts`, `lower_outputs`) | `observables/__init__.py`, Stable Core for five of the six | Moves to Core as a cross-domain execution-output contract; the six names stay importable from `flagquantum` and keep their stable status |
| Pauli observable algebra (`Observable`, `I`, `X`, `Y`, `Z`) | `observables/` | Stays in `observables/`, which becomes the algebra package |
| Overlapping-wire products | `ValueError` at `observables/__init__.py:47` | General Pauli multiplication; a Stable Core input change and therefore an API-change decision of its own |
| `HamiltonianTerm` and `pauli_term()` | `algorithms/core.py:86,342` | Either declared the canonical weighted-term type with a conversion to `Observable`, or declared an application-local type with a documented boundary. Silence is not an option |
| `ObservableGroup` and `group_observables` | `runtime/parallel.py:13,82`, index-based over `HamiltonianTerm`, qubit-wise only | One grouping implementation over the canonical type, with qubit-wise and full-commuting strategies selected explicitly |
| Duplicate `_PauliTerm` | Two private classes, same name, incompatible fields | One, or two renamed so that a grep is unambiguous |
| `observables/` as an architectural actor | Not a domain, not a layer, no boundary rule | Named in the long-horizon `domains` map and given an explicit dependency direction, or explicitly recorded as an API-surface package owned by Core |

## Source and behavioral compatibility impact

No behavior changes in this document. The implementation it proposes has these
impacts, in decreasing order of risk:

1. **Moving the output-request contract.** Five of the six names are Stable Core,
   so their import path from `flagquantum` must keep working; only their
   defining module moves. The 16 static import sites change, and the three dynamic
   lookups (`__init__.py:89`, `_api.py:121,261`) will not be caught by a static
   import rewrite, so they must be found by test rather than by grep.
2. **General Pauli products.** Removing the raise at `observables/__init__.py:47`
   widens accepted inputs. Code that relied on the `ValueError` to reject a mistake
   will now compute a result. That is a behavior change on a Stable Core name and
   needs the API-change path plus a test that pins the new product semantics,
   including the phase convention.
3. **Unifying grouping.** `runtime/parallel.py:82` is the only implementation and
   has a caller in `runtime/module.py:969` and an indirect one through
   `deployment/cloud.py:154`. A unified version must return the same groups for the
   same input or the change is visible in deployment packages.
4. **Renaming a private class.** `runtime/executors/statevector/split_real_imag.py:232`
   is private, so no public surface moves, but
   `split_real_imag_optimizer_conformance.py:17` imports from that module and must
   be checked.

## Migration example

Nothing here is executable yet. The shape of the intended end state, for review:

```python
# Today: one module, two responsibilities, ten public names.
from flagquantum.observables import OutputRequest        # Core-ish contract
from flagquantum.observables import Observable           # algebra

# Proposed: the cross-domain contract is owned where shared contracts live,
# and the algebra package keeps the names W12 already targets.
from flagquantum.core import OutputRequest               # canonical
from flagquantum.observables import Observable           # unchanged import

# Today: two private types with the same name and different fields.
class _PauliTerm:                                        # observables/__init__.py:19
    coefficient: float
    factors: tuple[tuple[int, str], ...]
class _PauliTerm:                                        # split_real_imag.py:232
    coefficient: Any
    ops: tuple[tuple[int, str], ...]

# Proposed: the algebra's term type is the only one named _PauliTerm, and the
# executor-local one is named for what it is.
class _PrecisionPauliTerm:                               # executor-local
    ...
```

## Versioning

Moving the output-request contract is a **patch** change for users: no name, no
signature, and no documented behavior changes, and `flagquantum.<name>` keeps
working. It is a **minor** change for anyone importing
`flagquantum.observables.OutputRequest` directly, and `docs/development/PUBLIC_API_PROTECTION.md`
controls whether that path needs a deprecated alias. The document recommends the
alias for one release with a named owner and removal version, because
`observables` is imported by `ecosystem/`, which is the boundary external code
touches.

General Pauli products are a **minor** change: accepted inputs widen, results do
not change for inputs that were already accepted, and the exception that disappears
was never a documented guarantee. The phase convention of the widened product is
new documented behavior and must be pinned by test.

Unifying the grouping implementation is **patch** scope if the returned groups are
identical, and **minor** if full-commuting groups become available, because that is
additive.

## Documentation and tooling impact

- `flagquantum/observables/README.md` must stop describing two responsibilities in
  one sentence, and must state the canonical weighted-term type and its relationship
  to `HamiltonianTerm`.
- `flagquantum/core/README.md` gains the output-request contract if the split is
  approved.
- `docs/reference/API.md` and `docs/public_api_v1.json` are unaffected in names but
  the module of record changes; `tools/public_api_snapshot.py` output must be
  regenerated, not hand-edited.
- `docs/architecture/decisions/` gains an ADR if the owners prefer B over A, because
  the choice is architectural rather than a naming preference.
- `capability-maturity.toml` has **no** observables entry today, which the gap
  analysis already records as a declared gap. Whichever option is approved, the
  operator-algebra work needs a capability entry before the first W12 PR, or the
  first W12 PR must add it.
- `tools/check_architecture.py` gains an `observables` rule only if option B or C is
  approved; under option A the omission should be recorded deliberately rather than
  left as an accident, which is the same standard the compiler proposal applies.

## Owner and approvals

- Owning domain: `core` for the output-request contract and the canonical algebra
  type; `integration` for the boundary record. `docs/api-changes/**` is a shared
  path, and `python tools/check_team_scope.py --team integration --files
  docs/api-changes/FQ-PAULI-QUANTUM-INFO-BOUNDARY-20260930.md` passes.
- Required approvals before implementation: **API owner** (two Stable Core names
  change module, and one changes accepted inputs), **core domain owner** (canonical
  weighted-term type), and **integration owner** (whether `observables` becomes a
  named dependency direction or stays an unlisted API surface).
- Sequencing: this document, then the W12-01 API decision, then the split, then the
  algebra work. W12-03 (`PauliList`) and W12-05 (`SparsePauliOp`) must not land
  before the canonical type is chosen, because both are containers over exactly the
  choice this document asks for.
- Implementation note for the owner: the question to answer first is option C
  versus option B, because it decides whether `Observable` is extended or
  deprecated. `HamiltonianTerm` accepts `torch.Tensor` coefficients and `Observable`
  does not, and this is a differentiable framework, so the answer has a real
  consequence beyond tidiness.
