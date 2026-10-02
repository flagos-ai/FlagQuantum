# Quit calling them wires: finish the migration across the whole package

One document per proposed change to the Stable Core public API. This one extends
[Complete the qubit terminology migration on the public surface](FQ-QUBIT-TERMINOLOGY-COMPLETION-20261002.md)
from a documented subset to the whole package, and replaces that document's
tier arithmetic with a ledger a gate can enforce.

## Decision and authorization

The user asked on 2026-09-13 for gradual retirement of wire terminology, restated
it on 2026-10-02 as a binding constraint on the PennyLane alignment program, and
on 2026-10-05 gave the scope decision this document records: **the user-facing
surface of FlagQuantum says `qubit`, never `wire`**, and the boundary is the
whole package rather than a curated list.

That is the explicit user authorization that
[`PUBLIC_API_PROTECTION.md`](../development/PUBLIC_API_PROTECTION.md) and
`AGENTS.md` rule 8 require before a protected signature may be renamed. It covers
every parameter on the package's public function surface, listed in the ledger in
`contracts/qubit-vocabulary-contract.toml`. It does not cover serialized payload
keys, environment variable names, or vendor adapter payload keys; those are
decided in *Compatibility*.

The relationship to the two earlier documents is a **supersession of scope, not of
decision**:

| Document | What it decided | What this document changes |
|---|---|---|
| [Qubit terminology migration](FQ-QUBIT-NAMING-20260913.md) | retire `wire` gradually; `Circuit(n_qubits=)`, `probabilities/samples/counts(qubits=)`, `RuntimePolicy(observable_qubits=)`, `CloudBackendProfile(n_qubits=)` land | nothing; this document does not revisit those names |
| [Complete the qubit terminology migration on the public surface](FQ-QUBIT-TERMINOLOGY-COMPLETION-20261002.md) | the T1–T4 tiers, a 22-row inventory, the `Omitted` sentinel rule, the acceptance clause "the gate must fail on a fixture that introduces a `wire` parameter" | the tiers are **replaced** by one whole-package boundary, and the inventory is **replaced** by a scanned ledger; the sentinel rule and the acceptance clause are carried forward unchanged |

## The tier split could not be enforced, and that is why it is gone

The previous document split the surface into four tiers measured as
17 / 3 / 250 / 5 signatures. Two of those numbers do not survive measurement.

**The tiers were computed with a private-module filter that drops public
namespaces.** The filter skipped any path component beginning with `_`, which
also skipped `flagquantum/observables/__init__.py` — a package's `__init__.py` *is*
its public namespace, not a private module. That one file carries eight
parameters, seven of them shipped aliases, and it is the module that defines
`fq.X`, `fq.Y`, and `fq.Z`. The same error applies to `core/__init__.py` and
`runtime/__init__.py`. This is the whole of the discrepancy between the previous
document's T3 = 250 and the ledger's measured total.

**The "documented user surface" tier is decided by prefix matching, not by
reading.** Selecting modules that appear in `docs/guides/**` or
`docs/reference/API.md` matches `flagquantum.compiler`, `flagquantum.runtime`, and
`flagquantum.simulation` as *prefixes*, which pulls their entire subtrees into the
"documented" tier. Measured that way the tier is 87, or 138 under a wider match.
A boundary whose size depends on how the query is written is not a boundary, and
no gate can enforce it. **The whole-package boundary needs no query at all.**

## Measured scope

Measured on the WQ-1 base commit by `tools/census_wire_vocabulary.py`:

| Surface | Count | Disposition |
|---|---:|---|
| Parameters on the public function surface, still spelled `wire` | **341** | renamed by this program |
| Parameters that are already deprecated aliases of a `qubit` sibling | **11** | deleted at 0.4.0 |
| Parameters in private code | 397 | out of scope; measured so the exclusion stays honest |
| Public attribute and property names containing `wire` | 102 | owned gap, see *Open Questions* 1 |
| String literals containing `wire` | 1291 | reported only, see *Open Questions* 2 |

Four corrections were needed to reach a number that reproduces, and each is
recorded where it belongs rather than absorbed:

1. **Variadic parameters were missing.** Reading only
   `posonlyargs + args + kwonlyargs` misses `Circuit.any(*wires, unitary=...)` and
   its deprecated `unitary` spelling. `circuit.py` reads 2 instead of 3.
2. **`__init__.py` was treated as a private module.** Described above.
3. **Methods of private classes were reported as public.** An `ast.walk` over the
   module visits a method of `_GroverOperator` exactly as it visits a module-level
   function. Sixteen sites across `algorithms/` and `simulation/` were counted that
   are not part of any public surface, because a name nested inside private code is
   private whatever it is called. The shipped census descends only into public code.
4. **Every public constructor was invisible.** "Starts with an underscore" is not
   the same as "private": a dunder is Python's public protocol. Applying the
   underscore test to `__init__` moved *every* class constructor out of the public
   surface and into the private bucket — including `fq.Circuit(n_wires=)`, the single
   most user-visible call in the package, which is also one of the eleven shipped
   aliases. Twelve canonical sites and three aliases were hidden this way, and the
   private bucket was inflated by the same amount. The census now treats a
   `__x__` name as public and every other underscore-led name as private.

## Problem and affected user journey

The migration stopped at the `Circuit`-adjacent outputs. One screen of ordinary
user code therefore requires two vocabularies:

```python
import flagquantum as fq

q = fq.Circuit(2).h(0).cx(0, 1)          # n_qubits: fine
fq.Circuit(1).rx(qubit=0, theta=0.3)     # gates accept only "qubit"
fq.Z(qubit=0)                            # fine since the partial landing
result = fq.run(q, outputs=fq.probabilities(qubits=(0,)))
result.n_wires                           # the old word is back, on the result
fq.from_engine_qir(payload)              # n_wires= again
```

The second half of that screen is what the previous document left out. Its
inventory was scoped to names reachable from `fq.*`, and reachability is not the
same as visibility: `fq.run` returns objects whose attributes are part of the
user's screen, and `fq.from_engine_qir` is reached through `fq.*` while its
parameter sits in `flagquantum/core/`.

## Naming

One rule, applied by root substitution, so the naming has a single source of
truth and the ledger never restates it:

| Old | New | Note |
|---|---|---|
| `wire` | `qubit` | single index |
| `wires` | `qubits` | index sequence |
| `*wires` | `*qubits` | variadic; the `*` is preserved |
| `n_wires` | `n_qubits` | width |
| `n_counting_wires`, `n_evaluation_wires`, `n_support_wires` | `n_counting_qubits`, … | compound names substitute the root |
| `wire_order`, `show_all_wires`, `left_wire` | `qubit_order`, `show_all_qubits`, `left_qubit` | drawing |
| any other compound containing `wire` | substitute the root | `observable_wires`, `sharded_wires`, `preferred_local_wires`, `terminal_wires`, … |

**Prohibited alternatives**: `qubit_indices`, `qubit_ids`, `qubit_labels`,
`logical_qubits`. `Circuit` already says `qubits`, `compose` already says
`qubits`; introducing a second spelling for the same concept would turn one
rename into two vocabularies, which is exactly what `AGENTS.md` rule 9 asks a
stable name not to do. The gate fails if any of these appears as a parameter.

## Alias policy

**Aliases only where `AGENTS.md` rule 8 protects the name.** Every other site is
renamed in place.

| Tier | Meaning | Alias |
|---|---|---|
| T1 | reachable from `fq.*` | yes — `new=`, with `old=` deprecated through `warn_qubit_alias` |
| T2 | reachable from a sub-namespace of `fq.*` | yes, treated as T1 |
| T3 | public modules imported by path, not by `fq.*` | no |
| T4 | public modules that need an optional extra to import | no |

The measured fact behind the split: `fq.compiler`, `fq.algorithms`,
`fq.simulation`, `fq.noise`, `fq.qec`, `fq.core`, `fq.deployment`, and
`fq.services` are **not** attributes of `fq` — `fq.deployment` raises
`AttributeError` — so their parameters are not Stable Core and carry no
compatibility obligation. An alias for them would preserve two vocabularies
forever on behalf of a caller who cannot legitimately exist, which `AGENTS.md`
engineering decision principle 1 forbids.

The eleven already-shipped aliases stay until their declared removal version and
are deleted by the slice that owns their file, because their canonical spelling is
already in place. Their removal is the completion marker for that slice.

## Compatibility

- **No Stable Core export changes.** `fq.__all__` keeps its current membership and
  `docs/public_api_v1.json` keeps its names; this program renames *parameters*.
  `flagquantum/__init__.py` keeps its line budget, which is why the work can be
  sliced by subpackage without touching the root module.
- **`IR_VERSION` stays `"1.0"`.** The serialized payload keys `CircuitIR.n_wires`,
  `Instruction.wires`, `MeasurementNode.wires`, and `ObservableNode.wires` are an
  exact-match version pin, and renaming a key is a schema change, not a naming
  change. They are listed in the contract's exclusion table so the gate passes them
  deliberately rather than by accident.
- **`contracts/public-api-v0.2-baseline.json` does record constructor parameter
  names**, including six occurrences of `n_wires`. The slice that renames those
  constructors updates the baseline in the same change.
- **Environment variable names are excluded.** `FQ_STATEVECTOR_PERSISTENT_WIRE_LAYOUT`
  is a published deployment switch; renaming it would break deployments that set
  it, for no naming benefit.
- **Vendor adapter payload keys are excluded.** The adapter translates at the
  boundary, per `AGENTS.md` rule 5; an external schema does not adopt our
  vocabulary.

## Migration path

`contracts/qubit-vocabulary-contract.toml` holds the baseline and seven slices.
The gate enforces **equality**, not containment: the live scan must be exactly the
baseline minus the retired identifiers. That single rule produces all three
desired failures — a new `wire` parameter fails, a slice that claims retirement it
did not perform fails, and a baseline line deleted to manufacture progress fails.
Progress is reported per slice on every run, so "how much is left" is a command
rather than an estimate.

| Slice | Scope | Baseline sites | Depends on |
|---|---|---:|---|
| WQ-1 | contract, census, gate, CI and pre-push wiring; no signature changes | 0 | — |
| WQ-2 | `circuit.py`, `core/`, `qec/`, `ecosystem/`, `testing/`, `drawer/`, `observables/` | 16 | WQ-1 |
| WQ-3 | `runtime/measurements.py`, `runtime/policy.py`, `runtime/planner/`, `runtime/dynamic/` | 14 | WQ-1 |
| WQ-4 | `runtime/executors/` | 49 | WQ-3 |
| WQ-5 | `algorithms/` | 52 | WQ-1 |
| WQ-6 | simulation CPU cores | 77 | WQ-4 |
| WQ-7 | simulation accelerator and tensor cores | 78 | WQ-6 |
| WQ-8 | `compiler/`, `benchmarking/`, `kernels/`, `noise/`, `remote/`, `deployment/` | 55 | WQ-5, WQ-7 |

After every slice lands the baseline is empty, and the gate's only remaining job
is to reject the first new `wire` parameter.

## Acceptance Tests

1. `python tools/check_qubit_vocabulary.py` exits 0 on the baseline commit, and
   reports 0 retired of 341.
2. The gate fails when a `wire`-named parameter is added to a module in the
   ledger's scope, demonstrated by running it against a fixture package that
   declares one.
3. The gate fails when a live alias forwards to a name the contract does not
   record.
4. The gate fails when a retirement entry still exists in the live scan.
5. Removing a baseline line without adding the matching retirement entry fails.
6. The gate is wired into the `quality` job as a **step**, not a job, and into
   `tools/pre_push.py`; `tools/validate_required_checks.py` still passes, because
   the required-check roster is unchanged.
7. No Stable Core export, default, or serialized key changes, verified by the
   existing contract and public-API gates.

## Open Questions

1. **Public attribute names are not yet ledgered.** 102 public attribute and
   property names contain `wire` — `MeasurementResult.wires`,
   `CircuitAnalysis.n_wires`, `ExecutionPlan.shardable_wires`, and so on. They are
   on the user's screen, so they are in scope for the program, and they are not in
   this ledger. The contract records the count and names WQ-2 as the owner: **the
   gate must ledger this surface before the owning slice lands**, and the ledger
   must exclude the serialized keys by name when it does. They are left out here
   rather than guessed at, because a per-item judgement made now would be a
   second, unreviewed copy of the exclusion table.
2. **Message strings are reported, not ledgered.** 1291 string literals contain
   `wire`. A literal scan cannot separate a serialized key from a refusal
   sentence, so a ledger built on that test would fail on legitimate rewording and
   pass on a serialized key that happened to be reformatted. The census reports the
   number so the surface is visible; the contract records why it is not gated.
3. **Whether the removal version should be 0.4.0 for the newly created T1 aliases.**
   `warn_qubit_alias` already publishes 0.4.0 for the eleven existing aliases. This
   document assumes new aliases inherit that version rather than opening a second
   removal window, which keeps one deprecation schedule instead of two.

Status: proposed. Implementation is authorized by the user for parameter names on
the Python surface. Nothing in this document depends on *Open Questions* 1-3 being
answered before WQ-1 lands; WQ-1 changes no signature.
