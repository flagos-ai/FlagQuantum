# Complete the qubit terminology migration on the public surface

One document per proposed change to the Stable Core public API. This one
finishes the migration that [Qubit terminology migration](FQ-QUBIT-NAMING-20260913.md)
started and explicitly scoped out.

## Decision and authorization

The user asked on 2026-09-13 for gradual retirement of wire terminology, and on
2026-10-02 restated it as a binding constraint on the PennyLane alignment
program: **the FlagQuantum user-facing surface must say `qubit`, never `wire`.**

That is the explicit user authorization that
[`PUBLIC_API_PROTECTION.md`](../development/PUBLIC_API_PROTECTION.md) and
`AGENTS.md` rule 8 require before a protected signature may be renamed. It
covers the surfaces listed in *Rename inventory* below. It does not cover the
serialized IR payload keys, which are decided separately in *Compatibility*.

`FQ-QUBIT-NAMING-20260913.md` stated its own boundary (lines 42-45):

> This proposal does not rename IR fields, backend-native adapter payload keys,
> plan identity inputs, or every internal occurrence.

This proposal is the named follow-on for that sentence. It is deliberately not a
second source of truth about what `#23` already did; it addresses only what `#23`
left out.

### What this proposal is authorized to change, and what it is not

`AGENTS.md` rule 8 protects the **Stable Core**: exports reachable from
`import flagquantum as fq`. The words `compiler`, `algorithms`, `simulation`,
`noise`, `qec`, `core`, `deployment`, and `services` are not attributes of `fq`
(measured: `fq.compiler` raises `AttributeError`), so their public parameters are
**not** Stable Core and need no proposal. Splitting the surface by that boundary
is the difference between a 17-signature protected change and a 275-signature
mechanical one:

| Tier | Meaning | Signatures with a `wire` parameter | Needs this proposal |
|---|---|---:|---|
| T1 | reachable from `fq.*` | **17** | yes |
| T2 | `fq.twin` / `fq.experimental` (reachable, but sub-namespaces) | 3 | yes, treated as T1 |
| T3 | public **modules** imported by path, not by `fq.*` | 250 | no — ordinary rename |
| T4 | module-level public functions in modules that fail to import without an optional extra (`matplotlib`, `triton`) | 5 | no — ordinary rename, but must be verified with the extra installed |

Only the T1/T2 rows in the table below carry a compatibility obligation. The T3
and T4 rows are renamed in place, and the gate in *Acceptance Tests* covers them
so the vocabulary does not drift back.

User-visible **message strings** are likewise not a frozen schema; rewording them
is not an API change and needs no proposal. They are in scope here only because
the user required the whole user-visible surface, and they are enumerated so the
change is reviewable rather than incidental.

## Problem and affected user journey

The migration stopped at the boundary of the `Circuit`-adjacent count outputs and
`RuntimePolicy`. The result is that a single screen of ordinary user code requires
two vocabularies:

```python
import flagquantum as fq

q = fq.Circuit(2).h(0).cx(0, 1)          # n_qubits: fine
fq.Circuit(1).rx(qubit=0, theta=0.3)     # gates accept only "qubit"
fq.Z(qubit=0)                            # TypeError: unexpected keyword argument 'qubit'
fq.CircuitIR(n_qubits=2, instructions=())
                                         # TypeError: unexpected keyword argument 'n_qubits'
fq.OutputRequest("counts", qubits=(0,))  # TypeError: unexpected keyword argument 'qubits'
```

Measured on `45a85cc3`:

```bash
cd repos/fq-pl && PYTHONPATH="…:$PWD" python - <<'EOF'
import inspect, flagquantum as fq
for name in sorted(fq.__all__):
    obj = getattr(fq, name)
    cands = [obj] + ([v for m, v in inspect.getmembers(obj, inspect.isfunction)
                      if not m.startswith('_')] if inspect.isclass(obj) else [])
    for fn in cands:
        try: sig = inspect.signature(fn)
        except Exception: continue
        if [p for p in sig.parameters if 'wire' in p.lower()]:
            print(f'{name}: {sig}')
EOF
```

`Circuit` gates already accept only `qubit`:

```python
fq.Circuit(1).rx(wire=0, theta=0.3)
# TypeError: rx requires qubit arguments: qubit
```

while the observable constructors and every IR/result type accept only `wire`.
The inconsistency is the user journey: the vocabulary a user learns from the gate
methods is contradicted by the types they must construct to read results.

The `wire` surface is not small. Measured counts:

| Surface | Count | Command |
|---|---:|---|
| Public signatures in `fq.__all__` | **25** | script above |
| Public (non-`_`) function parameters under `flagquantum/` | **283** | AST scan, §Evidence |
| private helpers with the same parameter | 313 | AST scan |
| `wires=` keyword calls under `flagquantum/` | 550 | `grep -rn 'wires=' flagquantum/ --include=*.py \| wc -l` |
| string literals containing `wire` under `flagquantum/` | 495 | `grep -rn '"[^"]*wire[^"]*"' flagquantum/ --include=*.py \| wc -l` |
| `wires=` in `docs/` | 116 | `grep -rn 'wires=' docs/ --include=*.md \| wc -l` |
| `wires=` in `examples/` | 11 | `grep -rn 'wires=' examples/ --include=*.py \| wc -l` |
| `wires=` in `tests/` | 964 | `grep -rn 'wires=' tests/ --include=*.py \| wc -l` |

The 283 is not all public API. Distribution by owning subpackage shows that a
large share is benchmark harness entry points (`run_case(n_wires)`,
`build_workload(n_wires)` in `flagquantum/benchmarking/`), which are not `fq.*`
user surface:

```text
simulation 104 | algorithms 57 | runtime 54 | benchmarking 24 | compiler 8
observables 8  | kernels 6     | noise 6    | deployment 4     | remote 3
circuit.py 2   | ecosystem 2   | testing 2  | core 1 | drawer 1 | qec 1
```

`flagquantum.algorithms` **is** user surface: it has a maintained guide at
[`docs/guides/ALGORITHMS.md`](../guides/ALGORITHMS.md), which states at line 61
that these units carry no root-level `fq.` name and are imported as
`from flagquantum.algorithms.<unit> import …`. Its public entry points take
`n_wires` today:

```python
from flagquantum.algorithms import transverse_field_ising   # (n_wires, *, coupling, field, periodic)
from flagquantum.algorithms.primitives.qft import qft       # (n_wires, *, inverse)
from flagquantum.algorithms.grover import grover_circuit    # (n_wires, ...)
```

`fq.twin` is also reachable from `import flagquantum as fq` and exposes one
`wire` parameter — `TwinPrediction(snapshot_identity, circuit_identity,
n_wires, …)`. `fq.experimental` exposes none.

## Evidence

All counts above are reproducible with the commands printed beside them. The
per-surface signature list is the output of the `inspect.signature` script under
*Problem*. Two further measurements fix the design:

**The rename is not a cosmetic edit for anything already recorded in the frozen
baseline.** `contracts/public-api-v0.2-baseline.json` freezes the constructors of
the types being renamed:

```bash
cd repos/fq-pl
grep -n '"n_wires"' contracts/public-api-v0.2-baseline.json   # 6 occurrences
python3 - <<'EOF'
import json
d = json.load(open('contracts/public-api-v0.2-baseline.json'))
for n in ('CircuitIR', 'Instruction', 'MeasurementResult'):
    print(n, [p['name'] for p in d['exports'][n]['constructor']['parameters']])
EOF
# CircuitIR ['n_wires', 'instructions', 'version', 'dtype', 'shape', 'observables', 'measurements', 'metadata']
# Instruction ['name', 'wires', 'params', 'matrix', 'metadata']
# MeasurementResult ['kind', 'wires', 'value', 'shots', 'metadata', 'statistics']
```

`CircuitIR.n_wires` and `Instruction.wires` are positional-or-keyword, so a rename
is observable both by keyword callers and by the position-order record.

**`IR_VERSION = "1.0"` is an exact-match pin, and this repository has already
ruled that the version must not be bumped to make a field readable.**
[`FQ-CIRCUIT-EXPRESSIVENESS-CONTRACT-20260930.md`](FQ-CIRCUIT-EXPRESSIVENESS-CONTRACT-20260930.md)
lines 154-165 and its Prohibited Practices item 5 record:

> **Bumping `IR_VERSION` to make a new field readable.** The version is an
> exact-match pin with persisted readers; a bump requires a stated migration path
> for `execution_plan_contract` payloads and `training_state` checkpoints, or it
> silently invalidates them.

and its Compatibility section gives the two admissible shapes for changing what
`"1.0"` means: keep `"1.0"` readable for programs without the new fields, or
accept a range and migrate on read. That ruling constrains *Compatibility* below.

## Rename inventory

Disposition per surface. "Alias" means the old keyword keeps working under
`warn_qubit_alias` with the removal version `FQ-QUBIT-NAMING-20260913.md` already
published (deprecation 0.3.x, removal 0.4.0).

### Canonical name becomes `qubit`, old name retained as a deprecated alias

| # | Public object | Today | After |
|---|---|---|---|
| 1 | `fq.CircuitIR.__init__` | `(n_wires, instructions, …)` | `(n_qubits, instructions, …)`, `n_wires=` alias |
| 2 | `fq.CircuitIR.n_wires` | writable field | `n_qubits` property; `n_wires` deprecated property |
| 3 | `fq.Instruction.__init__` | `(name, wires, params, matrix, metadata)` | `qubits=`; `wires=` alias |
| 4 | `fq.Instruction.wires` | field | `qubits` field; `wires` deprecated property |
| 5 | `fq.MeasurementResult.__init__` | `(kind, wires, value, …)` | `qubits=`; `wires=` alias |
| 6 | `fq.MeasurementResult.wires` | field | `qubits` field; `wires` deprecated property |
| 7 | `fq.OutputRequest.__init__` | `(kind, wires=(), …)` | `qubits=()`; `wires=` alias |
| 8 | `fq.OutputRequest.wires` | field | `qubits` field; `wires` deprecated property |
| 9 | `fq.ExecutionPlan.__init__` | `(…, shardable_wires, …)` | `shardable_qubits`; `shardable_wires=` alias |
| 10 | `fq.ExecutionPlan.shardable_wires` | field | `shardable_qubits` field + deprecated property |
| 11 | `fq.X` / `fq.Y` / `fq.Z` | `(wire)` | `(qubit)`; `wire=` alias |
| 12 | `fq.I` | `(wire=None)` | `(qubit=None)`; `wire=` alias |
| 13 | `fq.Circuit.gate` | `(name, wires, …)` | `(name, qubits, …)`; `wires=` alias |
| 14 | `fq.Circuit.expectation_z` | `(wires=None)` | `(qubits=None)`; `wires=` alias |
| 15 | `fq.Circuit.n_wires` | property + setter | `n_qubits` is canonical; `n_wires` deprecated property (setter kept, warns) |
| 16 | `fq.Circuit(n_wires=…)` | live keyword | deprecated alias (already warns today) |
| 17 | `fq.RuntimePolicy(observable_wires=…)`, `.observable_wires` | deprecated already | **delete** at 0.4.0; no change in this proposal |
| 18 | `fq.{counts,probabilities,samples}(wires=…)` | deprecated already | **delete** at 0.4.0; no change in this proposal |

Rows 17 and 18 are listed only so the inventory is complete. They are implemented;
this proposal does not touch them.

### Renamed with no compatibility obligation (positional-only or keyword-less surface)

| # | Public object | Today | After |
|---|---|---|---|
| 19 | `fq.Circuit.any` / `fq.Circuit.unitary` | `(*wires, unitary, name)` | `(*qubits, unitary, name)` — variadic, not keyword-callable, so the label exists only in `inspect.signature` and help output |
| 20 | `flagquantum.algorithms.<unit>` public entry points | `n_wires`, `counting_wires`, `evaluation_wires`, `support_wires` | `n_qubits`, `counting_qubits`, `evaluation_qubits`, `support_qubits`; keep `*_map` where the value is a mapping |
| 21 | `fq.twin.TwinPrediction.__init__` | `(snapshot_identity, circuit_identity, n_wires, …)` | `n_qubits`; `n_wires=` alias, because the dataclass has a published `schema` string |
| 22 | `fq.simulation`, `fq.runtime`, `fq.compiler`, `fq.noise`, `fq.kernels`, `fq.drawer`, `fq.qec`, `fq.deployment`, `fq.remote` public function parameters | `wires` / `n_wires` | `qubits` / `n_qubits` |

Rows 20 and 22 have no alias: these are pre-freeze surfaces, and `Public API
Protection` applies to Stable Core only. Rename in place.

### User-visible messages

Every `raise` / `warnings.warn` string that a user can reach, not every string
that contains the word. Measured examples that must change:

| file:line | today |
|---|---|
| `flagquantum/circuit.py:56,60` | `f"{owner} wire must be an integer, got {value!r}"` |
| `flagquantum/circuit.py:570` | `"A wire can appear in only one of x, y, or z."` |
| `flagquantum/core/ir.py:42,44,46` | `f"{owner} requires at least one wire"` / `wires must be non-negative` / `cannot repeat a wire` |
| `flagquantum/core/ir.py:439` | `f"n_wires must be positive, got {n_wires}"` |
| `flagquantum/observables/__init__.py:50` | `"Pauli tensor products require disjoint wires"` |
| `flagquantum/observables/__init__.py:135,139,141,174` | `f"{owner} wire must be an integer…"` / `f"{owner} wire must be a non-negative integer"` / `"output wires must be unique"` |
| `flagquantum/core/_compilation_evidence.py:159,161` | `f"{owner} must contain two distinct wires"` / `f"{owner} contains a wire outside the coupling map"` |
| `flagquantum/circuit.py:356-359` | `f"Gate {name!r} references wire(s) {outside} outside circuit range [0, {self.n_wires - 1}]."` |

An AST scan is the enumeration mechanism (`ast.Raise` and `*warn*` call nodes),
because grepping string literals sweeps in backend-adapter payload keys that must
stay. The scan currently finds 495 wire-bearing string literals under
`flagquantum/`; the reachable subset is what changes.

### Boundary exceptions — wire stays, deliberately

| Surface | Why it stays | Authority |
|---|---|---|
| Serialized IR payload keys `n_wires`, `wires` | Changing them changes what `IR_VERSION "1.0"` means; see *Compatibility* | `FQ-CIRCUIT-EXPRESSIVENESS-CONTRACT-20260930.md` Prohibited Practices item 5 |
| `wire_map` / `qubit_map` **as a mapping concept** | The name denotes a map from one index space to another; `OPEN_SOURCE_API_QUALITY_PLAN.md:158` reserves wire terminology for IR/mapping/internal indices. The **public spelling is `qubit_map`**; the internal type/helper may keep `wire` | `OPEN_SOURCE_API_QUALITY_PLAN.md:158` |
| Third-party payload keys inside `flagquantum/ecosystem/{qiskit,cirq,pennylane,braket}/` and `remote/` | These mirror the external tool's own vocabulary at its owning boundary; renaming them would misreport the vendor schema | `AGENTS.md` engineering principle 4 |
| `flagquantum/benchmarking/**` internal benchmark parameters | Not `fq.*` user surface. They appear in the **exclusion list of the gate**, explicitly, rather than being left to memory | this proposal |
| Historical records: `contracts/public-api-v0.2-baseline.json`, past proposal documents, generated evidence JSON | These are immutable audit records; `FQ-QUBIT-NAMING-20260913.md` line 40 already states historical fixtures are preserved rather than regenerated to hide the transition | `AGENTS.md` rule 8 |
| `FQ_STATEVECTOR_PERSISTENT_WIRE_LAYOUT` **as an environment variable** | An environment variable is an operational interface, not a Python name. Renaming it silently changes the behaviour of every existing deployment and of `tests/distributed/statevector_reverse_executor.py:25,123`, `benchmarks/statevector_training_scaling.py:270`, `benchmarks/flagquantum_statevector_value_and_grad.py:374`, and `examples/distributed_statevector_topologies/README.md:48` | this proposal — see below |

**The environment variable needs a read-both-names window, not a rename.**
`flagquantum/runtime/executors/statevector/reverse_support.py:42` reads
`FQ_STATEVECTOR_PERSISTENT_WIRE_LAYOUT` and `forward_sweep.py:111-115` warns about
it. A deployment that sets the old name must keep working, so the reader accepts
`FQ_STATEVECTOR_PERSISTENT_QUBIT_LAYOUT` when set, falls back to the old name, and
warns only when the old name alone is used. Setting both to different values is an
error rather than a precedence rule, matching the keyword-alias rule. This is the
same install-time compatibility question as `RuntimePolicy`'s legacy payload
reader, and it is the reason the environment variable is listed here rather than
buried in the rename rows.

### Serialized artifacts this proposal does **not** rename

Enumerated so the exclusion is checkable rather than assumed. Each is a payload
key family the migration doc already declares unchanged:

| Family | Keys | Read/written by |
|---|---|---|
| Circuit IR | `n_wires`, `wires` | `core/ir.py` `to_dict`/`from_dict`, the three node writers/readers |
| Execution plan | `n_wires`, `shardable_wires` (embedded full `CircuitIR` program) | `runtime/execution_plan_contract.py:635` |
| Training state | the plan payload above | `runtime/training_state.py:311` (rejects a differing version at `:363`) |
| Measurement contract | `wires` | `MeasurementContract`, declared **required** in `docs/runtime_contracts.schema.json:90` |
| Twin prediction | `n_wires` | `TwinPrediction.to_dict()` |
| Compilation evidence | `n_wires`, `physical_wires`, `logical_wires`, `logical_wire_count`, `coupling_n_wires` | `core/_compilation_evidence.py` |

`RuntimePolicy` is the one family that has already flipped on the **writer** side
(it emits `observable_qubits` under schema version 2.0) while its reader still
accepts the legacy `observable_wires` key. That is the shape Candidate A applies
to the rest only if *Open Questions* 1 is answered yes; today it is the shape for
the environment variable above.

`wire_map` deserves a note. `N1-1` in the PennyLane alignment plan proposes
`compose(other, *, wires=None, wire_map=None)`. Under this proposal the public
signature is `compose(other, *, qubits=None, qubit_map=None)`. The **mapping
concept is not renamed away** — only its public spelling. Keeping the internal
token `wire` in a private helper that computes a relabelling is consistent with
the reservation above; exposing it in a public parameter name is not.

## Decision Candidates

### Candidate A — rename the Python surface, keep the serialized keys (recommended)

Rename every row of *Rename inventory*. Leave the JSON keys `n_wires` and `wires`
exactly as they are, so `IR_VERSION "1.0"` keeps meaning what it means today, and
have `to_dict()`/`from_dict()` convert at the boundary:

```python
# flagquantum/core/ir.py — the two vocabularies meet in exactly two functions
def to_dict(self) -> dict[str, Any]:
    return {"n_wires": self.n_qubits, ...}       # payload vocabulary is "1.0"

@classmethod
def from_dict(cls, payload):
    return cls(n_qubits=payload["n_wires"], ...)
```

- Cost: the word `wire` remains in `CircuitIR.to_json()` output.
- Benefit: zero IR version change, zero checkpoint invalidation, no migration
  path needed for `execution_plan_contract` payloads or `training_state`
  checkpoints, and the user-facing Python surface is fully `qubit`.
- It also keeps `N1-11`'s required answer #3 true: `IR_VERSION` stays `"1.0"`.

### Candidate B — rename the serialized keys under a new IR version

Rename the payload keys and accept both `"1.0"` and `"2.0"` on read, migrating
on read.

- Cost: `IR_VERSION` becomes a range, which contradicts the exact-match pin at
  `core/ir.py:440-445` and the "keep `1.0` readable" ruling; every reader
  (`runtime/execution_plan_contract.py:635`, `runtime/training_state.py:311,363`)
  and every interop contract's `ir_version` check (`tools/check_*_interop_contract.py`,
  5 files) must handle the range; `training_state.py:363` currently **rejects** a
  differing version, so old checkpoints fail unless a migration is written.
- Benefit: `to_json()` also says `n_qubits`.
- This is a program-artifact compatibility decision, not a naming cleanup. If the
  product wants it, it belongs in its own proposal with its own migration path.

### Candidate C — leave the IR types alone, rename only the observable constructors

Smallest change: `X`/`Y`/`Z`/`I` accept `qubit=`, nothing else moves.

- Rejected. It removes the most visible inconsistency (`fq.Z(qubit=0)`), but a
  user still cannot construct `Instruction(qubits=…)` or read
  `MeasurementResult.qubits`, so the migration stays half-finished exactly as
  `#23` left it, and the next round re-opens the same question.

**Recommendation: A.** It is the smallest implementation that satisfies the
verified requirement (engineering principle 2), it needs no IR version change, and
it converts the naming decision into a boundary conversion in two functions
instead of a persisted-format migration.

## Prohibited Practices

1. **Adding a `wire`-named parameter to any new public signature.** This is the
   specific regression the gate exists to stop; `#23` renamed four surfaces and
   five later-added public types reintroduced `wires`.
2. **Renaming a serialized key without a version decision.** The key and the
   version are one decision; see Candidate B.
3. **Regenerating `contracts/public-api-v0.2-baseline.json` or
   `docs/public_api_v1.json` to make the rename pass.** `tools/public_api_snapshot.py`
   already refuses `--write`; neither artifact may be regenerated merely to make
   the check green.
4. **Silently accepting both `qubit=` and `wire=` on a new surface.** Supplying
   both is a `TypeError`, matching `#23`'s `OMITTED` sentinel rule; do not guess
   precedence.
5. **Renaming a vendor adapter's payload key** because it contains the word
   `wire`. The adapter translates at its boundary; it does not re-vocabulary the
   external schema.
6. **Doing this rename inside a PennyLane-alignment feature PR.** It is a
   protected-API change; it lands as its own authorized change before the feature
   PRs that depend on the new spellings.
7. **Deleting the deprecated aliases before 0.4.0.** The removal version is
   published; `#23`'s window applies.

## Compatibility

- **Python callers.** Every renamed row keeps the old keyword working with a
  `DeprecationWarning` naming the replacement and the removal release, through
  `warn_qubit_alias` in `flagquantum/core/_qubit_aliases.py`. Positional calls
  are unaffected because every rename preserves argument position.
- **`RuntimePolicy` payloads.** Unchanged by this proposal; the versioned
  `flagquantum.runtime_policy` v2.0 writer and the historical-payload reader from
  `#23` stay as they are.
- **IR payloads and training checkpoints.** Unchanged under Candidate A:
  `IR_VERSION` remains `"1.0"` and the JSON keys are byte-identical. This is the
  whole reason for choosing A.
- **Frozen baseline.** `contracts/public-api-v0.2-baseline.json` remains an
  immutable audit record of the historical surface. The affected
  `*-v1-candidate.json` files and `docs/public_api_v1.json` are synchronized as
  part of the implementation, per `#23`'s precedent for an explicitly requested
  migration.
- **Backward-reading only.** Older package versions do not understand a renamed
  Python keyword regardless of serialization, as `QUBIT_NAMING_MIGRATION.md`
  already records for the earlier slice.

## Acceptance Tests

- `fq.Z(qubit=0) == fq.Z(0)` and `fq.Z(wire=0)` warns with `0.4.0`; supplying both
  raises `TypeError`.
- `fq.CircuitIR(n_qubits=2, instructions=())` constructs; `n_wires=2` warns; both
  together raise.
- `CircuitIR.to_json()` is byte-identical to the pre-change output for the same
  circuit, and `from_dict` round-trips it. This is the test that pins Candidate A.
- `fq.Instruction(name="h", qubits=(0,))` and
  `fq.OutputRequest(kind="counts", qubits=(0,))` construct; the `wires=` spellings
  warn.
- `fq.ExecutionPlan(..., shardable_qubits=(0,))` constructs.
- Numerical parity: probability, expectation, counts, and sample results are
  bit-equal between the new and deprecated spellings for the same seed.
- Batched hybrid forward values and classical/quantum gradients are unchanged.
- `tools/public_api_snapshot.py` passes after the candidate contracts and
  `docs/public_api_v1.json` are synchronized.
- The new terminology gate fails on a fixture that reintroduces a `wire`-named
  public parameter, and the fixture is asserted in the gate's own test — a gate
  that has never rejected anything is not evidence.
- `hasattr(fq.Circuit, 'measure')` and every other *absent* capability in the
  PennyLane alignment plan is unaffected: this proposal renames, it does not add.
- `tests/api_contract/test_public_docstring_examples.py` passes. It fails closed
  in two directions that this change can trip:

  1. **It executes the examples.** `flagquantum/_api.py:30` is a live example
     asserting `fq.compile(fq.Circuit(2).h(0).cx(0, 1)).n_wires == 2`; it runs
     today because `fq.compile` is in the gate's `ENTRIES`. Measured: the finder
     reports that example and its expected result. It must be updated to
     `n_qubits` so the shipped example does not teach the name being retired.
  2. **It fails on any module with a `>>>` example that is absent from
     `ENTRIES`.** So adding an illustrative example of the deprecated spelling to
     a module that is not already listed — `flagquantum/ecosystem/simulators/advisor.py:897`
     already carries `recommend(n_wires=22)` and `recommend_simulator` is listed —
     forces an `ENTRIES` update in the same change. Deprecation guidance belongs
     in the migration document, not in a new public docstring example.

- The environment-variable reader accepts both names and refuses a conflicting
  pair: with only `FQ_STATEVECTOR_PERSISTENT_WIRE_LAYOUT=0` set the executor
  behaves as before and warns; with only `..._QUBIT_LAYOUT=0` set it is silent;
  with both set to different values it raises.

## Open Questions

1. **Does the product want `to_json()` to say `n_qubits`?** Candidate A says no for
   now. If yes, that is Candidate B and needs the IR version decision, which
   `FQ-CIRCUIT-EXPRESSIVENESS-CONTRACT-20260930.md` reserved as a compatibility
   decision with a migration path.
2. **How much of `flagquantum/benchmarking/**` is user surface?** This proposal
   treats it as not-user-surface and lists it in the gate's exclusions. If a
   benchmark harness entry is documented as a user workflow, it moves into
   row 22.
3. **Does `fq.algorithms` need the alias treatment or an in-place rename?** This
   proposal says in-place, since `API_CHANGE_PROPOSAL_001_STABLE_CORE.md` scopes
   Stable Core to construction/IR and the `fq.*` root, `docs/public_api_v1.json`
   contains no `algorithms` entry, and `docs/guides/ALGORITHMS.md:61` already
   documents that these units carry no root-level `fq.` name.
4. **`twin/` and `experimental/`.** Measured: `fq.twin` exposes exactly one
   `wire`-named public signature (`TwinPrediction.n_wires`, row 21) and
   `fq.experimental` exposes none. `TwinPrediction` carries a published
   `schema="flagquantum.twin_prediction.v1"`, so its rename follows row 21's
   alias treatment rather than row 22's in-place rename.
5. **Should the environment variable be renamed at all?** This proposal says yes,
   through a read-both-names window, because the user constraint is about the
   surface a user meets and the variable appears in a shipped example README
   (`examples/distributed_statevector_topologies/README.md:48`). If the product
   prefers to freeze operational names, the variable moves to the boundary
   exceptions and the gate excludes it explicitly.
6. **How many execution-metadata keys are user-visible?** `CircuitAnalysis.wire_usage`,
   `LayerPlan.wires`, `MPSPartition.wires`, `RankOwnedMPSState.n_wires`,
   `MPSReverseTapeRecord.wires`, and the `max_marginal_wires` key that
   `runtime/measurements.py:308-322` both reads and quotes in a user-facing
   `ValueError` are dataclass fields and dict keys, not function parameters, so no
   row above renames them. Whether they should move is unresolved and is not
   required by the user's constraint as stated.

## Owner and approvals

Owner: integration/API maintainers. Implementation owners by team, from
`team-ownership.toml`:

| Surface | Team |
|---|---|
| `flagquantum/circuit.py`, `flagquantum/core/**`, `flagquantum/observables/**` | `core` |
| `flagquantum/runtime/**` (`ExecutionPlan`) | `runtime` |
| `flagquantum/algorithms/**` | `algorithms` |
| `flagquantum/compiler/**`, `flagquantum/drawer/**`, `flagquantum/kernels/**`, `flagquantum/noise/**`, `flagquantum/deployment/**`, `flagquantum/remote/**`, `flagquantum/simulation/**` | the owning team for each |
| `contracts/**`, `docs/public_api_v1.json`, `tools/**`, `docs/architecture/decisions/**` | `integration` first, then `verification` for the gate |

The first deprecation release and the removal release are selected by the release
owner; removal must not precede the 0.3.x window that `#23` published. This
proposal asserts no calendar date and bumps no version.

Status: proposed. Implementation is authorized by the user for the Python
surface; the serialized-key question in *Open Questions* 1 remains open and
nothing in this proposal depends on it being answered.

Implemented under this authorization so far, by row number:

| Row | Surface | Landed |
|---|---|---|
| 11 | `fq.X` / `fq.Y` / `fq.Z` take `qubit`, with `wire=` deprecated | yes |
| 12 | `fq.I` takes `qubit`, with `wire=` deprecated | yes |
| — | `fq.{probabilities,samples,counts}` message strings, `I`/`X`/`Y`/`Z` docstrings, and the `Pauli tensor products require disjoint qubits` refusal | yes |

Every other row is unimplemented. A row is only "landed" when its
*Acceptance Tests* line passes in the same change, which is why rows 1-10 and
13-22 carry no entry yet.
