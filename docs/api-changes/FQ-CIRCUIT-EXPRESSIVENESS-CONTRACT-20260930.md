# Circuit expressiveness contract

## Decision and authorization

Status: **proposed, not approved.** This document records what the FlagQuantum IR
can and cannot express today, the evidence for each boundary, and the contract that
W2 (measure, reset, barrier, classical registers) and W3 (structured control flow)
would have to satisfy. It contains no code change and claims no approval.

It is the prerequisite named by the parity backlog: **W2-05 is an IR version
change** in the protected `flagquantum/core/ir/**` surface, and it must follow this
proposal's approval. The reason is in the evidence below — the IR version is an
exact-match pin with two persistence surfaces, so W2 cannot be a feature branch
that adds fields.

Scope of the affected surface: `flagquantum/core/**` (the protected IR),
`flagquantum/compiler/**`, `flagquantum/runtime/dynamic/**`,
`flagquantum/ecosystem/**`, and `docs/public_api_v1.json`.

## Problem and affected user journey

### The opcode contract is open, not closed

`flagquantum/core/operator_schema.py` opens with "Single-source backend-neutral
operator semantics" and registers **35 opcodes**: 31 unitaries and 4 channels
(`amplitude_damping`, `bit_flip`, `depolarizing`, `phase_flip`). None of them is
`measure`, `reset`, or `barrier`.

`Instruction.__post_init__` (`core/ir/__init__.py:338-379`) rejects an unknown opcode only
when three escape conditions are all false:

```python
is_channel = bool(self.metadata.get("is_channel"))
is_dynamic = bool(self.metadata.get("is_dynamic"))
if schema is None and self.matrix is None and not is_channel and not is_dynamic:
    raise IRValidationError(
        f"unknown opcode {name!r}; custom operations require an explicit matrix"
    )
```

Because the two flags are read from free-form `metadata`, any string becomes a legal
IR instruction. Measured against this checkout:

| Construction | Result |
| --- | --- |
| `Instruction("totally_bogus_opcode", (0,), metadata={})` | `IRValidationError: unknown opcode ...` |
| `Instruction("totally_bogus_opcode", (0,), metadata={"is_dynamic": True})` | **accepted into `CircuitIR`** |
| `Instruction("measure", (0,), metadata={"is_dynamic": True})` | accepted |
| `Instruction("reset", (0,), metadata={"is_dynamic": True})` | accepted |
| `Instruction("barrier", (0, 1), metadata={"is_dynamic": True})` | accepted |

The escape hatch is load-bearing, not accidental: `is_dynamic` is read or written at
**16 sites outside `core/ir/__init__.py`**, across `ecosystem/qiskit`, `ecosystem/cirq`,
`runtime/dynamic`, `runtime/planner`, `compiler/_hybrid`,
`compiler/target_emission.py`, `compiler/directed_topology.py`,
`compiler/schedule_legalization.py`, and `services/preflight.py`.

This is the expressiveness defect in one sentence: **the set of legal operations is
not owned by any module.** Core cannot enumerate what a program may contain, so no
consumer can either.

### `measure` and `reset` are Core opcodes whose semantics live in Runtime

`measure` and `reset` have no schema entry, so their contract is defined by whoever
happens to read them. Five consumers read them, with different rules:

| Consumer | Rule |
| --- | --- |
| `runtime/dynamic/circuit.py:25,37` | produces them with `metadata={"is_dynamic": True, "classical_bit": bit}` |
| `runtime/dynamic/hybrid_session.py:113-131` | **strict**: requires `set(metadata) == {"is_dynamic", "classical_bit"}` for `measure` and `== {"is_dynamic"}` for `reset`; requires a unique non-negative integer bit |
| `compiler/schedule_legalization.py:84-93` | **lenient**: requires only that `classical_bit` be a non-negative int; ignores the rest of the key set |
| `ecosystem/qiskit/conversion.py:355-381` | imports both, sets `is_dynamic`, `classical_bit` |
| `compiler/qcis.py:93` | drops `measure` and `barrier` by name |

The strict and lenient readings disagree: a `measure` instruction carrying an extra
metadata key is **accepted by the Compiler and rejected by the hybrid session**. That
is a fail-open/fail-closed divergence on a Core opcode, and it exists because neither
consumer can consult a shared schema.

`services/preflight.py:150` then special-cases `is_dynamic` on its own, and
`runtime/planner/__init__.py:616` treats any `is_dynamic` or `conditions` instruction
as dynamic. Six places, six notions of what "dynamic" means.

### `barrier` has two contradictory definitions, and one of them deletes it

There is no registered `barrier` opcode and no `Circuit.barrier` method, so the name
can only reach the IR through the `is_dynamic` lie. Meanwhile two mechanisms claim to
handle barriers:

**By name — lossy.** The Qiskit importer drops it
(`ecosystem/qiskit/conversion.py:343-353`) with severity `warning`:

> "Qiskit barriers constrain later compilation and are not part of FlagQuantum IR v1."

A dropped barrier is not lost information; it is a **changed program**. Two gates on
disjoint wires may be reordered across a barrier, and a user who inserted one to pin
that ordering gets a different schedule with only a warning to show for it.
`ecosystem/qiskit/conformance.py:396-411` pins this as
`barrier_requires_explicit_lossy_import`, so it is a documented, tested loss.

**By metadata — inferred.** `compiler/schedule_legalization.py:148-152` never looks at
the name:

```python
is_barrier = bool(
    instruction.metadata.get("is_dynamic")
    or instruction.metadata.get("is_channel")
    or condition_bits
)
```

So a `measure`, a `reset`, and a **noise channel** are all scheduling barriers, and
the dependency edges they create are labelled `"barrier"` in the recorded evidence
(`core/_compilation_evidence.py:98` lists `{"wire", "barrier", "classical"}`).

The two definitions meet in the worst possible way: **the only way to express a real
barrier is to claim it is dynamic, which is the same claim a measurement makes — so
the scheduler cannot distinguish the directive a user wrote from the measurement the
compiler inserted.** Both produce `dependency_kinds` containing `"barrier"`.

### Classical storage has no IR field, so Ecosystem escapes it into metadata

`CircuitIR` fields are `n_wires`, `instructions`, `version`, `dtype`, `shape`,
`observables`, `measurements`, `metadata`. There is no classical-bit count and no
classical register, and `MeasurementNode` is `kind`, `wires`, `shots`, `metadata`.

The classical width nevertheless exists, in two unowned places:

- `ecosystem/qiskit/conversion.py:492` writes `ir.metadata["interop"]["num_clbits"]`;
- `ecosystem/qiskit/conversion.py:566` reads it back and repairs the width from
  `classical_bit` and condition operands when it is absent.

An Ecosystem adapter is therefore the authority on a program-level property. The same
file compares `cregs` at `:307-312` to decide whether a Qiskit circuit is
"standard", which is a judgement about a structure Core cannot represent at all.

### `fq.Circuit` is not the expressive type

`fq.Circuit` exposes 123 public attributes — gate constructors, analysis, execution,
and drawing. `measure`, `reset`, and `barrier` are not among them. They exist only on `DynamicCircuit`
(`runtime/dynamic/circuit.py:15`), which subclasses `Circuit` and is exported from
`runtime/dynamic/__init__.py:20` and `flagquantum/dynamic.py:5` — but **not** from
`flagquantum`, and **not** in `docs/public_api_v1.json`.

Measured: `hasattr(fq.Circuit, "measure")` is `False`, `hasattr(fq, "DynamicCircuit")`
is `False`. So whether a user can express a mid-circuit measurement depends on
whether they knew to import a name the public API does not list.
`runtime/dynamic/execution.py:779-780` reinforces the split by raising
`TypeError("run_dynamic requires an experimental DynamicCircuit")` — but the
capability is described in the parity backlog as an L1 gap, not an experimental one.

### W2-05 cannot be an ordinary feature PR

`IR_VERSION = "1.0"` (`core/ir/__init__.py:24`) is an **exact-match pin**, not a compatibility
range:

- `CircuitIR.__post_init__:440-443` raises `IRValidationError` for any other value
  and then **overwrites** the field with the constant at `:445`;
- `runtime/execution_plan_contract.py:635` serializes `"ir_version": IR_VERSION`;
- `runtime/training_state.py:311` serializes it and `:363` **rejects** a payload whose
  `ir_version` differs;
- `tests/unit/test_ir_public_api.py` and `tests/test_qiskit_interop.py` pin it.

Adding classical resources therefore changes the meaning of every serialized program
artifact and every training checkpoint that carries the version. The backlog is right
that W2-05 is the version change; the point this document adds is that a version
change is a **compatibility decision with a migration path**, not a field addition,
and that the decision must be taken before W2-01 through W2-04 choose the fields.

### Affected user journeys

1. A user who writes a Qiskit circuit with a `barrier` to pin a two-wire ordering
   receives a schedule that may reorder it, with a warning they will not read.
2. A user who builds a circuit with `fq.Circuit` and then needs a mid-circuit
   measurement must discover `DynamicCircuit` outside the documented public API, and
   then must not use `fq.run`.
3. A contributor who adds a new opcode must decide whether to register a schema, set
   `is_dynamic`, set `is_channel`, supply a `matrix`, or some combination. Nothing
   rejects the wrong choice, and 16 call sites have already made it differently.
4. A maintainer planning W3 (structured control flow) cannot name the condition
   vocabulary, because it currently lives in free-form metadata read by
   `compiler/schedule_legalization.py:43-54` (`conditions` versus
   `condition_clauses`, mutually exclusive, checked at the point of use).

## Evidence

| Claim | Evidence |
| --- | --- |
| 35 registered opcodes, 0 of them measure/reset/barrier | `core/operator_schema.py`; enumerated via `OPERATOR_SCHEMAS` |
| Module self-describes as the single semantic source | `core/operator_schema.py:1` |
| Unknown opcode is accepted when `is_dynamic` is set | `core/ir/__init__.py:359-367`; reproduced against this checkout, table above |
| `is_dynamic` is used at 16 sites outside `core/ir/__init__.py` | `grep -rn 'metadata.get("is_dynamic")\|"is_dynamic":' flagquantum/` |
| `measure`/`reset` produced only by `DynamicCircuit` | `runtime/dynamic/circuit.py:25,37` |
| Strict metadata key-set validation | `runtime/dynamic/hybrid_session.py:113-131` |
| Lenient metadata validation for the same opcode | `compiler/schedule_legalization.py:84-93` |
| `measure` and `barrier` dropped by the QCIS emitter | `compiler/qcis.py:93` |
| Qiskit barrier dropped with severity warning | `ecosystem/qiskit/conversion.py:343-353` |
| The drop is pinned by a conformance test | `ecosystem/qiskit/conformance.py:396-411` |
| Barrier inferred from metadata, never from the name | `compiler/schedule_legalization.py:148-152` |
| Barrier edge kind exists in recorded evidence | `core/_compilation_evidence.py:98` |
| Channels are scheduling barriers | same `is_barrier` expression, `is_channel` branch |
| No classical field in `CircuitIR` or `MeasurementNode` | field lists from `dataclasses.fields` |
| Classical width lives in interop metadata | `ecosystem/qiskit/conversion.py:492`, re-read at `:566` |
| Classical structure compared inside an adapter | `ecosystem/qiskit/conversion.py:307-312` |
| `fq.Circuit` has no measure/reset/barrier | 123 public attributes, none of them these three; `hasattr` is `False` for all three |
| `DynamicCircuit` is not a public name | `hasattr(fq, "DynamicCircuit")` is `False`; absent from `docs/public_api_v1.json` |
| `DynamicCircuit` is reachable by module path | `runtime/dynamic/__init__.py:20`, `flagquantum/dynamic.py:5` |
| Dynamic execution requires the subclass | `runtime/dynamic/execution.py:779-780` |
| IR version is an exact match | `core/ir/__init__.py:440-445` |
| IR version is serialized twice and read once | `runtime/execution_plan_contract.py:635`, `runtime/training_state.py:311,363` |
| IR version is pinned by tests | `tests/unit/test_ir_public_api.py`, `tests/test_qiskit_interop.py` |
| Condition vocabulary is free-form metadata | `compiler/schedule_legalization.py:43-54` |
| Dynamic detection is duplicated | `runtime/planner/__init__.py:616`, `services/preflight.py:150` |

## Decision Candidates

These are the candidate contracts for the owners to choose between. They are not
alternatives to each other in every respect — candidates 1 and 2 are decisions the
others depend on.

**Candidate 1 — Close the opcode set, and give every semantic kind a schema entry.**
Replace the `is_dynamic` / `is_channel` metadata escapes with schema facts:
`measure`, `reset`, and `barrier` become registered opcodes with explicit
`semantic_kind` values beyond the current `{unitary, channel}`, and an unknown opcode
is rejected regardless of metadata. Custom matrices remain the supported extension
path and keep their current meaning. Consequence: the IR can enumerate what a program
may contain, and W3's control-flow nodes have a precedent to follow.

**Candidate 2 — Make classical storage a first-class IR field.** Add the classical
bit count and register model to `CircuitIR`, and delete the
`metadata["interop"]["num_clbits"]` round-trip. Consequence: this is the change that
forces the W2-05 version decision, and it must be decided together with candidate 1
because `measure` needs a destination for its bit.

**Candidate 3 — Represent `barrier` as an explicit directive.** Give it an opcode, a
wire list, and the barrier constraint it already produces at
`compiler/schedule_legalization.py:166-204`; stop inferring barriers from `is_dynamic`
and stop dropping Qiskit barriers at import. Consequence: the scheduler can
distinguish a user directive from a measurement, and the Qiskit importer's
`barrier_dropped` warning becomes a fidelity improvement rather than a documented
loss. This is the smallest candidate that fixes a live correctness problem, and it is
independently valuable even if 1 and 2 are deferred.

**Candidate 4 — Document the boundary and change nothing.** Record the open opcode set
and the metadata escapes as intentional, and let W2/W3 add opcodes the same way.
Consequence: the 16 `is_dynamic` sites keep growing, and the Compiler/Runtime
disagreement about `measure` metadata remains. This is the status quo, named so the
owners can choose it deliberately.

**Candidate 5 — Unify the dynamic path under `fq.Circuit`.** Make `measure`, `reset`,
and `conditionals` available on `fq.Circuit` (or promote `DynamicCircuit` to a
documented public name with a stated capability level) and remove the
`run_dynamic requires an experimental DynamicCircuit` split. Consequence: expressiveness
becomes a property of the public type instead of a subclass, but the user-facing API
gains new stable names and therefore needs the API-owner path.

## Prohibited Practices

The following are recorded as prohibited regardless of which candidate is approved, so
that the contract does not regress while it is being decided:

1. **Adding a new opcode by setting `metadata["is_dynamic"]` without a schema entry.**
   That is the mechanism this document exists to retire; using it once more for a new
   operation makes retirement harder.
2. **Silently dropping a directive that constrains compilation.** A barrier, a delay,
   or any future timing constraint may be unsupported, but its removal must be
   reported at the stage that removes it and must not be downgraded below the severity
   that a changed program deserves.
3. **Storing a program-level structure fact in `ir.metadata`.** Free-form metadata is
   for provenance and interoperability annotations; a field that the Compiler,
   Runtime, or an executor must read to be correct belongs in the IR.
4. **Defining a Core opcode's contract in a consumer.** `runtime/dynamic/hybrid_session.py`
   and `compiler/schedule_legalization.py` currently hold two different definitions of
   a valid `measure`. A third must not be added.
5. **Bumping `IR_VERSION` to make a new field readable.** The version is an exact-match
   pin with persisted readers; a bump requires a stated migration path for
   `execution_plan_contract` payloads and `training_state` checkpoints, or it silently
   invalidates them.

## Compatibility

- **`IR_VERSION` and serialized programs.** Any candidate that adds an IR field
  changes what `"1.0"` means. The compatible path is either to keep `"1.0"` readable
  for programs without the new fields, or to accept `"1.0"` and `"1.1"` as a range and
  migrate on read. `core/ir/__init__.py:440-443` currently refuses anything but the constant, and
  `runtime/training_state.py:363` refuses a checkpoint on the same basis, so the choice
  is a two-reader migration and not a one-line change.
- **Stable Core names.** `IR_VERSION`, `CircuitIR`, `Instruction`, and `IRSerializationError`
  are Stable Core (`docs/public_api_v1.json`). Adding fields is compatible; tightening
  validation is not, because programs that were accepted under the `is_dynamic` escape
  would begin to fail. If candidate 1 is approved, the tightening needs a stated
  deprecation window or an explicit "these were never valid" judgement with evidence.
- **Ecosystem adapters.** `ecosystem/qiskit/conformance.py:396-411` pins the lossy
  barrier import. Candidate 3 changes a pinned conformance expectation, which is a
  deliberate evidence update, not a test fixup.
- **`DynamicCircuit`.** Candidate 5 adds public names and therefore enters the
  API-change path. Candidates 1-4 leave the name where it is: reachable by module path,
  absent from the stable surface.
- **Compiler evidence.** `core/_compilation_evidence.py:98`'s dependency-kind set is
  recorded evidence. If barrier becomes an explicit directive while channels remain
  inferred barriers, the recorded kinds keep their meaning, but the *reason* a
  `"barrier"` edge exists changes and the evidence schema should say which.

## Acceptance Tests

Whichever candidate is approved, the change is complete only when these fail before
it and pass after:

1. **Closed-set test.** Constructing an `Instruction` with an unregistered name and
   `metadata={"is_dynamic": True}` raises `IRValidationError`, and every opcode that
   the Qiskit and Cirq importers can emit is registered. (Candidates 1, 3.)
2. **Single definition of `measure`.** One test asserts that a `measure` instruction
   with an unexpected metadata key is either accepted by every consumer or rejected by
   every consumer; the current accept-in-Compiler / reject-in-Runtime split fails it.
   (Candidates 1, 4-approval.)
3. **Barrier fidelity.** A Qiskit circuit with a barrier between two disjoint-wire
   gates either converts without loss or reports a severity that reflects a changed
   schedule; the converted program's recorded barrier constraint survives to
   `schedule_legalization`. (Candidate 3.)
4. **Directive distinguishable from measurement.** A test asserts that a program
   containing one barrier and one measurement produces recorded dependency evidence
   that distinguishes them. (Candidate 3.)
5. **Classical width has one owner.** A round-trip test converts a Qiskit circuit with
   non-standard classical registers and asserts that the width and register structure
   survive without consulting `metadata["interop"]`. (Candidate 2.)
6. **Version migration.** Loading a `"1.0"` execution-plan payload and a `"1.0"`
   training checkpoint built before the change either succeeds or fails with a
   documented, actionable error. (Any candidate that touches the IR.)
7. **Public expressiveness.** A test asserts that the documented public entry point can
   express a mid-circuit measurement and a barrier, or asserts explicitly that it
   cannot and that the capability is declared at its true maturity level.
   (Candidate 5.)

## Open Questions

1. **What is the minimum opcode set for W2 and W3?** W2 needs `measure`, `reset`,
   `barrier`; W3 needs control flow. Should the closed set be defined once, covering
   both, or incrementally per milestone?
2. **Does `barrier` carry wires or is it global?** `compiler/schedule_legalization.py:203-205`
   currently promotes a barrier to a global one by writing `last_by_wire[wire] = index`
   for every wire, which makes a single-wire barrier behave as a full-width barrier.
   Is that the intended semantics or an artifact of inferring barriers from channels?
3. **Is `is_channel` also to be retired?** `noise/channels.py` and the noise registry
   use it, and channels are not instructions a user writes. If channels keep a metadata
   flag while dynamic opcodes get schemas, the IR has two registration mechanisms.
4. **Which layer owns the condition vocabulary?** `condition_clauses` versus
   `conditions` is validated at the point of use in the Compiler, and W3 will need the
   same vocabulary in Core.
5. **Is classical storage per-circuit or per-program?** `num_clbits` is currently a
   circuit-level count, while conditions reference bit indices. The backlog's W2-02
   and W2-03 split a register model from a bit and type model; which of them the IR's
   `CircuitIR` field references is a real decision.
6. **Which maturity level does dynamic execution claim?** The parity backlog treats
   mid-circuit measurement as an L1 gap; the repository gates it behind a non-stable
   subclass and `run_dynamic`. The owners should state the level rather than leave the
   gap between the two.

## Owner and approvals

- Owning domain: `core` for the IR contract, with `integration` for the record.
  `docs/api-changes/**` is a shared path, and `python tools/check_team_scope.py --team
  integration --files docs/api-changes/FQ-CIRCUIT-EXPRESSIVENESS-CONTRACT-20260930.md`
  passes.
- Required approvals before implementation: **core domain owner** (opcode set,
  classical storage, and the version decision), **API owner** (only if candidate 5 is
  chosen, because it adds stable names), and **integration owner** (the change lands in
  the protected `flagquantum/core/ir/**` surface).
- Sequencing: this document, then the candidate choice, then W2-05 (the version and
  migration decision), then W2-01 through W2-04 (the fields and instructions). W2-06
  through W2-09 depend on the Core contract and must not precede it, because the
  Compiler, Runtime, and Ecosystem work all read the same five consumers listed above.
- Implementation note for the owner: the smallest useful step is **candidate 3 alone**.
  It closes a live correctness problem (barriers are dropped, then re-inferred from the
  wrong signal), it needs no classical storage and no control flow, and it makes the
  `is_dynamic` retirement easier rather than harder.
