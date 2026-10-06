# Circuit composition: `Circuit.compose`

## Decision and authorization

Status: **implemented on this branch, pending review.** This document records the
decision to add one method to the Stable Core type `fq.Circuit`, the evidence that
the method is a rewrite of qubit labels rather than a second program description, and
the compatibility analysis that follows from that.

It is written under non-negotiable rule 8 of `AGENTS.md` ("Treat the Stable Core
public API as protected"), and it is the narrative half of the API change whose
numbered half is
[`API_CHANGE_PROPOSAL_066_CIRCUIT_COMPOSITION.md`](../development/API_CHANGE_PROPOSAL_066_CIRCUIT_COMPOSITION.md).
This document was drafted while that proposal was still unwritten and named it as
`065`; `065` was subsequently issued to the `wire` → `qubit` vocabulary program and
`067` to the attribute half of that same program, so the composition proposal took
the free number `066`. This proposal covers `compose` only; `066` records the
family the other three belong to, and approves neither `control` nor `power`.

Scope of the affected surface: `flagquantum/circuit.py` (the Stable Core type),
`flagquantum/core/qubit_mapping.py` (new), `docs/reference/API.md`, and
`tests/**`.

## Problem and affected user journey

### A program cannot be placed on chosen qubits, so users re-walk it by hand

FlagQuantum has a library of gate sequences -- `algorithms/primitives/qft.py`,
`algorithms/primitives/oracle.py`, `algorithms/primitives/ansatz.py` -- and every one
of them exists in two disconnected forms:

| Form | Example | What it can do |
| --- | --- | --- |
| free function that appends | `append_qft(circuit, wires, inverse=False)` | writes into a caller's circuit, on qubits the caller names |
| factory that returns a circuit | `qft(n_wires, inverse=False)` | returns a standalone `Circuit(n_wires)` |

Neither reaches the other. `qft(3)` produces a three-qubit circuit that a user of a
five-qubit program cannot place at qubits 1..3 without re-emitting it gate by gate:

```python
from flagquantum.algorithms.primitives import qft

block = qft(3)
target = fq.Circuit(5)
for instruction in block.to_ir().instructions:
    target.gate(
        instruction.name,
        tuple(qubit + 1 for qubit in instruction.wires),
        params=instruction.params,
    )
```

That loop is the defect, and it fails in two measured ways. It passes `name`, `wires`,
and `params` only, so `matrix` and `metadata` are dropped:

- an arbitrary-unitary instruction does not survive it at all --
  `fq.Circuit(2).any(0, unitary=m)` re-walked onto another qubit raises
  `IRValidationError: unknown opcode 'any'; custom operations require an explicit matrix`,
  because the unitary the instruction carried is the only thing that made the opcode
  legal;
- a channel instruction survives but loses `metadata={"is_channel": True}`, which is
  the flag the planner, the noise selector, and five executors read to recognize a
  channel. The reported metadata is `{}` after the re-walk.

### The two existing forms encode one rule twice and disagree

`append_qft`'s docstring says "``wires``: The wires carrying the transform, most
significant first", and `twin/region_model.py:59` holds a third copy of the same
idea:

```python
def _remap_wires(wires, *, local_to_region):
    try:
        return tuple(local_to_region[wire] for wire in wires)
    except KeyError as error:
        raise ValueError("Twin noise rule references a wire outside its mapping")
```

Three implementations of "relabel the qubits of a recorded sequence", in three
domains, with three refusal messages. The relabelling rule is Core semantics -- it is
a property of `Instruction`, which has qubits and nothing else -- so this is a rule
that three callers each own a private copy of.

## Evidence

Measured on this checkout (`flagquantum 0.2.0`), before the change:

| Construction | Result |
| --- | --- |
| `qft(3).to_ir().n_wires` | `3` |
| `fq.Circuit(5).compose(qft(3), qubits=(1, 2, 3))` | `AttributeError: 'Circuit' object has no attribute 'compose'` |
| `fq.Circuit(5).gate("h", 1.0)` | `TypeError: Gate 'h' wire must be an integer, got 1.0` |
| a `matrix=` arbitrary unitary carried through the re-walk loop above | `IRValidationError: unknown opcode 'any'; custom operations require an explicit matrix` |
| a channel instruction carried through the re-walk loop above | survives with `metadata == {}`, losing `is_channel` |
| the same channel instruction carried through `compose` | survives with `metadata == {"is_channel": True}` |

After the change, on the same checkout:

| Construction | Result |
| --- | --- |
| `fq.Circuit(5).h(0).compose(qft(3), qubits=(1, 2, 3)).to_ir().instructions` | equal, instruction by instruction, to `fq.Circuit(5).h(0)` followed by `append_qft(manual, [1, 2, 3])` |
| `fq.Circuit(5).compose(bell, qubit_map={0: 3, 1: 4})` | equal to `compose(bell, qubits=(3, 4))` |
| `fq.Circuit(3).compose(bell.to_ir(), qubits=(1, 2))` | equal to composing the `Circuit` |
| `c.compose(c, qubits=(1, 0))` for `c = fq.Circuit(2).h(0)` | two instructions, no unbounded growth |
| an `rx` with a per-batch angle, composed | the same `torch.Tensor` identity of the angle, unmodified |
| `compose(..., qubits=(1,))` on a two-qubit block | `ValidationError: Circuit.compose qubits must name all 2 qubit(s) of the composed program, got 1` |
| `compose(..., qubits=(1, 1))` | `ValidationError: Circuit.compose cannot place two local qubits on qubit 1: (1, 1)` |
| `compose(..., qubits=(1, 2))` on a two-qubit circuit | `ValidationError: Circuit.compose target qubit(s) (2,) outside circuit range [0, 1].` |
| `fq.Circuit(4, bsz=2).compose(fq.Circuit(1).h(0))` | `ValidationError: Circuit.compose cannot mix batch sizes: this circuit has bsz=2, the composed program has bsz=1.` |

## Decision Candidates

**A. `Circuit.compose(other, *, qubits=None, qubit_map=None)` (chosen).** One method
on the type that already owns instructions, reading the source program's instructions
and appending rewritten copies.

**B. Keep the free-function form and add a second entry.** `compose_into(target,
source, wires)`. Rejected: it puts the operation outside the object that owns the
instruction list, so `c.compose(...)` -- the form every comparable framework offers --
would remain impossible, and the chaining style `fq.Circuit(5).h(0).compose(...)`
that the rest of the API uses would break at exactly the point where programs start
being reused.

**C. Make the primitives take a target qubit list** -- `qft(n_qubits, *, qubits=None)`
-- and leave `Circuit` alone. Rejected: it fixes `qft` and not the general problem.
Any program the user already has (`block = fq.Circuit(2).h(0).cx(0, 1)`, a circuit
loaded from QASM, an `algorithms` factory) would still be unplaceable, and it would
push the same qubit-mapping argument into every factory in the library.

**D. Introduce a composition IR node.** Rejected as a violation of rule 6 ("Do not
create a second source of truth") and of `AGENTS.md` engineering decision principle 7:
a nested-program node would be a second way to describe a program, and every executor,
compiler pass, drawer, and importer would have to learn it. Composition emits
instructions the existing IR already describes, so `IR_VERSION` stays `"1.0"`.

## Prohibited Practices

- **Do not make `compose` a second IR.** It appends `Instruction` values; it does not
  introduce a node, a wrapper, or a nested circuit type, and `to_ir()` is unchanged.
- **Do not give a source qubit an identity image by default.** A partial or mistyped
  map is refused. Mapping an unnamed qubit to itself would turn `qubits=(1,)` on a
  two-qubit block into a program that runs and acts on the wrong qubit.
- **Do not silently rebroadcast a batch dimension.** `bsz` is a property of the
  program, and two programs that disagree are refused rather than reconciled.
- **Do not deprecate `append_qft` in this change.** It is the in-place form the
  algorithms already use, and removing it is a separate decision with its own
  migration.
- **Do not add `adjoint`, `control`, or `power` here.** Each has its own semantics
  (inverse existence, ancilla allocation, exactness) and its own proposal, which is
  why the plan numbers them apart.

## Compatibility

| Item | Before | After |
| --- | --- | --- |
| `fq.Circuit` exports | unchanged (`docs/public_api_v1.json` `stable_exports` untouched) | unchanged |
| `Circuit` method set | `compose` absent | `compose` added |
| `IR_VERSION` | `"1.0"` | `"1.0"` |
| `CircuitIR` fields | unchanged | unchanged |
| `Instruction` fields | unchanged | unchanged |
| `gate()`, `any()`, `to_ir()`, `from_ir()` | unchanged | unchanged |
| label refusal message | `Gate 'h' wire must be an integer, got 1.0` | `Gate 'h' qubit must be an integer, got 1.0` |

The method addition is additive: no existing call site changes meaning, and no
existing keyword or default moves. The one message change is the user-facing
terminology rule of
[`FQ-QUBIT-NAMING-20260913.md`](FQ-QUBIT-NAMING-20260913.md): the label reader is new
code and speaks `qubit` from the start rather than acquiring a `wire` message that
the migration would have to rename later. `tests/test_native_circuit.py` is updated
with it; the IR field `Instruction.wires` is deliberately **not** renamed, for the
reason that proposal records.

Sharing the label reader with `twin/` is planned as its own change (`N1-2`), because
`flagquantum/twin/**` is owned by a different team: `tools/check_team_scope.py
--team core` refuses `flagquantum/twin/region_model.py` in this branch, and the
replacement belongs to the team that owns the consumer.

One third copy of the label rule is deliberately left in place:
`flagquantum/observables/__init__.py:123 _wire` and `:145 _wires`. They are owned by
`core`, so they could be consolidated here, but they carry two rules of their own --
a non-negative label and a uniqueness check with its own message -- and their
messages name `wire` because `OutputRequest` still takes `wires=`. Consolidating
them is part of the qubit-naming track (`Q3`/`Q6`), where the public keyword and the
message are renamed together; doing it here would rename half of that surface and
leave the other half inconsistent. This proposal records the copy rather than
silently leaving a second source of truth unremarked.

## Acceptance Tests

The contract of this change is "compose is a rewrite of qubit labels, not a new way
to describe a program", so the acceptance test is instruction-by-instruction equality
of the versioned IR against the hand-built circuit:

1. `tests/unit/test_circuit_compose.py::test_compose_is_end_to_end_the_hand_built_circuit`
   -- `compose(block, qubits=(1,2,3))` on a six-qubit circuit equals the circuit built
   by emitting the same gates on those qubits, compared through `to_ir()`.
2. `...::test_compose_accepts_a_mapping_as_well_as_a_sequence` -- the two spellings
   of one total map agree.
3. `...::test_compose_accepts_a_circuit_ir` -- the `Circuit` and `CircuitIR` forms of
   one program compose to the same circuit.
4. `...::test_compose_preserves_execution` -- the composed circuit runs and its
   `complex128` statevector equals the hand-built one.
5. `...::test_compose_keeps_parameters_and_the_batch_dimension` -- a per-batch angle
   keeps its tensor identity.
6. `...::test_compose_invalidates_the_cached_ir` -- a composed circuit does not
   report the cached IR it had before the append.
7. `...::test_compose_snapshots_the_source_so_a_circuit_composes_onto_itself` --
   `c.compose(c)` terminates.
8. The refusal parametrisation
   (`test_compose_refuses_a_placement_that_is_not_a_total_map`,
   `..._a_local_qubit_the_source_does_not_have`,
   `..._a_mapping_that_skips_a_local_qubit`,
   `..._two_spellings_of_one_placement`,
   `..._a_target_that_is_not_an_integer`,
   `..._a_source_that_is_not_a_program`,
   `..._mismatched_batch_sizes`) pins each fail-closed path.
9. `tests/test_native_circuit.py::test_a_gate_refuses_a_qubit_that_is_not_a_label`
   and its per-gate parametrisation pin the shared label reader.

## Open Questions

1. **Does the composition wave need `compose` to reach qubits outside the target
   circuit, widening it automatically?** Currently a target qubit past `n_qubits` is
   refused. Widening would make `fq.Circuit(2).compose(qft(3), qubits=(2, 3, 4))`
   succeed, which is convenient and hides a size decision inside a construction call.
   Open until `adjoint`/`control` land and the composition contract is written in one
   place.
2. **Should `compose` accept a raw instruction sequence**, so that a user can place
   the output of a compiler pass? The `CircuitIR` form covers most of that need;
   a third accepted type may not earn its place.
3. **Should the label reader move to `flagquantum/core/ir.py`?** It is Core
   semantics and `ir.py` already refuses a bad instruction qubit. It is a separate
   module today only to keep `ir.py` from growing; the placement is revisitable when
   `N1-2` adds the `twin/` consumer.

## Owner and approvals

- Implemented by: `core` (`flagquantum/circuit.py`, `flagquantum/core/**`).
- Proposal document by: `integration` (`docs/**` is a shared path).
- Requires: review approval of this document and of
  [`API_CHANGE_PROPOSAL_066_CIRCUIT_COMPOSITION.md`](../development/API_CHANGE_PROPOSAL_066_CIRCUIT_COMPOSITION.md)
  before `compose` is treated as a stable surface. The change is additive, so the
  branch could proceed to review without waiting for the numbered proposal; that
  document has now landed as `066`, which closes the condition this line recorded.
