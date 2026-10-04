# A global phase on the program: `CircuitIR.global_phase`

## Decision and authorization

Status: **proposed, pending review. Not implemented on this branch.** This
document requests one additive field on `CircuitIR`, and it lands as a proposal
with a measurement rather than as an implementation, because
`flagquantum/core/ir.py` is a protected integration surface and rule 8 of
`AGENTS.md` forbids changing a serialized public schema without explicit user
authorization. The authorization is what this document asks for.

What is requested:

1. A `global_phase: float` field on `CircuitIR`, defaulted to `0.0`, carrying the
   phase in radians such that the program's operator is `exp(1j * global_phase)`
   times the product of its instructions.
2. The field in `to_dict` / `from_dict`, with `IR_VERSION` moved from `1.0` to
   `1.1` and `1.0` payloads accepted as `global_phase = 0.0`.
3. Exactly one reader: the statevector executor, so `global_phase` reaches the
   computed amplitudes. Nothing else reads it.

Not requested here, and deliberately left to a separate proposal if it is wanted:
a writer. The first writer is the one that needs this field
(`Optimize1qGatesDecomposition`, backlog rung W9-06), and the field is useful on
its own without it, because it makes a phase that is already being lost *visible*.

It is written under rule 8 of `AGENTS.md` ("Treat the Stable Core public API as
protected"). The companion change that needs the field is the compiler pass
described under "Problem" below; that pass is **not** in this branch, and no
compiler code here reads a field that does not exist yet.

Scope of the affected surface: `flagquantum/core/ir.py` (protected, owned by the
integration team), `flagquantum/simulation/statevector/**` (owned by team
`simulation`), `contracts/qiskit-interop-contract.toml` (protected),
`docs/reference/API.md`, and the tests that pin a serialized IR. The eight
contract files that record `ir_version` and the tests that pin a literal
`content_hash` are enumerated in "Compatibility".

## Problem and affected user journey

### A phase that is dropped today, with an instrument that can see it

This is not a hypothetical missing generalization. The repository already reaches
a target basis through a synthesis whose result is documented as equal to its
source **up to one global phase**, and it already drops that phase.

`synthesize_one_qubit_matrix` states it in its own docstring
(`flagquantum/compiler/one_qubit_synthesis.py`): the result "is equal to `matrix`
up to one global phase, which FlagQuantum IR has no field to record."
`native_gate_legalization` reaches every target basis through that function, so
every legalization of a program carrying an `h`, a `t`, or any other gate the
target does not publish multiplies the program's statevector by a phase nothing
records.

Measured on `origin/main` at `76a416a8`, over 80 seeded two-wire programs per
target basis, comparing the statevector before and after `legalize_native_gates`
at `complex128`:

| Target basis | Programs legalized | Statevector unchanged | Phase is neither `1` nor `-1` | Amplitude gap, worst |
| --- | --- | --- | --- | --- |
| `ibm-rz-sx-cx` | 80 | 2 | 77 | `0.541` |
| `ibm-heron-cz` | 80 | 2 | 77 | `0.541` |
| `rotational` | 80 | 3 | 76 | `1.000` |
| `ion-trap-rz-rx-rzz` | 80 | 3 | 76 | `1.000` |
| `clifford-t` | 0 | 0 | 0 | — |

The smallest witness needs two instructions. `h` followed by the basis's
entangler legalizes on `ibm-rz-sx-cx` to `rz(pi/2); sx; rz(pi/2); cx`, and the
legalized statevector is the source's multiplied by `exp(-i*pi/4)`:

```python
>>> import torch
>>> from benchmarks.compiler_two_qubit_synthesis import DEFAULT_BASES, snapshot, _NOW
>>> from flagquantum.compiler.native_gate_legalization import legalize_native_gates
>>> from flagquantum.core.ir import CircuitIR, Instruction
>>> from flagquantum.simulation.statevector.local import run_local_statevector
>>> source = CircuitIR(2, (Instruction("h", (0,)), Instruction("cx", (0, 1))))
>>> legalized = legalize_native_gates(
...     source, snapshot=snapshot(DEFAULT_BASES[0]), evaluated_at=_NOW
... )
>>> [item.name for item in legalized.program.instructions]
['rz', 'sx', 'rz', 'cx']
>>> def statevector(program):
...     return run_local_statevector(
...         program, batch_size=1, device=torch.device("cpu"),
...         dtype=torch.complex128,
...     ).reshape(-1)
>>> statevector(source).tolist()
[(0.7071067811865475+0j), 0j, 0j, (0.7071067811865475+0j)]
>>> statevector(legalized.program).tolist()
[(0.5-0.49999999999999983j), 0j, 0j, (0.49999999999999994-0.49999999999999994j)]
```

Two instructions is the point: the phase cannot be attributed to a fold of many
gates. It is the single `h` the target has no `h` for.

### The route that exists is inert

The Qiskit interop boundary already carries this value in
`metadata["interop"]["global_phase"]`: `from_qiskit` writes it
(`flagquantum/ecosystem/qiskit/conversion.py:496`) and `to_qiskit` reads it back
(`:578`), which is what `contracts/qiskit-interop-contract.toml:19`
(`global_phase = "preserved"`) means. That route is preserved by the pipeline —
measured, both `optimize` and `legalize_native_gates` keep the `metadata` mapping
intact — and it is **invisible to execution**:

```python
>>> from flagquantum.core.ir import CircuitIR
>>> carried = CircuitIR(2, source.instructions, metadata={"interop": {"global_phase": 0.5}})
>>> torch.equal(statevector(source), statevector(carried))
True
```

So the phase survives a Qiskit round trip and vanishes the moment anything
computes with the program. `metadata` is the right home for provenance and the
wrong home for semantics: it is a free-form mapping with no validation, no
contract, and no reader. A phase in it is a comment.

### Why this blocks a backlog item rather than being a nicety

Backlog rung W9-06 is Qiskit's `Optimize1qGatesDecomposition`, which collects the
single-qubit runs and re-synthesizes each one **straight into the target's basis**,
folding the leftover phase into `dag.global_phase`. The collection half is already
here (`one_qubit_optimization.collapse_one_qubit_runs` collects a maximal
same-wire run). The synthesis half is blocked by arithmetic, not by a missing
routine:

> The determinant of a word over a target's arity-1 gates is the product of its
> factors' determinants. So a word over `{rz, sx, x}` — the three arity-1 gates of
> the `ibm-rz-sx-cx` snapshot — carries a determinant whose argument lies in
> `{0, pi/2, pi, 3*pi/2}` degrees, while a run containing `t`, `tdg`, `phase`,
> `u1`, or `u3` carries one outside that set.

Measured argument of the determinant, at a parameter of `1.1` rad, using this
repository's own runtime matrices:

| Opcode | `arg(det)` in degrees |
| --- | --- |
| `rz`, `rx`, `i` | `0.0000` |
| `t` / `tdg` | `45.0000` / `-45.0000` |
| `sx` / `sxdg` | `90.0000` / `-90.0000` |
| `x`, `y`, `z`, `h` | `180.0000` |
| `phase`, `u1` | `63.0254` — and it *moves with the parameter* |
| `u3`, `u2` | `126.0507` |

`phase` and `u1` are the only arity-1 gates whose determinant argument is their
parameter, which matters: a target that publishes one of them generates the whole
circle and blocks nothing. No shipped target basis does — measured over the five
snapshots, the blocked opcodes are `phase`, `t`, `tdg`, `u1`, `u2`, `u3` for
`ibm-rz-sx-cx` and `ibm-heron-cz`, 14 of the 18 declared arity-1 opcodes for
`rotational` and `ion-trap-rz-rx-rzz`, and `phase`, `u1`, `u2`, `u3` for
`clifford-t`.

Over 4000 seeded mixed runs drawn from all 18 declared arity-1 opcodes, run
lengths 2 to 8:

| Pair | Runs with a word up to a phase | Gates saved | Runs with an entry-for-entry word | Gates saved |
| --- | --- | --- | --- | --- |
| `rz` / `sx` | 4000 / 4000 | 3122 | 247 | 188 |
| `rz` / `rx` | 4000 / 4000 | 3123 | 230 | 193 |
| `phase` / `sx` | 4000 / 4000 | 3132 | 267 | 489 |
| `u1` / `rx` | 4000 / 4000 | 3132 | 425 | 529 |

The first column is the whole population and it is three gates cheaper on mean. The
second is bounded by the subgroup, which is why it is 247 and not 4000 — and the
gap between the columns is *exactly* the phase. Qiskit's pass takes the first
column because it can pay the difference into `dag.global_phase`; this repository
declines it because it has nowhere to pay it. That is the whole of the blocker.

The instrument and its contract are
[`benchmarks/compiler_one_qubit_decomposition.py`](../../benchmarks/compiler_one_qubit_decomposition.py)
and
[`tests/benchmark_contract/test_compiler_one_qubit_decomposition.py`](../../tests/benchmark_contract/test_compiler_one_qubit_decomposition.py),
both on this branch. Nothing in them reads a field that does not exist.

### Affected user journeys

1. **Compare a compiled program to an uncompiled one.** The repository's own
   evidence instruments compare statevectors directly. That comparison is already
   the acceptance test for `one_qubit_optimization`'s exactness policy, and it
   fails against a legalized program for a reason that is invisible in every
   reported number.
2. **Round-trip a Qiskit circuit.** `contracts/qiskit-interop-contract.toml`
   already claims `global_phase = "preserved"`. Today it is preserved into a
   mapping that execution ignores, so the claim holds only for a second conversion
   back to Qiskit.
3. **Deploy a training result.** A variational program whose final phase is part
   of the objective cannot be expressed at all, so the objective is
   unrepresentable rather than merely unoptimized.

## Evidence

Repository state: this branch, based on `origin/main` at `76a416a8`. The
measurements below are reproducible there with `benchmarks/`'s own modules; each
command was run in this checkout.

| Claim | Command | Result |
| --- | --- | --- |
| The determinant obstruction holds on the runtime matrices | `python -m benchmarks.compiler_one_qubit_decomposition --json-output <abs>` | `arg(det)` per opcode as tabulated; every entry has `abs(det) = 1`; `phase` and `u1` are the only arity-1 gates whose `arg(det)` moves with the parameter |
| The blocked opcodes per target basis are the ones outside its subgroup | same | `ibm-rz-sx-cx` and `ibm-heron-cz`: `phase, t, tdg, u1, u2, u3`; `rotational` and `ion-trap-rz-rx-rzz`: 14 of 18; `clifford-t`: `phase, u1, u2, u3` |
| Every run has a phase-blind word and few have an exact one | same, `declared_basis_reach` | 4000/4000 phase-blind vs 230–425 entry-for-entry, per pair |
| Legalization drops a non-trivial phase on nearly every program | same, `legalization_phase` | 76–78 of 80 per basis, worst amplitude gap `0.541`–`1.000`; `clifford-t` refuses all 80 |
| The minimal witness is two instructions | same, `phase_witness` | `h; cx` → `rz; sx; rz; cx`, phase `-45` degrees, gap `5.412e-01` |
| The existing metadata route is invisible to execution | `torch.equal` on the two statevectors with `global_phase = 0.5` in `metadata` | `True` — identical |
| `metadata` survives the pipeline | `optimize` and `legalize_native_gates` on a program carrying it | both return the mapping unchanged |
| A new key is refused by today's reader | `CircuitIR.from_dict` on `to_dict()` plus `global_phase` | `IRSerializationError: serialized CircuitIR contains unknown field(s): global_phase` |
| The contract pins the field this document is about | `grep -n global_phase contracts/qiskit-interop-contract.toml` | `global_phase = "preserved"` in the interop contract's policy block |
| The instrument's numbers are held in place | `python -m pytest tests/benchmark_contract/test_compiler_one_qubit_decomposition.py -q` | 7 passed |
| Qiskit records the phase this IR cannot | `qiskit==1.2.4`, `optimize_1q_decomposition.py` | `_gate_sequence_to_dag` sets `out_dag.global_phase`; `run(dag)` adds `best_circuit_sequence.global_phase` to `dag.global_phase` |

## Decision Candidates

**A. Add `global_phase: float` to `CircuitIR`, default `0.0`, with `IR_VERSION`
`1.1`, and make the statevector executor read it (chosen).** It is the smallest
change that removes the blocker, it is additive, and it makes a phase that is
already being lost observable to the ones losing it.

**B. Keep the phase in `metadata["interop"]["global_phase"]` and document it.**
Rejected. It is the present state, measured above to be inert: a free-form mapping
with no validation and no reader, which execution ignores. `metadata` is the right
home for provenance — where the program came from — and the wrong home for a value
that changes the answer. Promoting it would also make `metadata` a de-facto
contract while leaving it unversioned, which is the opposite of what an
interoperability claim needs.

**C. Carry the phase on an instruction, as a `phase` or `u1` gate on some wire.**
Rejected, and this is the candidate worth recording because it looks sufficient.
`phase` and `u1` have the exact matrix `diag(1, exp(i*theta))`, so a phase cannot
be carried on one wire without being a *relative* phase between that wire and the
rest — that changes the program. Putting it on a wire the program has spare is not
available in general, and it is not the identity operator on the other wires. As a
`u3` on an unused wire it becomes a real instruction the compiler may fold,
schedule, route, or draw, and a target that does not publish `u3` cannot be given
it. Measured above, `phase` is itself outside every target basis's arity-1
subgroup except through `one_qubit_synthesis`, so the "put it on a gate" route
routes the phase back into the synthesis that loses it.

**D. Compare programs up to a global phase instead of recording it.** Rejected.
This would rewrite the exactness policy in `one_qubit_optimization` and the tests
that pin it, converting an exactness guarantee into an equivalence-class
guarantee. It also cannot express a phase the *caller* supplied, so it changes the
meaning of a document rather than adding a field. It needs the same approval as A
and loses information rather than keeping it.

**E. Add the field but leave the executor unchanged.** Rejected. A field nothing
reads is the metadata route with more ceremony: it would let a pass write a phase
that still does not reach an amplitude, which is a new silent failure mode rather
than a fix.

## Prohibited Practices

1. Do not implement the writer in the same change. This proposal requests the
   field; the pass that needs it is W9-06 and lands separately, so a disagreement
   about the field cannot be hidden behind a pass that appears to work.
2. Do not update `IR_VERSION` to `1.1` in one location only. The eight contract
   files and `tools/check_*_interop_contract.py` read it from
   `flagquantum/core/ir.py`; update the reader contract by regenerating, never by
   hand-editing a snapshot to match.
3. Do not normalize the phase modulo `2*pi` in `__post_init__`. The product
   `exp(1j * (theta + 2*pi))` equals `exp(1j * theta)`, so normalizing is
   mathematically free — and it would make `content_hash` depend on which of two
   equal spellings a caller chose, which is a worse property than a redundant
   `2*pi`. Record what the caller stated.
4. Do not make the field an instruction. Candidate C is the route that looks
   local and is not.
5. Do not use the field for anything other than the program's operator. It is not
   a place for a calibration offset, a convention reminder, or a per-wire phase.
6. Do not weaken `tests/benchmark_contract/test_compiler_one_qubit_decomposition.py`
   to accommodate a different spelling of the phase. If the sign convention here
   is rejected, the instrument's direction and this document change together.

## Compatibility

- **`IR_VERSION` moves `1.0` → `1.1`.** A `1.0` payload is read as
  `global_phase = 0.0`, so the change is backward compatible for readers. It is
  not *forward* compatible: today's `from_dict` raises
  `IRSerializationError: serialized CircuitIR contains unknown field(s)` for an
  unrecognized key, measured above, so a `1.1` payload read by a `1.0` checkout
  fails closed. That is the correct failure, and it is why the version must move
  in the same change as the field rather than after it.
- **The serialized hash changes for every program, including on `global_phase =
  0.0`.** `content_hash` is the SHA-256 of `to_json()`, which serializes
  `sort_keys=True`, so an added key changes every hash even at the default value.
  Measured blast radius: 45 files under `tests/` reference `content_hash`; 9 lines
  across 6 files under `tests/` pin a literal 64-hex hash, three of which are
  circuit-derived and therefore move — `tests/unit/test_program_artifact_v2_candidate.py:189`
  and `tests/fixtures/program_artifact_v2_circuit_candidate.json` pin a circuit's
  `content_hash`, and `tests/fixtures/program_artifact_v1_compatibility.json`
  pins `expected_content_hash` over the serialized payload. The other three pin a
  manifest file's own SHA-256, a target snapshot id, and a task-plan identity;
  whether the last of those contains a program artifact is not measured here. 8
  files under `contracts/` record `ir_version`. `program-artifact-v2` declares
  `circuit_content_hash`, so a stored artifact's identity moves with it. The v1
  compatibility fixture exists precisely to freeze an old spelling, so an approved
  change must decide explicitly whether v1 artifacts are re-hashed or pinned —
  this document records the question and does not answer it.
- **No root export is added, removed, or renamed.** `docs/public_api_v1.json`
  freezes `CircuitIR` as a name, not its fields, so
  `tools/public_api_snapshot.py` is unaffected.
- **`contracts/qiskit-interop-contract.toml` line 19 (`global_phase =
  "preserved"`) becomes true through execution, not only through a second
  conversion.** When the field lands, the adapter should read and write it rather
  than `metadata["interop"]["global_phase"]`, which is then a compatibility
  duplicate that needs a named owner and a removal condition under engineering
  decision principle 1.
- **`flagquantum/simulation/statevector/**` gains one reader and one behavior
  change:** a program with a non-zero `global_phase` computes different amplitudes
  than it does today. Every program that exists today has the default `0.0`
  unless it carries the interop metadata, which is currently inert — so no
  existing result changes.

## Acceptance Tests

The change is verified when all of the following hold. They are stated as
acceptance criteria because this document does not implement the change.

- **The field defaults and round-trips.** `CircuitIR(n_wires=1,
  instructions=())` has `global_phase == 0.0`; `from_dict(to_dict())` is
  idempotent for `0.0`, `pi/4`, `-pi/4`, and `4*pi`, and `to_json` is stable
  across a second round trip.
- **A `1.0` payload is accepted.** A fixture serialized before the change loads
  with `global_phase == 0.0` and reports version `1.1`.
- **An unknown field still fails closed.** A payload with a key that is not
  `global_phase` still raises `IRSerializationError`, so the new key does not
  open the schema.
- **The executor reads it.** For the two-instruction witness above, a program with
  `global_phase = -pi/4` and the legalized instruction list reproduces the source
  statevector to `1e-12`, which is the acceptance test that the phase is no longer
  lost.
- **Nothing else reads it.** A source search shows one reader in
  `flagquantum/simulation/statevector/**`; the compiler, drawer, and ecosystem
  adapters do not read it in this change.
- **The instrument's contract still passes.** `python -m pytest
  tests/benchmark_contract/test_compiler_one_qubit_decomposition.py -q` — 7
  passed, with the two columns unchanged.
- **The contract gates agree.** `python tools/check_architecture.py`,
  `python tools/check_team_scope.py --validate`, and the eight
  `python tools/check_*_interop_contract.py` scripts pass with the regenerated
  `ir_version`.

## Open Questions

1. **Radians or turns?** Radians, matching `phase` and `u1` and Qiskit's own
   `global_phase`. Recorded because `u3`'s angles are also radians and the
   repository has no precedent for an angle-valued IR field.
2. **Does a stored artifact's `content_hash` need a migration, or is a v1
   artifact pinned forever?** `tests/fixtures/program_artifact_v1_compatibility.json`
   freezes one old hash deliberately, so the answer is not "regenerate the
   fixture". This is the largest unresolved cost in this proposal.
3. **Should `optimize` preserve the field, or fold it?** It must preserve it. The
   related hazard is the passes that already re-derive a product from matrices —
   `merge_adjacent_rotations`, the commuting-gap merge, and the fold — none of
   which may now *introduce* a phase, because a phase introduced there is a second
   writer by accident.
4. **Should the field's default be `0.0` or `None`?** `0.0` keeps every program
   total and makes `2.0 * pi` a legal spelled value; `None` would distinguish
   "unrecorded" from "recorded as zero", which is the distinction
   `native_gate_legalization` currently cannot make. This proposal chooses `0.0`
   and records the alternative.
5. **Is the Qiskit `metadata` duplicate removed in the same change or after it?**
   Engineering decision principle 1 requires a named owner and removal condition
   either way.

## Owner and approvals

Owner: FlagQuantum integration team (`integration` in `team-ownership.toml`,
which owns `flagquantum/core/ir.py` as a protected path).

Approval requested by this document: the field name `global_phase`, its unit
(radians), its type and default (`float = 0.0`), its place in
`to_dict`/`from_dict`, the `IR_VERSION` move to `1.1`, and the statevector
executor as its single reader.

Not requested here: the writer (`Optimize1qGatesDecomposition`, W9-06), the
`metadata` duplicate's removal, and any decision about stored artifacts'
`content_hash` migration (open question 2), which needs a separate answer because
it touches a released serialization identity.

Status of the companion change: the measurement that motivates this proposal is on
this branch and needs no approval, because it adds no public surface. This document
does not implement the field, and no code on this branch reads it.
