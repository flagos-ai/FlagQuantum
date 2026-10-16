# The primitive append forms are placements, and placement is `Circuit.compose`

## Decision and authorization

Status: **implemented on this branch, pending review.** This change adds no stable root
export, no opcode, no serialized key, no parameter name, and no default on an existing
signature. `fq.__all__` stays at **36** exports, `__all__` in
`flagquantum/algorithms/primitives/__init__.py` stays at **18**, the operator registry stays
at **35** opcodes, and `IR_VERSION` stays at `"1.0"`. Both public signatures in
`flagquantum/algorithms/primitives/qft.py` are unchanged:

```python
append_qft(circuit: Circuit, qubits: Sequence[int], *, inverse: bool = False) -> None
qft(n_qubits: int, *, inverse: bool = False) -> Circuit
```

No authorization under rule 8 of `AGENTS.md` is therefore required, and none is claimed.
The change is internal to one team's path: `python tools/check_team_scope.py --team
algorithms --files flagquantum/algorithms/primitives/qft.py
tests/unit/test_algorithms_primitives_placement.py` passes.

It is the construction half of the composition family. The numbered family decision is
[`API_CHANGE_PROPOSAL_066_CIRCUIT_COMPOSITION.md`](../development/API_CHANGE_PROPOSAL_066_CIRCUIT_COMPOSITION.md),
the placement semantics are owned by
[`FQ-CIRCUIT-COMPOSITION-20261002.md`](FQ-CIRCUIT-COMPOSITION-20261002.md), and inversion is
owned by [`FQ-CIRCUIT-ADJOINT-20261003.md`](FQ-CIRCUIT-ADJOINT-20261003.md). This record
settles what that pair left open for the `algorithms` layer: whether a primitive that writes
into a caller's circuit builds the sub-program itself or *places* one.

## Problem and affected user journey

### The transform was constructed twice

`qft` and `append_qft` described the same transform with two independent emissions. `qft`
built a circuit on a fresh register; `append_qft` hand-rolled a private step language
(`_Hadamard`, `_ControlledPhase`, `_Swap`, and a `_Step` union), a `sequence.reverse()` for
the inverse, and its own gate-by-gate emission with the caller's qubit numbers substituted
directly. The two agree — measured below, in every order and both directions — but they agree
by coincidence of maintenance rather than by construction.

That is engineering-decision principle 6 ("do not create a second source of truth") stated
against this file. `FQ-CIRCUIT-COMPOSITION-20261002.md` already recorded the intended
relation as an *expected* outcome rather than a fact:

> `fq.Circuit(5).h(0).compose(qft(3), qubits=(1, 2, 3)).to_ir().instructions` | equal,
> instruction by instruction, to `fq.Circuit(5).h(0)` followed by
> `append_qft(manual, [1, 2, 3])`

and then, in its Prohibited Practices, deferred the change: "**Do not deprecate `append_qft`
in this change.** It is the in-place form the algorithms already use." That deferral is what
this change closes, without deprecating anything.

### The inverse was a second reversed emission

`append_qft(..., inverse=True)` reversed the step list and negated each angle. A unitary's
inverse is its adjoint, `Circuit.adjoint` exists, and `qft(n).adjoint()` already reproduces
`qft(n, inverse=True)` exactly — so the reversed emission was a second expression of a
relation the repository already has a primitive for.

### A refused placement wrote half a transform

`append_qft` emitted as it walked. A qubit outside the receiver's range failed on the gate
that first reached it, after earlier gates had already been appended. Measured on the
pre-change head, with `c = fq.Circuit(2).x(0)` (one instruction):

| Call | Result | Instructions left in `c` |
| --- | --- | --- |
| `append_qft(c, (5,))` | `ValidationError: Gate 'h' references qubit(s) (5,) outside circuit range [0, 1].` | 1 |
| `append_qft(c, (1, 5))` | `ValidationError: Gate 'cphase' references qubit(s) (5,) outside circuit range [0, 1].` | **2** |
| `append_qft(c, (5, 1))` | `ValidationError: Gate 'h' references qubit(s) (5,) outside circuit range [0, 1].` | 1 |

The middle row is the defect: the circuit was left holding a Hadamard that the caller never
asked for and cannot see, because the failure arrived from inside the emission.

## Evidence

Every figure below is reproduced by the commands in
[Acceptance Tests](#acceptance-tests) on this branch.

### The equivalence is checked, and it held before the change

The first table is the claim, measured on the pre-change head and again after it: the
placement form and the append form emit the same instructions, with the same parameters, in
the same order.

| Register | `inverse` | Equal before | Equal after | Instructions |
| --- | --- | --- | --- | --- |
| `(0,)` | `False` / `True` | yes | yes | 2 |
| `(0, 1)` | `False` / `True` | yes | yes | 5 |
| `(0, 1, 2)` | `False` / `True` | yes | yes | 8 |
| `(1, 2, 3)` | `False` / `True` | yes | yes | 8 |
| `(3, 1, 4)` | `False` / `True` | yes | yes | 8 |
| `(2, 0)` | `False` / `True` | yes | yes | 5 |
| `(0, 2, 5)` | `False` / `True` | yes | yes | 8 |
| `(0, 1, 2, 3, 4, 5)` | `False` / `True` | yes | yes | 25 |

A register of width `n` emits `n` Hadamards, `n(n-1)/2` controlled-phase gates, and `n//2`
swaps, in that order: `n + n(n-1)/2 + n//2`. The three narrow columns sum to the fourth
column exactly, which is the arithmetic the test asserts rather than a recorded count.

### The inverse is the adjoint

`qft(n, inverse=True)` is instruction-identical to `qft(n).adjoint()` for every
`n` in `1..6`. At `n = 1` and `n = 2` this is a real statement rather than a trivial one: the
forward transform of two qubits already contains its own reversal (`swap(0, 1)`), so the
adjoint reproduces that swap instead of cancelling it.

### The transform is still the transform

Applying the forward transform to a basis state reproduces the discrete Fourier transform
amplitudes to `1e-6` in `complex64`, and a forward-then-inverse round trip returns the
register to its starting amplitudes to `1e-6`. These are the scenario tests the module always
needed; the equivalence tests alone would pass on a consistently wrong transform.

### The placement is atomic

The same three calls as the pre-change table now leave the receiver untouched: the placement
names its target qubits and is refused before any instruction is written. The refusal comes
from `Circuit.compose`, and the message names the placement rather than the first gate that
happens to arrive:

```text
ValueError: Circuit.compose target qubit(s) (5,) outside circuit range [0, 1].
```

Widening the register is not needed to make the change: the refusal text changed, and the
guarantee that the receiver is untouched is new. Both are improvements, and both are recorded
here rather than presented as unchanged behaviour.

## Decision Candidates

**A. `append_qft` composes `qft`, and `qft`'s inverse is its adjoint (chosen).** The
transform is constructed once, in `_qft_circuit`. `append_qft` builds nothing: it validates
the register and calls `circuit.compose(qft(n, inverse=inverse), qubits=ordered)`. The
inverse is `forward.adjoint()`. The user-visible signatures do not move.

**B. Keep the private step language and pin the equivalence with a test only.** Rejected:
a test that pins two constructions to agree is a permanent maintenance cost paid to preserve
a duplicate. The equivalence already holds, so the duplicate can be removed rather than
watched.

**C. Build the transform at the receiver's batch size so a batched receiver keeps working.**
Rejected. `compose`'s published rule is that two programs whose `bsz` disagrees are refused
rather than reconciled — "**Do not silently rebroadcast a batch dimension**" in
`FQ-CIRCUIT-COMPOSITION-20261002.md`. Passing the receiver's `bsz` into the sub-program would
make the same transform two different programs depending on who placed it, and would give
`qft` a batch dimension that has no meaning for a fixed unitary. The measured consequence is
recorded under [Compatibility](#compatibility) instead.

**D. Give `qft`/`append_qft` the same treatment for the other three primitives in this
change.** Rejected as scope. `append_arbitrary_state`, `append_bit_oracle`, and
`append_phase_estimation` each already derive their standalone builder from the append body,
so the single construction runs the other way and there is no duplicate to remove. What they
need is the *claim* checked, which is what the new test file does for all four.

## Prohibited Practices

- **Do not add a second emission of a primitive.** If a primitive needs to appear both as a
  standalone builder and as an append form, one of the two composes the other. The
  `qft`/`append_qft` pair is the reference: the standalone builder is the body, the append
  form is a placement.
- **Do not deprecate or remove `append_qft`.** It is what `append_phase_estimation` calls,
  and it is the in-place form the algorithms use. Removing it is a separate decision with its
  own migration; `FQ-CIRCUIT-COMPOSITION-20261002.md` already forbids it and this record does
  not revisit that.
- **Do not re-implement the inverse as a reversed emission.** `Circuit.adjoint` owns
  inversion, and a reversed emission drifts the moment either copy changes.
- **Do not special-case `compose`'s batch rule inside a primitive.** A primitive that needs a
  different placement rule has found a defect in the placement contract and should say so
  there, where every other consumer sees it.
- **Do not add a qubit or a gate to make a refusal go away.** `append_qft` allocates nothing;
  the receiver's width and its qubit numbering are the caller's decision.
- **Do not keep a private step union to preserve a "fragment" representation.** The plan's
  vocabulary is composition, not fragments. No second representation of a program is admitted
  into the IR or into a module.

## Compatibility

| Surface | Before | After |
| --- | --- | --- |
| `fq.__all__` | 36 | 36 |
| `primitives.__all__` | 18 | 18 |
| `append_qft` / `qft` signatures | as stated above | identical |
| Operator registry | 35 opcodes | 35 opcodes |
| `IR_VERSION` | `"1.0"` | `"1.0"` |
| Emitted instructions, valid placement | — | **identical**, all 16 register/direction pairs |
| Refusal for a qubit outside the receiver | `ValidationError` from the first gate that reached it, after earlier gates were written | `ValueError` from the placement, nothing written |
| Batched receiver (`bsz != 1`) | accepted, instructions appended | `ValueError: Circuit.compose cannot mix batch sizes: this circuit has bsz=4, the composed program has bsz=1.` |

The last row is the one behavioural change in this record, and it is a consequence the
decision candidates above chose deliberately rather than an accident. A batched circuit was
never a batched *transform*: `append_qft` wrote unbatched gates into it, and the batch
dimension belonged to the receiver all along. Callers who need the transform inside a batched
program place it in an unbatched circuit and compose that, or build the batched circuit
around the transform. The refusal names both batch sizes, so the fix is visible from the
message.

No exception type was widened: `Circuit.compose` raises `ValidationError`, which is a
`ValueError`, so a caller catching `ValueError` as the docstrings always said still catches
it. `ValidationError` is exported from `flagquantum` as `fq.ValidationError`.

## Acceptance Tests

```console
$ python -m pytest tests/unit/test_algorithms_primitives_placement.py -q
35 passed

$ python -m pytest tests/unit/test_algorithms_qft.py \
    tests/unit/test_algorithms_phase_estimation.py \
    tests/unit/test_algorithms_state_preparation.py \
    tests/unit/test_algorithms_oracle.py \
    tests/unit/test_circuit_adjoint.py \
    tests/unit/test_circuit_composition_contract.py \
    tests/unit/test_primitives_admission_contract.py -q
199 passed

$ python -m tools.check_primitives_admission_contract
Primitives admission contract passed: 18 exports, 9 admitted as public units

$ python -m tools.docs_source_of_truth --check
$ python -m tools.operator_manifest --check
$ python -m tools.runtime_contract_schema --check
$ python tools/check_docs_links.py
Checked 553 markdown files; all local links and anchors resolve.
$ python -m tools.public_api_snapshot
public API migration baseline passed
$ python tools/check_architecture.py
architecture boundaries passed
$ python tools/check_repository_hygiene.py
repository hygiene passed
$ python -m tools.correctness_certification --check
$ python tools/check_repository_language.py
$ python tools/check_legacy_root_api_usage.py
$ python tools/check_team_scope.py --team algorithms --files \
    flagquantum/algorithms/primitives/qft.py \
    flagquantum/algorithms/primitives/README.md \
    tests/unit/test_algorithms_primitives_placement.py \
    docs/api-changes/FQ-ALGORITHMS-PRIMITIVES-PLACEMENT-20261023.md
team ownership policy passed
```

The blast-radius selection `-k "qft or primitive or adjoint or compose or composition or
oracle or phase_estimation or state_preparation or arbitrary or algorithm"` over `tests/unit`
reports `1 failed, 643 passed, 147 skipped, 4551 deselected`. The single failure is
`tests/unit/test_native_cpu_adjoint.py::test_compact_cx_runtime_threshold_and_rollback`,
which fails on the pre-change head with this change stashed. It is pre-existing on Darwin and
is not attributable to this change.

## Open Questions

1. **Should `qft` gain a batch size?** Candidate C above declines it for this change. If a
   batched transform is a real user need, it is a new public parameter on a protected
   signature and needs its own rule-8 authorization and its own record.
2. **Should `append_qft`'s empty register stay silent?** `append_qft(c, ())` returns `None`
   and emits nothing, while `qft(0)` raises `ValueError`. The asymmetry is preserved here
   unchanged, because the plan's row is a construction refactor and the empty-register case is
   a validation question with its own evidence burden. It is recorded rather than settled.
3. **Do the other three primitives want the same treatment?** Candidate D declines it as
   scope, and the new test file states the direction their single construction runs in. If a
   duplicate appears in `state_preparation.py` or `oracle.py`, this record's rule applies to
   it: one of the two forms composes the other.

## Owner and approvals

Owner: `algorithms`. Shared surface: `tests/unit/test_algorithms_primitives_placement.py`
under the repository's `shared_paths` rule. Placing contract semantics are owned by `core`
through `Circuit.compose`; this change consumes them and adds none. No stable export,
signature, default, serialized schema, or opcode is added, removed, renamed, reordered, or
changed, so no authorization is requested under rule 8 and none is asserted.
