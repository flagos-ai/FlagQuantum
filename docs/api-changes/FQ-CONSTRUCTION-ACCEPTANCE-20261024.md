# Construction-layer acceptance: one program, not one program per route

## Decision and authorization

Status: **implemented on this branch, pending review.** This change adds no public
name, no keyword, no default, no result field, no opcode, no enum value, no
serialized key, and no version. It adds a contract, a gate, and a test whose
subject is *already-shipped behaviour*: what `Circuit.compose` builds must be
indistinguishable from what a user would have written by hand, in the IR and in
every execution mode that accepts the program.

Measured against this checkout:

- `fq.__all__` holds **36** names before and after, and the contract records that
  number so the gate fails if acceptance ever widens the root surface.
- `IR_VERSION` is `"1.0"` and `root_export_effect = "none"`, `ir_version_effect =
  "none"`.
- The construction layer on this tree is `{Circuit.compose, Circuit.adjoint}`.
  Measured: `hasattr(fq.Circuit(2), "compose")` and `hasattr(..., "adjoint")` are
  both true; `control`, `power`, and `__pow__` are all absent.

So this is written **without** a rule-8 authorization, and the contract says so in
a field (`authorization_required = false`) that the gate re-reads rather than
trusts. Acceptance is not an API change; it is the measurement that the API that
exists behaves as one program.

Two of those absent names are worth naming explicitly, because their absence is
branch-dependent rather than permanent. `Circuit.control` is delivered by
[#552](https://github.com/flagos-ai/FlagQuantum/pull/552) and `Circuit.power` by
[#542](https://github.com/flagos-ai/FlagQuantum/pull/542); neither is merged into
`main`. The contract records them under `[subject].not_covered` with that evidence,
**and the gate fails the moment either one appears in the tree** — so the first
person to merge those branches is told, by a failing gate, that this acceptance
must be extended to cover them. That is the intended behaviour: an acceptance that
silently stops covering a new member is worse than no acceptance.

`contracts/circuit-composition-contract.toml` also carries a row for these two
names, under `[scope].not_provided`, with the reason "no approved API change
proposal; control and power are unplanned, not partial". On this tree that row is
**true** and this change does not touch it. It becomes false when #552 or #542
merges, and the adjudication belongs to whichever of those merges first — see Open
Questions.

Scope of the affected surface: `contracts/construction-acceptance-contract.toml`
(new), `tools/check_construction_acceptance_contract.py` (new),
`tests/unit/test_construction_acceptance.py` (new), `.github/workflows/ci.yml`, and
`tools/pre_push.py`. `contracts/**`, `tools/**`, `.github/**`, and `tests/**` are
protected integration surfaces, so this change is classified **`integration`** and
verified as such:

```console
$ python tools/check_team_scope.py --team integration --files \
    contracts/construction-acceptance-contract.toml \
    tools/check_construction_acceptance_contract.py \
    tests/unit/test_construction_acceptance.py \
    .github/workflows/ci.yml tools/pre_push.py
team ownership policy passed
```

## Problem and affected user journey

### The plan row for this slice named three things that do not exist

The alignment plan records this slice as: *"after `compose`, the circuit agrees
across `statevector` / `mps` / `tensor_network` / `distributed_statevector` four
modes; `to_ir()`'s `n_qubits`, instruction order, and `semantic_fingerprint` are
identical to the hand-built circuit."*

Measured on this checkout, three of those identifiers are wrong:

| The row says | Measured | What it should say |
| --- | --- | --- |
| `CircuitIR.n_qubits` | `hasattr(ir, "n_qubits")` is `False`; the attribute is `n_wires` | `CircuitIR.n_wires` |
| `CircuitIR.semantic_fingerprint` | `hasattr(ir, "semantic_fingerprint")` is `False`; the name lives in `flagquantum/ecosystem/conformance.py` and the Qiskit adapter, not on the IR | `CircuitIR.content_hash` |
| `distributed_statevector` as a fourth mode | `ExecutionOptions(mode="distributed_statevector")` raises `ValidationError: mode must be one of: auto, density_matrix, mps, stabilizer, statevector, tensor_network` | a native run-mode alias, reachable through the runtime executor entry point |

None of these is a defect in the implementation. Each is a defect in the plan: the
row was written from prose about the four execution cores rather than read off the
code. This change records all three as `[[plan_corrections]]` rows, and the gate
**re-measures every one of them on every run**. A correction that stops being true
is a stale document, and a stale document that a gate does not notice is how a
plan quietly stops describing the repository.

The third correction has a second half worth stating plainly: the distributed
member is real, but neither root entry point can select it.
`inspect.signature(fq.run)` and `inspect.signature(fq.plan)` both lack
`world_size`, so `fq.run(c, world_size=2)` is a `TypeError`. The member is reached
through `flagquantum.runtime.execution.run_native(c, mode="distributed_statevector",
world_size=...)`, which returns the result and, on request, the plan that reports
`is_distributed=True`. Acceptance therefore cannot be written as "run the same
circuit in four modes through `fq.run`" — the fourth route has a different door,
and pretending otherwise would have produced a test that never touched the
distributed core at all.

### One word covered two different claims

"Agrees" hides two claims, and they are not the same claim:

1. the composed program and its hand-built twin agree with **each other**;
2. each mode agrees with the **reference mode**.

Claim 1 is exact. Claim 2 is not. Measured on a program that places a two-qubit
ladder and then applies three gates, every serving mode returns a vector that is
**bit-identical** between the two construction routes (`torch.equal` is `True` in
all five), while the modes differ among themselves at the last representable bits:

| Mode | hand vs. composed | vs. `statevector` |
| --- | --- | --- |
| `auto` | exact | `0.000e+00` |
| `statevector` | exact | `0.000e+00` |
| `density_matrix` | exact | `2.980e-08` |
| `mps` | exact | `1.490e-08` |
| `tensor_network` | exact | `2.980e-08` |
| `distributed_statevector` (ws 2, 4, 8) | exact | `8.429e-08` |

complex64 epsilon is `1.192e-07`, so the cross-mode spread sits below one unit in
the last place of the arithmetic everything here is stored in. An acceptance test
that asserted claim 2 exactly would fail on `mps` and `tensor_network` for reasons
that are arithmetic and not construction — and the tempting repair, comparing
everything in float64 or loosening until it passes, would have erased the
distinction this slice exists to draw. The contract therefore declares claim 1
`"exact"` and claim 2 `"tolerance"` **in two separate fields**, and the test asserts
them separately.

### Half of the mode set cannot answer the question at all

The plan's four modes are not four of the six `ExecutionOptions` modes.
`_MODES` is `{auto, density_matrix, mps, stabilizer, statevector, tensor_network}`;
`stabilizer` is the sixth, and it cannot serve a probabilities request:
`CapabilityError: mode='stabilizer' samples measurement outcomes; it cannot serve
measurement kind ...`. The contract records it under `[modes].refused` with the
class and the phrase, so the refusal is contracted rather than tolerated, and
`serving` plus `refused` must **partition** the measured mode set. A mode added
later fails that partition check instead of going unmeasured.

The right output to compare on is probabilities, not state. Both tensor modes
return `state = None`, so a state comparison would have silently compared nothing
on exactly the two modes whose agreement is least obvious.

### Affected user journeys

- *"I built my circuit by placing a subroutine with `compose`. Is it the same
  circuit?"* — answered exactly, on four observable IR surfaces, for eleven
  placements including the out-of-order, sparse-map, and single-qubit cases.
- *"Will my result change if the planner picks a different mode?"* — answered with
  a measured bound and a reason for it, instead of an unqualified yes.
- *"Can I run this sharded?"* — answered with the route that exists, including the
  fact that `world_size` is not a `fq.run` keyword.

## Evidence

The gate, on this checkout:

```console
$ python tools/check_construction_acceptance_contract.py
Construction acceptance contract passed: 36 exports, opcode census 35, IR_VERSION 1.0
```

The conformance test, `tests/unit/test_construction_acceptance.py`, 22 tests:

```console
$ python -m pytest tests/unit/test_construction_acceptance.py -q
22 passed in 0.63s
```

The IR-identity claim over the contract's eleven placements (`qubits` and
`qubit_map`), each expanded by hand and compared on `to_dict`, `content_hash`,
`n_wires`, and the `(name, qubits, params)` instruction signature:

```text
IR mismatches: 0     (8 qubits placements, 3 qubit_map placements)
instruction count == 2 * width - 1 in every case
```

The blast radius, `tests/unit` selected by `compose`, `adjoint`, `acceptance`,
`mode`, `distributed`, `tensor_network`, `statevector`, `plan`:

```text
5 failed, 1568 passed, 193 skipped, 3564 deselected
```

All five failures are pre-existing and unrelated to this change: `git status
--porcelain` shows this branch adds three untracked files and modifies two
configuration files, and no test file that fails is among them. Four are in
`tests/unit/test_statevector_batch_chunking.py` and one is
`tests/unit/test_native_cpu_adjoint.py::test_compact_cx_runtime_threshold_and_rollback`.

The repository gates re-run on this tree, all passing:
`check_circuit_composition_contract.py`, `check_primitives_admission_contract.py`,
`check_docs_links.py` (553 files), `check_architecture.py`,
`check_repository_hygiene.py`, `check_repository_language.py`, and the
`documentation-source-of-truth`, `operator-manifest`, `runtime-contract-schema`,
`public-api-baseline`, `legacy-root-api-usage`, and `correctness-certification`
hooks.

## Decision Candidates

**Candidate A — one gate, one test, and a contract that re-measures the plan's
mistakes.** *Chosen.* The contract owns the census and the deterministic IR claim,
because those are cheap and they are exactly what goes stale when someone adds a
construction member or a mode. The test owns the numerical claims, because they
need pytest's tolerance reporting to be readable when they fail. The three
`[[plan_corrections]]` rows make the plan's errors machine-checked rather than
corrected once in prose.

**Candidate B — a test only, no contract.** Rejected. The plan row's errors would
have been fixed in a document and nowhere else, and the next reader would have no
way to tell a corrected row from an unchecked one. It also gives up the fail-closed
property: a test that exercises `compose` cannot fail when a *new* construction
member appears, because it never looks at the member list.

**Candidate C — four modes through `fq.run`, as the plan wrote it.** Rejected as
literally impossible: the distributed member is not an `ExecutionOptions` value and
`fq.run` has no `world_size`. It is kept in the contract as a correction row
precisely so that the next person who reads the plan and tries it finds the
measured answer here instead of a `TypeError`.

**Candidate D — assert cross-mode equality in float64, where it holds.** Rejected.
It would pass, and it would be dishonest: it would report that five execution cores
agree exactly when what is true is that they agree to within the precision the
framework stores results in, after a promotion the user did not ask for.

## Prohibited Practices

- Do not widen `MODE_TOLERANCE` to make a failing mode pass. The tolerance is
  recorded in the contract with its measured rationale; a mode that moves outside
  it is a finding, not an inconvenience.
- Do not compare state vectors across modes. Two of them do not carry state.
- Do not delete a `not_covered` entry to make the census pass. The census exists to
  fail when the construction layer grows.
- Do not assert the acceptance as evidence of a *new* capability. It adds none.

## Compatibility

No public surface changes: no export, keyword, default, result field, enum value,
opcode, serialized key, or IR version. `docs/public_api_v1.json` and
`docs/generated/STABLE_API.md` are unaffected and the `public-api-baseline` hook
passes unchanged. The new contract and gate are additive to the repository's own
verification surface; the new test is additive to `tests/unit`. Nothing here
changes what `Circuit.compose` does.

## Acceptance Tests

- `tests/unit/test_construction_acceptance.py::test_a_composed_program_has_the_hand_built_ir`
- `tests/unit/test_construction_acceptance.py::test_an_adjoint_program_has_the_hand_built_inverse`
- `tests/unit/test_construction_acceptance.py::test_every_mode_is_measured_exactly_once`
- `tests/unit/test_construction_acceptance.py::test_the_two_construction_routes_agree_exactly_in_every_serving_mode`
- `tests/unit/test_construction_acceptance.py::test_each_serving_mode_agrees_with_the_reference_within_the_declared_tolerance`
- `tests/unit/test_construction_acceptance.py::test_the_distributed_member_serves_the_composed_and_hand_built_program_alike`
- `tests/unit/test_construction_acceptance.py::test_the_construction_layer_covers_exactly_what_the_contract_covers`

and the gate `tools/check_construction_acceptance_contract.py`, which fails on a
construction member that is present while contracted absent, a mode set that the
contract's partition does not equal, an IR-identity mismatch on any contracted
placement, a `[[plan_corrections]]` row whose name has appeared, an identifier the
contract measures that the gate does not read, and a requirement row naming a test
that does not exist.

## Open Questions

1. **Who adjudicates `circuit-composition-contract.toml`'s `not_provided` row?**
   #552 and #542 falsify it on their branches; on `main` it remains true. The
   cheapest correct answer is that whichever of the two merges first updates the
   row in that same PR, because that is the PR that makes it false. This change
   deliberately does not edit another slice's contract from a third branch.

2. **Should the distributed member be reachable from `fq.run`?** Today
   `fq.run(c, world_size=2)` is a `TypeError` and the native alias lives on
   `run_native`. That is a real gap in the user-facing surface — a user cannot
   select the distributed core from the documented entry point — but it is a
   *runtime* slice, not an acceptance slice, and it needs its own proposal because
   it adds a keyword to two Stable Core entry points.

3. **Is `2 * width - 1` the right contracted instruction count?** It is the block's
   own shape for the ladder the contract places. If a future placement test uses a
   block with a different shape, the field must move with it rather than be
   relaxed.

4. **Should the gate also measure the `stabilizer` refusal's full message?** It
   matches the contraction phrase today. The exact tail is a runtime message, and
   pinning it would make this gate fail on wording changes that are not semantic.

## Owner and approvals

Owner: team `integration` (protected contracts, `tools/**`, CI, and `tests/**`).
This change requires no API-owner approval because it changes no API; the
`authorization_required = false` field is re-read by the gate. Reviewer attention
is best spent on the two claims the contract separates — exact between the
construction routes, tolerance between the modes — and on whether the census
`not_covered` list is the right set to fail on.
