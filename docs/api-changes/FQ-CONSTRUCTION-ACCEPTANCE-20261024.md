# Construction-layer acceptance: one program, not one program per route

## Decision and authorization

Status: **implemented on this branch, pending review.** This change adds no public
name, no keyword, no default, no result field, no opcode, no enum value, no
serialized key, and no version. It adds a contract, a gate, and a test whose
subject is *already-shipped behaviour*: what `Circuit.compose` builds must be
indistinguishable from what a user would have written by hand, in the IR and in
every execution mode that accepts the program.

Measured against the pre-merge checkout this was first written on:

- `fq.__all__` held **36** names before and after, and the contract records that
  number so the gate fails if acceptance ever widens the root surface. It reads
  **37** once this branch is merged with `main`, because the `density_matrix`
  output — an unrelated slice — added an export of its own. The contract's 36 was
  a true measurement of the tree it was written on and a stale one of the merged
  tree, which is why the gate re-reads the number instead of trusting it.
- `IR_VERSION` is `"1.0"` and `root_export_effect = "none"`, `ir_version_effect =
  "none"`.
- The construction layer on that tree was `{Circuit.compose, Circuit.adjoint}`.
  Measured: `hasattr(fq.Circuit(2), "compose")` and `hasattr(..., "adjoint")` were
  both true; `control`, `power`, and `__pow__` were all absent. Both of the first
  two absent members have since arrived and are covered — see *The census fired
  twice* — so the merged layer is `{compose, adjoint, control, power}` and only
  `__pow__` remains contracted absent.

So this is written **without** a rule-8 authorization, and the contract says so in
a field (`authorization_required = false`) that the gate re-reads rather than
trusts. Acceptance is not an API change; it is the measurement that the API that
exists behaves as one program.

Two of those absent names were worth naming explicitly, because their absence was
branch-dependent rather than permanent. `Circuit.control` is delivered by
[#552](https://github.com/flagos-ai/FlagQuantum/pull/552) and `Circuit.power` by
[#542](https://github.com/flagos-ai/FlagQuantum/pull/542). The contract recorded
both under `[subject].not_covered` with that evidence, **and the gate failed the
moment either appeared in the tree** — which is what happened twice, once when
#552 merged and once when #542 did, each time against a branch that was still
open. That is the intended behaviour, and it is the only reason the extensions
below exist at all: an acceptance that silently stops covering a new member is
worse than no acceptance. Both members are now covered, `[subject].not_covered`
is empty and asserted as empty, and only `__pow__` remains under
`[census].also_measured_absent`.

`contracts/circuit-composition-contract.toml` also carried a row for these two
names under `[scope].not_provided`, with the reason "no approved API change
proposal; control and power are unplanned, not partial". That row was **true** of
the tree this record measured, and it is no longer: `N1-4` removed `control` from
it in its own PR, which is the adjudication Open Question 1 asked for, and `N1-5`
removed `power` the same way. This change still does not edit another slice's
contract; it re-reads the census against it, and that contract now declares the
family complete — which is why the emptiness of `not_covered` above is checked
against it rather than asserted on its own.

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

## The census fired twice

This record was written against a tree where `Circuit.control` did not exist, and its
census wrote that absence down as a contracted fact: `[subject].not_covered` held
`control`, and the gate fails if a name contracted absent becomes present. `N1-4`
([#552](https://github.com/flagos-ai/FlagQuantum/pull/552)) then merged `Circuit.control`,
and this slice's branch, which was still open, was merged with `main` and the gate failed
with exactly the sentence it was built to say:

```console
$ python tools/check_construction_acceptance_contract.py
construction member 'Circuit.control' is contracted as absent but is present; the
acceptance must be extended to cover it
measured_root_export_count is stale: contract says 36, fq.__all__ has 37
```

The second line is the same mechanism reading a different fact: `control` added no root
export, so the 36 in this contract is not wrong about this slice, it is stale about the
repository. Both are the gate doing the job, and both are repaired here rather than
suppressed.

The gate then said the same sentence a second time, about the other member it had
contracted absent. `N1-5` ([#542](https://github.com/flagos-ai/FlagQuantum/pull/542))
merged `Circuit.power`, and the merge of `main` into this branch, which by then carried
the `control` extension, failed again:

```console
construction member 'Circuit.power' is contracted as absent but is present; the
acceptance must be extended to cover it
```

Two firings, one mechanism, and the second is the better evidence of the two: it is the
mechanism working on a member this record had never written an extension for, arriving
through a merge rather than through a plan. The root-export line did not appear the second
time — `power` adds no export either, and 37 was already the measured number.

### The third firing, and why it has nothing to do with the construction layer

The root-export line came back on `main` on its own, with no merge into this branch
involved and no construction-layer change anywhere in it:

```console
$ python tools/check_construction_acceptance_contract.py
measured_root_export_count is stale: contract says 37, fq.__all__ has 40
```

`37` was a true measurement of the tree that wrote it, and it stopped being one when
`N3-6` added three root exports — `fq.jacobian`, `fq.jvp`, and `fq.vjp` — and raised
`contracts/public-api-v1-candidate.json`'s `root_export_budget` from 36 to 39 with its own
record. That slice did not re-read this contract's census, so the repository carried two
numbers for one fact and the `quality` job failed on this step for every open pull request
that merged `main`, including pull requests that touch no construction-layer file at all.

The count is re-measured to `40` here. What is *not* changed is the claim this contract
makes: `root_export_effect = "none"` and `authorization_required = false` still hold, and
neither `control` nor `power` added an export between them. The number is a reading of a
repository fact that other slices own, which is exactly why the gate re-reads it instead of
trusting the line — and why a repair here is a re-measurement rather than a relaxation.
A number that only ever moves when this contract's own slice moves would be the
assertion-shaped version of the same fact, and a weaker one.

### The decision this forced

`Circuit.control` is now `[subject].covered`, and it is measured by the claim this
contract exists for rather than by a restatement of what it means:

```text
routes = ["compose then control", "hand-built placement then control"]
```

The receiver is the same `h`/`cnot` ladder the placements place, `compose` places it, and
both routes are then controlled. If `control` read anything about its receiver other than
the receiver's instructions — its provenance, a bookkeeping field, an internal ordering —
or if `compose` left something in the receiver that a hand-written receiver does not have,
the two programs would differ. Four measured rows:

```text
receiver 2  qubits=(0, 1)     n_controls=1  ctrl_qubits=[2]     -> 3 qubits,  7 instructions
receiver 3  qubits=(2, 0)     n_controls=1  ctrl_qubits=[3]     -> 4 qubits,  7 instructions
receiver 5  qubits=(4, 0, 2)  n_controls=1  ctrl_qubits=[5]     -> 6 qubits, 11 instructions
receiver 4  qubits=(3, 1)     n_controls=2  ctrl_qubits=[5, 4]  -> 6 qubits, 57 instructions

to_dict mismatches: 0   content_hash mismatches: 0
n_wires mismatches: 0   instruction-signature mismatches: 0
```

`compose` then `control` and the hand-built placement then `control` are the same program
on all four observable surfaces, exactly, on every row. The width is `max(ctrl_qubits) + 1`
on every row, which is what "a control qubit is added rather than taken" means when it is
measured instead of quoted.

`Circuit.power` is now `[subject].covered` as well, and is measured by the same claim with
the routes spelled for it:

```text
routes = ["compose then power", "hand-built placement then power"]
```

`power` is the sharpest of the four members for this claim, because it is the only one that
*derives* instructions from its receiver rather than copying them out. A `power` that read
provenance, or a `compose` that left a trace in the receiver, would show up here before it
showed up anywhere else. Six rows, two of which exist to close the ways the claim could pass
for the wrong reason:

```text
receiver 2  qubits=(0)     exponent=0   -> 2 qubits,  0 instructions   (empty program)
receiver 2  qubits=(0)     exponent=1   -> 2 qubits,  1 instruction    (the copy case)
receiver 2  qubits=(0)     exponent=2   -> 2 qubits,  2 instructions
receiver 3  qubits=(1)     exponent=2   -> 3 qubits,  2 instructions
receiver 3  qubits=(2, 0)  exponent=3   -> 3 qubits,  9 instructions
receiver 4  qubits=(0, 1, 2)  exponent=-1 -> 4 qubits,  5 instructions (the inverse)

to_dict mismatches: 0   content_hash mismatches: 0
n_wires mismatches: 0   instruction-signature mismatches: 0
```

Three things are asserted beyond the route identity, because a route that returned its
receiver would satisfy an equality of two routes if its twin were equally wrong:

- the width is the receiver's own on every row — `power` repeats or rewrites a program, it
  never adds a qubit, unlike `control`, which adds one;
- the instruction count is `abs(exponent) * (2 * width - 1)` on every row, which is the
  ladder's own shape repeated rather than a number copied from a run. The `h`/`cnot` ladder
  is in the count because neither gate declares the exponential form the single-instruction
  rewrite applies to, so every non-zero exponent emits whole copies;
- `exponent = 0` is in the list on purpose: it is the empty program, so it is where a
  comparison written the wrong way round passes trivially — and `exponent = 1` is in the
  list for the mirror reason, because the identity power copies rather than rescaling and
  must not grow the program.

The one case the contracted block cannot supply is the rewrite itself, so it is measured in
the conformance file rather than by the gate and named in the contract as such: a
single-instruction receiver whose gate declares an exponential form is rewritten as that
gate with the parameter multiplied by the exponent, while `power(1)` stays a copy. Both
routes hold the same `rx`, so a `power` that consulted provenance would rewrite one of them
differently; measured with a tensor angle carrying `requires_grad`, `power(1)` hands back
the same tensor object and the wider exponents hand back a scaled one.

The mode half is measured here too, and pinned the same way it is for `control`: `power` is
placed among the serving modes on the row carrying `mode_agreement = true`, which the gate
requires to be a repeated program. `power(0)` is empty and `power(1)` is its receiver, so
neither of those exponents could tell a route that returned its input apart from one that
built the program, and neither is allowed to carry the flag.

The rewrite test the contract names under
`power_acceptance.single_instruction_rewrite_measured_in` is checked the same way a
`[verification].requirements` row is: the gate reads the conformance file's test names and
fails if the named test is not in it. A field that said "measured elsewhere" while pointing
at a test that no longer exists would be the same failure this record is about, one level
up.

### What was deliberately *not* asserted

The mode half of the claim — that the two routes stay one program once a mode serves them
— is measured on the three one-control rows and **not** on the two-control row, and the
reason is recorded in the contract rather than left in this document:

```text
one-control rows, widest spread against the statevector reference: 5.960e-08   (0.50 x eps)
two-control row, widest spread against the statevector reference: 8.941e-07   (7.50 x eps)
declared MODE_TOLERANCE:                                          1.000e-06   (8.39 x eps)
```

The two-control ladder is 57 instructions, and the spread it reaches is 75 percent of the
declared bound. Asserting the mode half on that row would have been a statement about the
platform's arithmetic — one BLAS revision away from red — dressed up as a statement about
the construction layer. The alternative was to raise `MODE_TOLERANCE` to cover a program
this acceptance chose to add, which the Prohibited Practices below forbid, so the
measurement is pinned to rows the declared bound clearly covers and the wider number is
recorded here instead. `mode_agreement = true` marks those rows in the contract, and the
gate refuses a marked row with `n_controls != 1`, so the pinning is a checked fact and not
a comment.

The two routes are still compared **exactly** on all four rows in every serving mode,
including the two-control row. That comparison is between two runs of the same program and
does not depend on the bound, so it costs nothing to make it on the widest row as well.

### What the census could not see, measured

The paragraph above ends with an agreement between two documents: `power` "is the same
single entry on both sides". That was true, and it was checked by reading. It is now
checked by the gate, because the reading was shown to be insufficient -- and the second
firing above is what "insufficient" meant in practice: the agreement was true and the
member was still uncovered.

Five mutations of this contract were run, each one a different way of walking a
contracted member backwards, and the gate's verdict on each was recorded:

```text
control moved back to not_covered, dropped from covered            -> fails, names the member
measured_root_export_count 37 -> 36                                -> fails, names both numbers
control dropped from covered, not_covered untouched                -> fails: requirement names
                                                                      absent member
control dropped from covered AND its requirement row deleted       -> PASSED (gate rc=0,
                                                                      conformance 26 passed)
mode_agreement added to the two-control row                        -> fails: one-control only
```

The fourth row is the finding. `[verification].requirements` and `[subject].covered` are
checked against each other -- a requirement naming a member that is not covered is an
error -- but nothing checked that every member of the family appears in one of the two
lists. Deleting a row from both sides is therefore consistent with itself, and the census,
whose whole job is to re-measure the rows the contract has, never noticed that it had one
fewer row to re-measure. The conformance file stayed green for a second reason that is
worth writing down: `test_a_controlled_program_has_the_hand_built_ir` was still in it and
still passing. The test had not been deleted or weakened. It had stopped being required by
anything, which is indistinguishable from passing.

The same mutation was re-run against the `power` extension, and the fourth row of that
table is now the first row of this one: the repair holds for a member added after it.

```text
power moved back to not_covered, dropped from covered       -> fails: names the member, and
                                                              names its requirement
power dropped from covered, not_covered untouched           -> fails: not the family, the
                                                              spelling row, the requirement
power dropped from covered AND its requirement deleted      -> fails: not the family, and the
                                                              spelling row
mode_agreement added to the exponent=1 row                  -> fails: repeated programs only
no power row carries mode_agreement                         -> fails: measured on nothing
a power_placement row repeats a qubit                       -> fails: the receiver repeats
a power_placement row sits outside the receiver             -> fails: targets outside it
also_measured_absent emptied                                -> fails: the spelling loses its row
```

The third row is the point of the paragraph above, measured a second time: the mutation
that passed before the repair now fails, and it fails for two independent reasons rather
than one, because `[census].also_measured_absent` also names the member its spelling
belongs to. `Circuit.__pow__` staying absent is what keeps that second reason alive.

This is the same class of failure the mode half of the contract already guards against:
`serving` plus `refused` must equal the measured mode set, not merely be disjoint from
each other. That check exists because a mode added later would otherwise go unmeasured.
The family needed the identical check and did not have it.

The repair states the family once. `[census].family_declaration_source` points at
`contracts/circuit-composition-contract.toml` -- the contract that owns what these members
mean -- and `family_declaration_keys` names the two fields that declare the family.
`covered` plus `not_covered` must equal that declaration:

```console
$ python tools/check_construction_acceptance_contract.py     # control + its requirement deleted
the construction family this acceptance covers is not the family
'contracts/circuit-composition-contract.toml' declares:
only here [], only there ['Circuit.control']
```

Two smaller things came out of the same measurement. `[census].operator_spellings` declares
that `Circuit.__pow__` is how Python spells `Circuit.power`, so `power` is one family member
under two names and the partition counts it once; the gate refuses a declared spelling that
no row measures, because that is a member it has stopped looking at while this contract
still claims to look at it. And the gate's own `CENSUS_ATTRIBUTES` tuple was deleted: the
names it measures are now read from the contract, so the family exists in one place instead
of two. A second list inside the gate would have been the list that survived a member being
removed from the contract.

The limit of this, stated rather than implied: the gate cannot know which dunder spellings
exist without being told, so a change that deletes the `also_measured_absent` row *and* the
`operator_spellings` entry together would stop measuring `Circuit.__pow__`. Teaching the
gate about dunders would put a second copy of the family back inside it, which is the thing
this repair removes. What still holds in that case is the part that matters most: `power`
remains censused by name against the declaration, and the declaration itself cannot be
emptied, because a source that reads no members is refused rather than treated as an empty
family.

### Open Question 1, answered

The first Open Question below asked who adjudicates
`circuit-composition-contract.toml`'s `[scope].not_provided` row for `control` and
`power`. The answer is that each slice did it in its own PR, which is the cheapest of the
answers considered here: `N1-4` removed `control` from that row and `N1-5`
([#542](https://github.com/flagos-ai/FlagQuantum/pull/542)) removed `power`, and this
contract's census now agrees with both. That contract's `[scope].not_provided` is empty and
its `[scope].provided` lists all four members, so this contract's `covered` lists all four
and its `not_covered` is empty — a fact the gate checks against the declaration rather than
accepts as prose, which is why the second firing's repair could not be written by deleting
the row.

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
Construction acceptance contract passed: 37 exports, opcode census 35, IR_VERSION 1.0
```

The conformance test, `tests/unit/test_construction_acceptance.py`, 33 tests:

```console
$ python -m pytest tests/unit/test_construction_acceptance.py -q
33 passed in 1.30s
```

The IR-identity claim over the contract's eleven placements (`qubits` and
`qubit_map`), each expanded by hand and compared on `to_dict`, `content_hash`,
`n_wires`, and the `(name, qubits, params)` instruction signature:

```text
IR mismatches: 0     (8 qubits placements, 3 qubit_map placements)
instruction count == 2 * width - 1 in every case
```

and the same claim over the two members that were added to it later: four control
placements, six power placements, all four observable surfaces exact on every row, with
the width and instruction-count rules the contract states re-measured on each row rather
than quoted.

The blast radius after the merge, `tests/unit` selected by `compose`, `adjoint`,
`acceptance`, `mode`, `distributed`, `tensor_network`, `statevector`, `plan`:

```text
5 failed, 1599 passed, 195 skipped, 3783 deselected
```

All five failures are pre-existing, and this time that is **proven** rather than
argued: the same selection run in a detached worktree at `origin/main`
(`0ebc710b2`) fails the same five tests with the same names. Four are in
`tests/unit/test_statevector_batch_chunking.py` and one is
`tests/unit/test_native_cpu_adjoint.py::test_compact_cx_runtime_threshold_and_rollback`.
None is in a file this change touches.

The repository gates re-run on this tree, all passing:
`check_circuit_composition_contract.py`, `check_primitives_admission_contract.py`,
`check_docs_links.py` (562 files), `check_architecture.py`,
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
  fail when the construction layer grows. Extending it to *cover* a member that
  arrived, as the addendum above does, is the other thing entirely — and it is not
  satisfied by deleting the entry, because `[verification].requirements` then has
  no test for the new member and the gate fails on that instead.
- Do not raise `MODE_TOLERANCE` to bring a controlled program inside the bound.
  The bound is a property of the arithmetic; a program this acceptance chose to
  add does not get to move it. Pin the measurement to a program the bound already
  covers and record the wider number, which is what the addendum does.
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
- `tests/unit/test_construction_acceptance.py::test_a_controlled_program_has_the_hand_built_ir`
- `tests/unit/test_construction_acceptance.py::test_the_control_placements_in_the_contract_are_well_formed`
- `tests/unit/test_construction_acceptance.py::test_the_controlled_program_is_one_program_in_every_serving_mode`
- `tests/unit/test_construction_acceptance.py::test_a_controlled_program_is_refused_by_the_refused_mode_the_same_way`
- `tests/unit/test_construction_acceptance.py::test_a_powered_program_is_the_hand_built_power`
- `tests/unit/test_construction_acceptance.py::test_the_power_placements_in_the_contract_are_well_formed`
- `tests/unit/test_construction_acceptance.py::test_a_powered_single_instruction_receiver_is_rewritten_by_both_routes_alike`
- `tests/unit/test_construction_acceptance.py::test_a_powered_program_is_one_program_in_every_serving_mode`
- `tests/unit/test_construction_acceptance.py::test_the_construction_layer_covers_exactly_what_the_contract_covers`
- `tests/unit/test_construction_acceptance.py::test_the_census_is_a_partition_of_the_declared_family`
- `tests/unit/test_construction_acceptance.py::test_the_census_rejects_a_member_deleted_from_the_family`
- `tests/unit/test_construction_acceptance.py::test_the_census_rejects_a_spelling_that_no_row_measures`

and the gate `tools/check_construction_acceptance_contract.py`, which fails on a
construction member that is present while contracted absent, a mode set that the
contract's partition does not equal, an IR-identity mismatch on any contracted
placement, a control row whose width is not `max(ctrl_qubits) + 1`, a control row
whose controlled program is no larger than its receiver, a control row carrying
`mode_agreement` while adding more than one control, a power row whose width is not the
receiver's, a power row whose instruction count is not `abs(exponent) * (2 * width - 1)`,
a power row carrying `mode_agreement` with an exponent below two, a power row whose block
sits outside its receiver, a route section whose identity is not exact or that names a
measure the gate cannot read, a `[[plan_corrections]]` row
whose name has appeared, an identifier the contract measures that the gate does not
read, a requirement row naming a test that does not exist, a `covered` plus `not_covered`
that is not the family `contracts/circuit-composition-contract.toml` declares, a declared
operator spelling that no census row measures, and a declaration source that reads no
members at all.

## Open Questions

1. **Who adjudicates `circuit-composition-contract.toml`'s `not_provided` row?**
   ~~#552 and #542 falsify it on their branches; on `main` it remains true.~~ This
   one is **answered**, twice, by the merges that falsified it: `N1-4`/#552 and
   `N1-5`/#542 each updated the row in its own PR, which is the cheapest of the
   answers considered here. `control` and `power` are both in neither `not_provided`
   nor this contract's `not_covered`; that row is now empty, this contract's
   `not_covered` is empty with it, and the gate checks the two against each other
   rather than trusting either.

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

5. **Does the mode half of the control claim belong in this contract at all?** It
   is here because it is the same claim, measured on the same pair of routes. But
   the two-control row shows the bound is not generous for a 57-instruction
   expansion, so a future multi-control row will land outside it. The honest
   options are a per-row bound, or moving the mode measurement out of this
   contract entirely — neither is obviously right, and the current arrangement
   (measure the mode half on marked rows, record the wider number) is a holding
   position that should be revisited when a second multi-control row is added.

6. **Does the mode half of the power claim belong here either?** It is measured on the
   smallest repeated program (a two-qubit receiver halfway through a square), where the
   spread is what a copy of one instruction produces and nothing more. `power` cannot
   widen the mode spread the way `control` can, because it repeats or rescales a program
   instead of expanding a gate into a ladder, so no row is currently near the bound. If a
   later power row is measured near it, the same question as above applies.

## Owner and approvals

Owner: team `integration` (protected contracts, `tools/**`, CI, and `tests/**`).
This change requires no API-owner approval because it changes no API; the
`authorization_required = false` field is re-read by the gate. Reviewer attention
is best spent on the two claims the contract separates — exact between the
construction routes, tolerance between the modes — and on whether an empty
`not_covered`, checked against the family the composition contract declares, is the right
way to keep the census honest now that every declared member is covered. The addendum
adds a third thing to look at: whether pinning the mode half to the one-control rows is
the right response to a bound the widest row nearly reaches, or whether that is a finding
about the bound that this contract is papering over. The census addendum adds a
fourth: whether `circuit-composition-contract.toml` is the right authority for the family,
or whether it belongs to a declaration of its own. It is read from that contract because
that contract already states what these members mean and already carries their proposal
records, so a separate declaration would be a third list of the same names.
