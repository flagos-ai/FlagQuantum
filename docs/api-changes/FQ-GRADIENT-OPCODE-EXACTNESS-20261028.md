# Every differentiable opcode agrees with its own declared rule, in every mode

- **Status:** implemented
- **Owner:** core (`flagquantum/gradients.py`, `flagquantum/core/operator_schema.py` as the declaration it reads)
- **Date:** 2026-10-28
- **Related:** [Gradient parameter frequencies](FQ-GRADIENT-PARAMETER-FREQUENCIES-20261002.md),
  [The batch parameter-shift profile reads the opcode declaration](FQ-GRADIENT-BATCHED-SHIFT-PROFILE-20261020.md),
  [`fq.gradient` and the reported gradient method](FQ-GRADIENT-API-20261002.md)

## Problem

Three artifacts already said things about the parameter-shift rule, and none of them
said the thing a user depends on.

`contracts/gradient-api-v1-candidate.json` records which gradient **methods** exist.
`contracts/parameter-shift-coverage-contract.toml` records which opcodes the
remote-safe **batch** profile admits, and it measures each admitted parameter against
PyTorch autograd. Neither answers the question the plan asks for this slice:

> For every opcode whose derivative rule is declared, is the analytic rule the
> derivative the implementation actually applies -- **in every execution mode the
> caller can reach**, not just the default one?

That distinction is not academic. `statevector`, `mps`, `tensor_network` and
`density_matrix` are separate executors. A rule can be correct in the default mode and
wrong in another mode, and a claim that covers one mode is a claim about one executor.
Nothing in the repository compared the derivative across modes.

Two traps make the comparison harder than it looks, and both of them produce a
**passing** result when handled naively.

1. **Two routes can agree and both be wrong.** `fq.gradient(..., method="autograd")`
   and `fq.gradient(..., method="parameter_shift")` can share the same analytic
   derivative rule. Measuring that they agree to machine precision is evidence that
   they are consistent, not that either is right. The batch profile's exactness column
   has this same structure.
2. **A probe can measure nothing and read as agreement.** A differentiated phase-type
   opcode applied to a bare basis state leaves a state that differs from `|0...0>` by a
   scalar phase, so its derivative is exactly zero -- and every route reports that zero
   identically. During this slice an initial probe did exactly this: four opcodes
   (`cphase`, `phase`, `u1`, `u3`) all measured a zero gradient **together**, which is
   the signature of a global-phase-invisible circuit rather than four correct answers.
   A comparison against an all-zero reference passes vacuously.

## Decision

The plan's opcode axis becomes a measured contract with an independent third route and a
measured observability floor.

```console
$ python tools/check_opcode_gradient_exactness.py
Opcode gradient exactness contract passed: 14 differentiable opcodes x 5 modes = 70 cells
re-measured against an independent difference scheme, 1 excluded mode, 3 opcodes split
off by the batch profile
```

### The opcode axis is derived, not listed

The row set is the set of `OperatorSchema`s whose `differentiable` property is true.
That property is itself derived -- it returns `bool(self.parameter_frequencies)`, so an
opcode cannot claim to be differentiable without declaring the frequency set the claim
rests on. Measured on this checkout: **35 opcodes, 14 differentiable**, and the 21 that
declare no rule are not rows. Registering a frequency set adds a row by existing, and
the gate fails until the new opcode has been measured. There is no hand-written opcode
list anywhere in either artifact.

The 14 are `cphase`, `crx`, `cry`, `crz`, `phase`, `rx`, `rxx`, `ry`, `ryy`, `rz`,
`rzz`, `u1`, `u2`, `u3`. Twelve declare one frequency per parameter and need a two-term
rule; `crx`, `cry` and `crz` declare `{0.5, 1.0}` and need a four-term rule. `u2` and
`u3` are the only multi-parameter opcodes in the set, with two and three declared
parameters respectively, each with its own two-term rule.

### The mode axis is measured, and `stabilizer` is a refusal rather than a column

Five modes are columns: `auto`, `density_matrix`, `mps`, `statevector`,
`tensor_network`. The mode vocabulary is read from
`flagquantum/runtime/options.py::_MODES`, so a new execution mode fails the gate until
the matrix is revisited.

`stabilizer` is **not** a column. It refuses all 14 differentiable opcodes, with one
message, before any derivative is attempted:

```console
$ python -c "..."
CapabilityError | mode='stabilizer' samples measurement outcomes; it cannot serve
measurement kind(s) expectation_ps
```

That is a property of the measurement kind and not of any opcode, so it is recorded once
as a refusal row naming the two files its phrase occurs in, rather than fourteen times
as an absence. The gate does not trust the exclusion: it drives every differentiable
opcode through a `stabilizer` loss and requires the `CapabilityError`, so a mode that
starts serving an expectation value turns the gate red.

### A third route that shares no code with either gradient path

Each of the 70 cells compares three gradients:

| Route | How it is produced |
| --- | --- |
| `autograd` | `fq.gradient(..., method="autograd")` over the loss |
| `parameter_shift` | `fq.gradient(..., method="parameter_shift")` over the circuit and loss |
| third route | Richardson-extrapolated central difference of `fq.run` alone |

The third route is three central differences at displacements `1e-2`, `5e-3`, `2.5e-3`,
combined through two Richardson levels to cancel the leading truncation terms. It calls
`fq.run` and nothing else, so it shares no code with either gradient path -- that is
what makes the comparison a test rather than a consistency check. `[difference_scheme]`
records that its base step is a probe choice rather than a quantity the implementation
declares, so nobody later mistakes it for a contract.

Measured maxima over all 70 cells:

| Quantity | Measured | Contracted tolerance |
| --- | --- | --- |
| Largest `parameter_shift` vs `autograd` deviation | `3.925231146709e-16` | `1e-12` |
| Largest `autograd` vs third-route deviation | `1.606145771937e-13` | `1e-12` |
| Largest `parameter_shift` vs third-route deviation | `1.605798827242e-13` | `1e-12` |
| Largest cross-mode spread within one opcode | `1.221245327088e-15` | `1e-12` |

The third-route figure is two orders of magnitude above the two-route figure, and that
is the expected shape rather than a defect: the two analytic routes agree to round-off,
while the difference scheme carries the round-off of the loss evaluations it is built
from.

### The reference program is non-degenerate on purpose

Writing the probe correctly was most of this slice's work. The reference prepares every
qubit with a fixed rotation, joins them with `cx(0,1).cx(1,2)`, applies the
differentiated opcode, then closes with `cx(2,0).ry(0, theta=0.9)` before measuring
`expectation(Z(0))`. The closing rotation is what turns the measurement basis back onto
the differentiated parameter; without it, a phase-type opcode acts on a state that
differs from `|0...0>` by a scalar phase only and every one of its derivatives measures
as zero.

The **observability floor** is therefore measured, not assumed:

```console
observability_floor = 3.360360652996e-02
```

It is the smallest `parameter_shift` norm over all 70 cells, reached by `crx` and `crz`
in all five measured modes (the two share a generator spectrum and differ only in the
control), and the largest is `u3` in `auto` at `8.352901355015e-01`.
The gate requires it to be above the tolerance, because a comparison against a reference
that small would be vacuous. This, and not the tolerance, is the clause that would have
caught the initial all-zero probe.

### The relationship to the batch profile is re-derived

Every differentiable opcode is admitted by the remote-safe batch profile except three,
and the reason is not a wider defect:

| | Opcodes | Why |
| --- | --- | --- |
| Admitted by both | 11 | one frequency per parameter, one evaluation pair |
| Refused by the batch profile | `crx`, `cry`, `crz` | `{0.5, 1.0}` needs two evaluation pairs; the profile's `max_evaluation_pairs` is `1` |

The gate reads the batch contract's own file and re-derives that split rather than
restating it, and it requires the refusal to be the pair limit -- a refused opcode that
carried a `channel` reason instead would fail, because then the recorded split reason
would not hold for every refused opcode.

## Scope

| | Before | After |
| --- | --- | --- |
| Opcode x mode exactness | unmeasured | 70 cells, three routes each |
| Independent third route | none | Richardson-extrapolated difference of `fq.run` |
| Observability floor | unstated | measured, `3.360360652996e-02`, gated above tolerance |
| `stabilizer` in the gradient story | one row in the method matrix | excluded with a driven refusal |
| Batch-profile split | two files that never met | re-derived by the gate from both |

### Explicitly out of scope

- **No opcode gains or loses a derivative rule.** `flagquantum/core/operator_schema.py`
  is read and not modified. This change measures the declaration; it does not widen it.
- **No new gradient method and no new execution mode.** `N3-7` (`hessian`) and `N3-8`
  (metric tensor) are not here. Both would add a method to `_GRADIENT_METHODS`, which
  now turns **two** gates red -- this one and the method matrix -- until both are
  revisited. That is the intended behaviour and it is why this contract had to exist
  before either of them.
- **No noisy axis.** The method matrix measured that noise is a `fq.run` argument rather
  than a mode, and narrowed stable noisy execution to `auto` and `density_matrix`. This
  contract's reference is unnoisy; adding a noisy axis here would duplicate that matrix
  rather than extend this one.
- **Open Question 1 of the batch-profile record is left open.** Whether `crx`'s
  generator really has the eigenvalues `{0.5, 1.0}` is verified against autograd and not
  derived. This contract now verifies it in five modes instead of one, which strengthens
  the evidence without answering the question.

## Prohibited practices

- Do not hand-write the opcode list. A list in either artifact is the defect this change
  removes; the gate fails on a row for an opcode that declares no rule.
- Do not compare only `autograd` against `parameter_shift`. They can share a rule.
- Do not lower the observability floor to make a probe pass. If the floor is small, the
  reference program is degenerate.
- Do not add `stabilizer` as a column. It cannot serve an expectation value, and the
  refusal row is where that belongs.
- Do not edit `contracts/opcode-gradient-exactness-contract.toml` to make the gate pass.
  The contract records a measurement; a disagreement is a finding.
- Do not add a graded-mode exclusion without driving it. The gate runs every excluded
  mode against every differentiable opcode.

## Compatibility

- **No stable API changes.** Nothing in `flagquantum/__init__.py`, `fq.__all__`,
  `docs/public_api_v1.json`, or any signature is touched. `fq.gradient`'s signature and
  `GradientResult` are unchanged. No change proposal is required and none is claimed.
- **No capability row moves.** `capability-maturity.toml` and
  `docs/reference/KNOWN_LIMITATIONS.md` are unchanged: this change adds evidence for
  behaviour the repository already claimed, and claims nothing new.
- **No implementation file is modified.** `flagquantum/` is untouched, so the `core`
  team's owned source is read rather than edited.

## Verification

`tools/check_opcode_gradient_exactness.py` rebuilds the reference program for every
differentiable opcode, runs all 70 cells, and compares each measured triple against the
cell's record. It re-derives the opcode and mode vocabularies from the implementation,
re-derives the batch-profile split from that contract's file, drives every excluded mode,
and recomputes the four aggregate figures from the cells rather than trusting them. It
runs in `.github/workflows/ci.yml` as a **step** and in `tools/pre_push.py`.

The gate is negative-tested, so a zero finding is evidence rather than silence. Every
mutation below was applied to the contract in memory, `contract_errors` was called, and
the gate was required to name it:

| Mutation | Gate output |
| --- | --- |
| `rx`/`mps` cell's `parameter_shift_vs_autograd` set to `0.5` | records `0.5` but re-measured `…` |
| `observability_floor` set to `1e-30` | cells re-measure `3.360360652996e-02` |
| `ry.cross_mode_spread` set to `1e-30` | modes re-measure `…` |
| `max_autograd_vs_richardson` set to `1e-30` | cells re-measure `…` |
| A row copied to `name = "x"` | describes `'x'`, which declares no derivative rule |
| `rzz`'s row dropped | no `[[opcode]]` row for `'rzz'` |
| `u3`'s `mps` cell dropped | has no cell for `('u3', 'mps')` |
| `cell_count` incremented | measured 70 |
| `stabilizer` removed from `declared_modes` | the runtime declares `[…, 'stabilizer', …]` |
| `mps` added to `excluded_modes` | names `['mps', 'stabilizer']`, expected `['stabilizer']` |
| `excluded_modes` set to `statevector` | served 14 differentiable opcode(s) |
| `measured_modes` gains `stabilizer` | a mode that starts serving has to be measured here |
| `message_phrase` reworded | does not occur in `flagquantum/runtime/planner/__init__.py` |
| `exception` set to `ErrorNobodyRaises` | a class this gate cannot observe |
| `no_silent_fallback` deleted | not contracted as true |
| An unread rule flag added | not read by this gate |
| An excluded mode stripped of its reason | an excluded mode carries no reason |
| `refused_differentiable_opcodes` set to `["rx"]` | the batch profile refuses `['crx', 'cry', 'crz']` |
| `max_evaluation_pairs` set to `99` | the batch profile declares `1` |
| `order` set equal to `levels` | cannot extrapolate |
| `tolerance` set to `1e-30` | above the contracted tolerance |
| `verification.gate` pointed at a missing file | does not exist |
| `authorization` pointed at a missing record | does not exist |

`tests/unit/test_opcode_gradient_exactness_contract.py` owns the contract and the gate:
37 tests, of which 23 are mutations of the kind listed above. It also asserts the two properties that
make the matrix meaningful rather than merely self-consistent -- that the third route
shares no code with the compared routes, and that the reference is not vacuous -- and
that the contract states the qubit vocabulary and no retired synonym.

```console
$ python -m pytest tests/unit/test_opcode_gradient_exactness_contract.py -q
37 passed
```

## Open Questions

1. **Should the third route be the batch profile's exactness column too?** That
   contract compares only against autograd, which is the trap this change exists to
   avoid. Extending it would mean two contracts carrying the same difference scheme, so
   the scheme would want one home first.
2. **Is `u2`/`u3`'s multi-parameter rule evidenced as strongly as the single-parameter
   ones?** They are the only two rows with more than one declared parameter, and their
   rules are measured per parameter. A generator-spectrum derivation would make the
   whole declaration self-checking rather than autograd-verified, which is Open Question
   1 of the batch-profile record seen from a second angle.
3. **Should the cross-mode spread be a per-mode-pair record rather than a maximum?**
   The maximum is `1.221245327088e-15` and dominated by `ry`; a per-pair record would
   say which pair, at the cost of ten more rows per opcode.
4. **Does `density_matrix` need its own reference program?** Its derivative is measured
   on the same circuit as the other modes, which is what makes the cross-mode spread
   meaningful, but a density-matrix executor is also exercised under noise where the
   method matrix refused three modes. Whether that belongs here or in a noisy follow-up
   is unsettled.

## Owner and approvals

- **Implementation owner:** core (`flagquantum/gradients.py` and
  `flagquantum/core/operator_schema.py` as the declaration it reads).
- **Integration surfaces touched:** `contracts/` (one new contract),
  `.github/workflows/ci.yml` (one step), `tools/` (one gate, one pre-push entry),
  `docs/api-changes/` (this record and one index line). These are protected paths, so
  the change is an integration change and carries its own contract, gate, tests, and
  record.
- **Authorization:** `docs/api-changes/FQ-GRADIENT-OPCODE-EXACTNESS-20261028.md` (this
  document), listed in the contract's `authorization` field.
- **No API-owner approval required.** The change touches no stable export.
