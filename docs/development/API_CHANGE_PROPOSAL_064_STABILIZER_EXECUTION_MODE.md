# API Change Proposal 064: A stabilizer execution mode

## Status

**Approved by the repository owner and implemented by W1-01b. Every acceptance
item below is checked against its measured result.**

The change moved under review while it was implemented, and the record says so.
The proposal was written against an inspection of the runtime; three of its
premises then changed once the code was measured, and each correction is stated
where it applies rather than silently absorbed:

- the route was not missing, only mislabelled (`### The route already exists`);
- the estimate is now the tableau's, and the ceiling it created is gone rather
  than moved;
- the refusals are split across two authorities by which fields own the fact
  (`### 2a. Where a refusal lives is part of the decision`).

The change is one new value in an existing closed vocabulary plus the planner and
executor behaviour that value selects. It adds no Stable Core root export, no new
plan property, no new result field, no new `ExecutionResult` field, and no new
serialization schema. It does add one name to the already-registered
`flagquantum.simulation.stabilizer` namespace; the deviation is recorded under
"Deviation from the proposed surface" below rather than left implicit.

## Problem

### The route already exists, and that is the surprising part

Proposal 063 (backend execution admission, landed by W0-08) gave an admitted
extension backend a route through `fq.run`. Before requesting a new mode, the
existing route was measured against the engine W1-01a landed, because
engineering decision principle 6 requires inspecting what already exists.

A samples-only backend that declares `distribution_semantics =
"single_device_fast_path"` and returns an object with a `sample()` method
already completes the full path with no Stable Core change:

```python
result = fq.run(circuit, options=fq.ExecutionOptions(backend="..."),
                outputs=fq.samples(), shots=64)
result.samples            # (64, n_wires), torch.int64
result.state              # None
result.runtime["execution_path"]         # 'admitted_extension_backend'
result.runtime["distribution_semantics"] # 'single_device_fast_path'
result.runtime["scalability_claim_allowed"]  # False
```

That was measured on a GHZ chain at 3, 24, 256, 1024, 4096 and 8192 wires: every
one planned and executed, the engine was called once per run, and at 8192 wires
the plan and the run took 11.0 seconds each. So the residual gap in the
`backend_stim_stabilizer` parity row is **not** "there is no route".

### What the plan says about that run is false

The same measurement shows the planner does not know that this route holds no
amplitudes. For every width it reported:

| Field | Reported | Meaning for a stabilizer engine |
| --- | --- | --- |
| `mode` | `statevector` | the engine never builds a statevector |
| `state_mode` | `statevector` | same |
| `state_bytes` | `8 * 2**n_wires` | the dense amplitude store it will not allocate |

At 1024 wires the plan claims `state_bytes = 2**1027` bytes, roughly 10^309
bytes, for an engine whose live state is a few hundred kilobytes. At 4096 wires
it claims `2**4099` bytes. Planning that circuit took 2.7 seconds, and the cost
grows with the number of digits in an integer that is never used for anything.

This is not a cosmetic detail. `docs/roadmap/CAPABILITY_MATURITY.md`,
`docs/reference/KNOWN_LIMITATIONS.md`, and the capability registry are all built
on the position that a plan states what will happen. A plan that reports a
representation the execution route does not use is a misreport, and the
repository's required workflow treats exactly this class of defect as a test
obligation, not as wording.

### The misreport is also a hard ceiling

The estimate is serialized into the plan identity, which is
`sha256(canonical_json(...))`. `state_bytes` is an `int`, so it becomes a decimal
string, and CPython caps integer-to-string conversion at 4300 digits by default
(`sys.get_int_max_str_digits() == 4300`).

Measured on the same route:

| `n_wires` | `state_bytes` digits | Result |
| --- | --- | --- |
| 8192 | 2467 | planned and executed |
| 14284 | above the cap | `ValueError: Exceeds the limit (4300 digits) for integer string conversion` |
| 16384 | above the cap | same |

So a stabilizer route cannot plan past roughly fourteen thousand wires — not
because the engine cannot represent the state, and not because memory ran out,
but because the framework wrote down an amplitude count it was never going to
allocate. W1-01a demonstrated a 1024-wire GHZ precisely because the wire counts a
dense store cannot hold are the point of the representation.

### Why the capability registry cannot fix this alone

W0-08 gave `BackendCapabilities` the fields needed to describe execution honestly
(`distribution_semantics`, `scalability_claim_allowed`, `blockers`). A backend
can now say *how* it distributes work. It cannot say *what it holds*, so the
planner falls back to the native representation cost model for every admitted
backend and stamps the native mode. `backend_selection.py` computes
`estimate_state_bytes(ir.n_wires, ...)` before it looks at the selected backend,
and `_VALID_STATE_MODES` in `planner/execution_policy.py` admits only
`statevector`, `density_matrix`, `mps`, and `tensor_network`.

The one place in the framework that can express "the state is a tableau, not an
amplitude store" is the mode vocabulary. Nothing else in the existing contracts
carries that fact.

## Decision

### 1. Add `stabilizer` to the execution mode vocabulary

`ExecutionOptions.mode` gains one value, `"stabilizer"`, alongside `auto`,
`statevector`, `mps`, `tensor_network`, and `density_matrix`. The candidate
contract `contracts/execution-options-v1-candidate.json` records it in
`allowed_values.mode`, and `_MODES` in `runtime/options.py` gains it in the same
change so the implementation and the contract cannot drift.

The value names the representation. It is not `stim`, not `clifford`, and not a
backend name: `stim` is one implementation of the representation, and the
existing `backend` field already names implementations. Rule 9 forbids a
name that describes a vendor or an implementation era where a precise domain
term exists, and "stabilizer" is the domain's own term for this state.

### 2. The planner may only select it explicitly

`"auto"` does not begin selecting `stabilizer`. Choosing it requires stating it:

```python
fq.plan(circuit, options=fq.ExecutionOptions(mode="stabilizer", shots=1024))
```

The reason is fail-closed behaviour, not conservatism. Automated selection would
need a Clifford-membership test and a circuit-wide gate sweep — including the
`ccx`/`cswap` distinction that W1-01a had to verify by Pauli conjugation rather
than by assumption — before it could promise a plan it can honour. A wrong
automatic choice is a silent fallback in the direction of an approximation, which
non-negotiable rules 3 and 9 prohibit. Explicit selection fails closed at plan
time with the gate named.

### 2a. Where a refusal lives is part of the decision

The refusals split by which fields carry the fact being refused, and there is
exactly one authority for each:

- `flagquantum/runtime/planner/__init__.py::_require_stabilizer_request` reads
  the resolved options and refuses an output target other than sampling, a
  measurement kind the tableau cannot answer, a request with neither shots nor a
  sampling measurement, a non-CPU device, and a non-Clifford program.
- `flagquantum/runtime/planner/__init__.py::_require_stabilizer_representation`
  reads the plan's own fields and refuses `bsz != 1`, `require_gradients`, and
  `world_size > 1`. It is called from `plan_advanced` after the state mode is
  normalized, which means `fq.plan`, the expert planner, and the executor's own
  plan build in `run_stabilizer_mode` all reach it.

The split is not cosmetic. `bsz`, `require_gradients`, and `world_size` describe
the representation rather than the request that asked for it, and they are the
three fields a stabilizer plan could otherwise record while the engine produced
none of them: one tableau is one circuit, sampling has no gradient, and a tableau
has no layout to partition. Keeping those checks in the options-level guard would
have left `plan_advanced(..., state_mode="stabilizer", bsz=2)` — and a bare
`run_native(..., mode="stabilizer", bsz=2)` — building a plan whose `batch_size`
the executor contradicts, which is the same class of misreport this proposal
exists to remove. `test_the_expert_planner_owns_the_structural_refusals` and
`test_a_bare_native_run_cannot_build_the_plan_the_planner_refuses` pin both
paths, so the two cannot drift back apart.

### 3. `state_bytes` means the estimate for the selected representation

For `mode="stabilizer"` the plan reports the tableau estimate rather than the
amplitude-store estimate. The tableau for `n` qubits is `2n` rows of `2n + 1`
bits (a Pauli frame plus a sign), so the estimate is `O(n**2)` bits and
`state_bytes_for_stabilizer(n) = ceil((4 * n**2 + 2 * n) / 8)`.

This is presented as an estimate, not as a measurement. It is the
Aaronson-Gottesman representation size, which bounds any bit-packed tableau; the
engine's own reported allocation remains the authority for what it actually
used. The formula is stated here so the proposal can be reviewed before the
number is coded, and so a reviewer can check the plan's claim against the
representation rather than against a magic constant. W1-01b codes it as
`estimate_stabilizer_bytes` in `flagquantum/runtime/planner/estimates.py`.

At 16384 wires that estimate is about 134 MB — a number that serializes without
trouble, which is the point.

### 4. The result reports the mode that ran

`ExecutionResult.runtime["mode"]`, and the plan's `mode` and `state_mode`, carry
`"stabilizer"` for this route. A reader must be able to tell from the artifact
alone that no amplitude store existed, in the same way that
`distribution_semantics` already tells them that no sharding existed.

## Required evidence before this proposal can be accepted

- [x] `contracts/execution-options-v1-candidate.json` lists `stabilizer` in
      `allowed_values.mode`, and `runtime/options.py::_MODES` matches it exactly.
      `test_execution_mode_vocabulary_matches_the_implementation` asserts the
      equality of the two sets and that every member constructs.
- [x] `tests/unit/test_execution_options_candidate.py` pins the new set —
      `auto` plus five representations — so a further mode cannot be added by
      editing one side of the pair.
- [x] Planning a Clifford circuit at 16384 wires succeeds and reports a
      `state_bytes` of `O(n**2)` bits, not `8 * 2**n`. Measured: 134221824 bytes
      (9 decimal digits) where the amplitude estimate is `2**16387`. The same
      plan at 20000 wires reports 200005000 bytes, so the ceiling recorded in
      "The misreport is also a hard ceiling" is gone rather than moved.
- [x] Planning the same circuit at `mode="auto"` still selects a native
      representation, so no existing plan changes meaning. Measured at 2, 3, 24
      and 1024 wires: `state_mode` stays `statevector`.
- [x] Planning a non-Clifford circuit at `mode="stabilizer"` fails with the
      offending gate named, before any execution route is selected:
      `CapabilityError: instruction 1 't' is not a Clifford gate; stabilizer
      sampling accepts ...`. A noise channel is refused the same way.
- [x] A plan at `mode="stabilizer"` records `scalability_claim_allowed = false`
      and `distribution_semantics = "single_device_fast_path"`, and
      `release_gate_allowed = false`.
- [x] `docs/public_api_v1.json`, `IR_VERSION`, and every serialized schema
      version are unchanged apart from the added enum value.
- [x] `python tools/ci_tier.py pr-runtime` passes.
      Measured: `1195 passed, 53 skipped, 5505 deselected in 138.56s`.
- [x] `python tools/ci_tier.py pr-default` passes everything this change can
      affect. Measured: `1 failed, 4666 passed, 186 skipped, 1900 deselected in
      109.83s`. The single failure is
      `tests/unit/test_native_cpu_adjoint.py::test_compact_cx_runtime_threshold_and_rollback`,
      which asserts `use_compact_cpu_cx_mapping(22)` and returns `False` because
      the checkout holds no compiled `flagquantum/simulation/native_cpu/_C`
      extension: `native_cpu_compact_cx_index_available()` is `False` here. The
      change under this proposal touches no file in `simulation/native_cpu`, no
      statevector executor, and no adjoint path, and the failure reproduces from
      a clean tree in this checkout. It is recorded as an environment limit of
      this checkout rather than as a passing tier.
- [x] Repository owner authorizes the vocabulary change. Authorized explicitly for
      this proposal and this implementation (W1-01b) before any of the code above
      was written; see `## Status`.
- [x] Every refusal added by this proposal is proven by a test that fails when the
      refusal is deleted. Measured: 26 mutations of the guards, the mode
      vocabulary, the estimate, the selection rule, the dispatch, and the result
      stamp, with **all 26 killed** by the focused tests. A green focused suite
      alone does not show that a guard is exercised, so each mutation was applied
      to the source and the suite re-run. Five guards survived the first pass —
      the planner's measurement-kind check, the executor's measurement-kind and
      shot-count checks, and the target's encoding check — and each was given a
      test that reaches it: the first three through an IR whose `MeasurementNode`
      names a kind or omits a shot count, the fourth through the sampler protocol
      the target implements. The Literal alias
      `StateRepresentation` is excluded from the mutation set deliberately: it is
      a type-level alias used by `NoisyExecutionPlan`, which no stabilizer path
      constructs, and `ExecutionPlan.state_mode` is a plain `str` validated at
      runtime, so mutating the alias cannot change stabilizer behaviour.

## Deviation from the proposed surface

The proposal said the change adds no new public entry point. The implementation
adds one name to the already-registered `flagquantum.simulation.stabilizer`
namespace, `require_clifford_program`, and registers it in the capability's
`public_apis`.

The acceptance item above requires planning to refuse a non-Clifford circuit
*with the offending gate named*. The Clifford gate set is a property of the
representation, and the representation has exactly one owner: the simulation
domain that implements it. Runtime can therefore either ask that owner, or hold a
second copy of the classification — a channel check, a canonical-opcode
normalisation, a membership test against thirteen opcodes, and a second
refusal text. Engineering decision principle 6 forbids the second source of
truth, and principle 11 forbids the duplicated validation. The helper is that
question asked of the owner, `sample_stabilizer` already calls the same sweep,
and `test_the_planning_refusal_is_the_engine_refusal` asserts the two refusal
texts are identical so they cannot drift.

The added name is not a Stable Core root export, adds no mode, no plan property
and no result field, and its removal is a one-line change in the planner. It is
recorded here because the proposal's surface claim was narrower than the
implementation's, and a reader comparing the two should find the difference
stated rather than inferred.

## Non-goals

- A first-party built-in backend, an entry-point group, or a second registry.
  Whether the stabilizer engine should be a built-in backend or an adjudicated
  extension is a separate decision, and W0-08's admission boundary already
  permits the second one today without this proposal.
- Automatic selection, cost-model integration, or mode recommendation. Explicit
  selection only, per decision 2.
- Expectation values, detector error models, decoding, or noise over the
  stabilizer representation. Those extend what the engine computes; they do not
  need a mode value and must not be bundled with one.
- Any change to `state_bytes` for the existing four modes. Their estimates are
  correct because their engines do allocate what they claim.
- Any distributed claim. A tableau is not partitioned across ranks here, so
  `scalability_claim_allowed` stays false.
- Promotion of the `backend_stim_stabilizer` parity row past `partial`. This
  proposal removes one named blocker; it does not itself constitute parity
  evidence.
