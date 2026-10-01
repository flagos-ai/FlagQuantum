# API Change Proposal 064: A stabilizer execution mode

## Status

**Proposed. A Stable Core change is requested.**

This proposal asks for one new value in an existing closed vocabulary and the
planner behaviour that value selects. It adds no new public entry point, no new
plan property, no new result field, and no new serialization schema.

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
representation rather than against a magic constant.

At 16384 wires that estimate is about 134 MB — a number that serializes without
trouble, which is the point.

### 4. The result reports the mode that ran

`ExecutionResult.runtime["mode"]`, and the plan's `mode` and `state_mode`, carry
`"stabilizer"` for this route. A reader must be able to tell from the artifact
alone that no amplitude store existed, in the same way that
`distribution_semantics` already tells them that no sharding existed.

## Required evidence before this proposal can be accepted

- [ ] `contracts/execution-options-v1-candidate.json` lists `stabilizer` in
      `allowed_values.mode`, and `runtime/options.py::_MODES` matches it exactly.
- [ ] `tests/unit/test_execution_options_candidate.py` pins the new set —
      `auto` plus five representations — so a further mode cannot be added by
      editing one side of the pair.
- [ ] Planning a Clifford circuit at 16384 wires succeeds and reports a
      `state_bytes` of `O(n**2)` bits, not `8 * 2**n`.
- [ ] Planning the same circuit at `mode="auto"` still selects a native
      representation, so no existing plan changes meaning.
- [ ] Planning a non-Clifford circuit at `mode="stabilizer"` fails with the
      offending gate named, before any execution route is selected.
- [ ] A plan at `mode="stabilizer"` records `scalability_claim_allowed = false`
      and `distribution_semantics = "single_device_fast_path"`.
- [ ] `docs/public_api_v1.json`, `IR_VERSION`, and every serialized schema
      version are unchanged apart from the added enum value.
- [ ] `python tools/ci_tier.py pr-default` and `pr-runtime` pass.
- [ ] Repository owner authorizes the vocabulary change.

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
