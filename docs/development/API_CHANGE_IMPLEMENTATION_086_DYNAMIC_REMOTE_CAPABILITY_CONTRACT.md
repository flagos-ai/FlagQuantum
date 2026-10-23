# API Change Implementation 086: Dynamic remote capability contract

## Status

**Candidate contract and failing-first conformance baseline approved on
2026-10-23. Runtime, provider, network, and task-submission activation remain
unauthorized.**

The API owner authorized this bounded stage with:

```text
approve API-CHANGE-086-DYNAMIC-REMOTE-CAPABILITY-CONTRACT
```

The protected baseline is
[`dynamic-remote-capability-v1-candidate.json`](../../contracts/dynamic-remote-capability-v1-candidate.json).

## User outcome

Users retain one eventual execution entry:

```python
import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit

circuit = DynamicCircuit(2).h(0).measure(0, classical_bit=0)
circuit.conditional("x", 1, classical_bit=0)

# Candidate spelling only: this stage deliberately refuses before networking.
fq.run(
    circuit,
    target="amazon-braket:<device-arn>",
    shots=128,
)
```

The example is intentionally failing-first. Until the requirement/snapshot
bridge and provider route are separately approved, the call raises before it
can construct a provider, compile a provider artifact, submit a task, or incur a
charge. The currently runnable credential-free assessment remains:

```bash
python -m examples.local.dynamic_deployment_preflight
```

## Frozen decisions

- The feature vector is derived from ordered Circuit IR, never supplied by the
  caller and never replaced by `supports_dynamic_circuits=True`.
- Provider facts distinguish missing, stale, unknown, unmeasured, unsupported,
  and not-exposed values, and distinguish declared, verified, and certified
  evidence.
- The target is always explicit. No provider choice, simulator fallback, static
  rewrite, or backend substitution is silent.
- Admission runs both before and after provider routing. A later stage cannot
  weaken or erase an earlier requirement or blocker.
- Refusal creates no provider task, consumes no RNG state, incurs no charge, and
  produces no partial result.
- The eventual result remains `fq.ExecutionResult`; unavailable mid-circuit
  measurements are reported as unavailable rather than invented.

## Boundary

This stage adds a candidate JSON contract and executable API-contract tests. It
does not modify production source, public exports, signatures, serialized
schemas, provider discovery, credentials, transport, or stable behavior.

The next separately approved stage is the requirement and snapshot bridge. It
must reuse `DynamicFeatureSet`, Core Target Capabilities, and
`CloudBackendProfile` rather than adding a second public capability model.

## Exit evidence

The stage exits when the contract records the exact approval, feature and
evidence vocabulary, blocker categories, admission order, result normalization,
and no-submit guarantee; tests prove the coarse backend boolean is only a hint,
the existing offline feature oracle produces precise blockers, and the proposed
Amazon Braket spelling remains closed without importing or constructing a
provider.
