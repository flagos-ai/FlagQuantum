# API Change Implementation 087: Dynamic remote stable-facade preflight

## Status

**Implementation authorized on 2026-10-24. Provider activation, networking,
task submission, and support promotion remain separately gated.**

The API owner authorized this implementation with:

```text
approve API-CHANGE-087-DYNAMIC-REMOTE-STABLE-FACADE-PREFLIGHT-IMPLEMENTATION
```

This implementation applies the approved fail-closed boundary from
`API_CHANGE_PROPOSAL_087_DYNAMIC_REMOTE_STABLE_FACADE_PREFLIGHT.md`. It does not
claim that Quafu, Azure, Amazon Braket, IQM, or another QPU can execute a
FlagQuantum `DynamicCircuit`.

## User-visible behavior

Local dynamic execution keeps the existing stable entry:

```python
import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit

circuit = DynamicCircuit(2).h(0).measure(0, classical_bit=0)
circuit.conditional("x", 1, classical_bit=0)

result = fq.run(circuit, shots=128)
assert result.samples.shape == (128, 2)
```

A remote request uses the same stable facade but fails before provider-owned
work while no provider supplies a qualifying detailed snapshot:

```python
import flagquantum as fq
import flagquantum.errors as fqe
from flagquantum.dynamic import DynamicCircuit

circuit = DynamicCircuit(2).h(0).measure(0, classical_bit=0)
circuit.conditional("x", 1, classical_bit=0)

try:
    fq.run(circuit, target="quafu:Baihua", shots=128)
except fqe.CapabilityError as error:
    message = str(error)
    assert "target=quafu:Baihua" in message
    assert "dynamic_capability_fact_missing" in message
```

Run the maintained credential-free example with:

```bash
python -m examples.local.dynamic_deployment_preflight
```

## Implementation

The stable facade now classifies a remote `DynamicCircuit` immediately after
the unchanged local-dynamic branch and before every static remote/provider
branch. The internal preflight:

1. requires one explicit `provider:backend` target;
2. derives the canonical requirement set from ordered Circuit IR;
3. consumes one exact device-scoped `TargetCapabilitySnapshot`;
4. reuses `match_dynamic_remote_capabilities(...)` from API Change 086;
5. preserves requirement, snapshot, and target identities in its private
   decision; and
6. raises `CapabilityError` with the exact target and deterministic ordered
   blocker codes when admission fails.

When the stable facade has no detailed provider snapshot, it creates an
evidence-free internal snapshot that deliberately supplies no capacity or
dynamic facts. It therefore cannot turn a coarse
`supports_dynamic_circuits=True` declaration into acceptance.

The pure internal evaluator accepts an explicit snapshot for deterministic
contract tests and future provider integration. The stable facade does not yet
have a positive provider binding, so every remote dynamic request remains
closed. The existing experimental backend assessment remains a diagnostic for
deployment/profile workflows and is not an admission authority for `fq.run`.

## Ordering and unchanged surfaces

The refusal occurs before:

- provider or remote module import;
- credential lookup, discovery, or network access;
- compilation, routing, dialect lowering, or package creation;
- task creation, submission, polling, charging, or result construction; and
- random-number consumption.

This change adds no public export, stable parameter, schema version, provider
option, target alias, or result field. Stable local `DynamicCircuit` execution
and static remote dispatch retain their previous behavior.

## Verification

The implementation covers:

- Quafu, Azure, non-IQM Braket, incomplete IQM, and unknown targets;
- missing and malformed explicit targets;
- exact target and deterministic blocker reporting;
- an accepted detailed snapshot through the API Change 086 matcher;
- wrong-device snapshot identity refusal;
- no remote-module import, no compilation, and no RNG mutation on refusal;
- unchanged stable local dynamic execution; and
- unchanged static Quafu dispatch.

Verified in the repository Python 3.12 Docker environment:

```text
18 passed: focused stable-facade and capability-contract tests
64 passed: dynamic, Azure, Quafu, and capability regression selection
ruff: passed
mypy --strict: passed
```

## Remaining gates

The following are intentionally not completed by this implementation:

- reproducible preflight and static-bypass performance evidence;
- provider-owned snapshot acquisition and freshness policy;
- a second capability match after routing and dialect lowering;
- a sealed provider package and exactly-once mocked submission;
- task identity, polling, cancellation, and normalized result semantics;
- bounded live QPU evidence; and
- promotion of any remote dynamic provider/device/feature slice.

Until those gates are separately proposed, approved, implemented, and
verified, FlagQuantum supports stable `DynamicCircuit` execution locally only.
