# API Change Implementation 087: Inspectable assessment convergence

## Status

**Implementation authorized on 2026-10-24.**

The API owner authorized this convergence with:

```text
approve API-CHANGE-087-INSPECTABLE-ASSESSMENT-CONVERGENCE
```

This closes the assessment mismatch left after the stable-facade preflight
landed. It does not activate a remote provider, acquire a capability snapshot,
submit a task, or promote an experimental API.

## User workflow

```python
import flagquantum as fq
from flagquantum.deployment import CloudBackendProfile
from flagquantum.dynamic import DynamicCircuit

circuit = DynamicCircuit(2).measure(0, classical_bit=0)
circuit.conditional("x", 1, classical_bit=0)
backend = CloudBackendProfile(
    provider="quafu",
    name="Baihua",
    n_qubits=156,
    basis_gates=("x",),
    supports_dynamic_circuits=True,
)

report = fq.experimental.dynamic.assess_dynamic_backend(circuit, backend)

assert not report.compatible
assert "dynamic_capability_fact_missing:mid_circuit_measurement" in report.blockers
```

The coarse profile still describes experimental package and dialect constraints,
but it does not contain device-scoped facts, freshness, or evidence. Assessment
therefore reports the same fail-closed capability blockers as stable remote
`fq.run(...)`.

Run the maintained example with:

```bash
python -m examples.local.dynamic_deployment_preflight
```

## Implementation

`assess_dynamic_backend(...)` now performs two deliberately separate readings:

1. the API Change 086 requirement/snapshot matcher supplies the admission verdict
   and ordered capability blockers; and
2. the existing experimental package checker adds dialect, basis, classical-bit,
   and routing diagnostics without becoming an admission authority.

For a valid profile with no provider-owned detailed snapshot, the assessment's
admission blockers are exactly those returned by the internal stable-facade
preflight for `provider:name`. A coarse
`supports_dynamic_circuits=True` value cannot make the report compatible.

The experimental deployment-package builder continues to use its private
package checker. Creating a local sealed package is not evidence that stable
remote execution is admitted, and this change keeps those concepts visibly
separate.

## Compatibility boundary

- No root export, stable signature, serialized schema, provider option, or
  result type changes.
- Stable local `DynamicCircuit` execution remains unchanged.
- Static remote dispatch remains unchanged.
- Existing package diagnostics remain available in `report.blockers` after the
  canonical admission blockers.
- Experimental callers that previously treated a coarse positive profile as
  executable now receive a fail-closed report. This is an intentional correction
  before remote support is published.

## Verification

The focused contract proves that:

- experimental assessment and stable remote refusal consume one matcher;
- equivalent program and target inputs produce identical ordered admission
  blockers;
- a coarse positive dynamic flag cannot admit execution;
- package-only diagnostics remain visible;
- local package creation remains available as an experimental workflow; and
- the maintained example is credential-free and performs no provider call.

Verified in the repository Python 3.12 Docker environment:

```text
58 passed: dynamic, Braket IQM, provider, and remote preflight selections
1852 passed, 143 skipped: pr-runtime integration tier
black: passed
ruff: passed
mypy --strict: passed
credential-free example: passed
documentation source-of-truth, links, language, and maturity checks: passed
public API migration baseline: passed
runtime team scope: passed
```

## Remaining gates

- reproducible preflight and static-bypass performance evidence;
- provider-owned snapshot acquisition and freshness policy;
- post-routing capability rematch;
- sealed provider submission and exactly-once task semantics;
- normalized task/result lifecycle;
- bounded live evidence; and
- promotion of an exact provider/device/feature slice.
