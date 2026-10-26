# API Change Proposal 094: DynamicCircuit static degeneracy

## Status

**Approved for implementation on 2026-10-26.**

After the API owner asked whether `DynamicCircuit` is compatible with static
circuits, the implementation recommendation was named
`DYNAMIC-STATIC-DEGENERACY-CONTRACT`. The API owner then authorized that exact
scope with:

```text
do
```

This approval changes stable remote dispatch behavior only. It does not
authorize credentials, discovery, network access, task submission, charging,
or a claim that any remote QPU supports dynamic circuits.

## User outcome

`DynamicCircuit` is a superset of `Circuit`: users may start with the dynamic
builder before deciding whether measurement, reset, or feed-forward is needed.
A program that never adds those semantics must not require dynamic-hardware
capability evidence merely because of its Python class.

```python
import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit

circuit = DynamicCircuit(2).h(0).cx(0, 1)
job = fq.run(circuit, target="quafu:Baihua", shots=1024)
```

Expected result: subject to the same credential and service prerequisites as an
equivalent `fq.Circuit`, the program follows the existing static Quafu path. It
does not require a dynamic-capability snapshot.

## Problem

The stable facade currently selects dynamic preflight with
`isinstance(program, DynamicCircuit)`. This rejects a program containing only
ordinary unitary instructions even though its IR requirements are identical to
those of an ordinary static circuit. The class used to build a program is not a
hardware requirement.

## Decision

Remote dispatch classifies a `DynamicCircuit` by its validated IR semantics:

- if no instruction carries measurement, reset, or classical-control metadata,
  the program is **statically degenerate** and follows the existing static
  remote workflow;
- if any instruction carries dynamic semantics, the existing dynamic remote
  preflight remains mandatory and fail-closed; and
- no simulator, provider, compiler, backend, or target substitution occurs.

The implementation reuses the canonical classical-control predicate. It does
not add a second definition of dynamic semantics and does not change the public
IR schema.

## Deliberate boundary

An explicit terminal `DynamicCircuit.measure(...)` remains dynamic in this
proposal. Normalizing terminal measurements into a provider's static result
contract requires a separate decision for classical-bit mapping, bit order,
repeated measurement, output shape, and provider format. Until that contract is
approved, measurement, reset, and conditional instructions continue to fail
closed when the target lacks exact dynamic evidence.

## Compatibility

- no public export, signature, default, exception family, or result schema
  changes;
- ordinary `Circuit` remote behavior is unchanged;
- local `DynamicCircuit` behavior is unchanged;
- true dynamic remote programs keep the existing preflight and refusal order;
  and
- the behavior is provider-neutral even though the positive offline contract
  exercises the currently supported static Quafu route.

## Acceptance criteria

- [x] A unitary-only `DynamicCircuit` reaches the same mocked static Quafu
      boundary as an equivalent `Circuit`.
- [x] Static degeneration does not invoke dynamic preflight.
- [x] Measurement, reset, and conditional programs remain fail-closed before
      static provider execution.
- [x] No credential, network request, SDK client, or live task is used by the
      contract tests.
- [x] The public API snapshot is unchanged.
