# API Change Proposal 033: Twin v1 freeze review

## Status

**Freeze preparation implemented; final freeze requires explicit approval.**
Proposals 017 and 027–032 established the first complete Twin vertical slice.
This review removes one avoidable pre-release ambiguity and extends the
machine-readable contract to the complete intended v1 surface. It deliberately
leaves `candidate_is_frozen_contract` false until a maintainer approves the
final contract.

## Product boundary

FlagQuantum provides a framework API for constructing a digital model of any
QPU, predicting classical measurement distributions, binding explicit hardware
validations, and carrying their evidence. Quafu calibration conversion and
submission are natively supported adapters.

FlagQuantum does not own agents, natural-language invocation, MCP, dashboards,
global publication, tenancy, routing policy, automatic calibration changes, or
continuous model operation. Applications such as FlagQAI may compose the
framework API to provide those capabilities.

## Construction decision

The only supported construction style is the concise module-level form:

```python
twin = fq.twin.from_noise_model(
    device_noise_model,
    target="provider:backend",
    qubits=(12, 13),
)
```

Native Quafu calibration uses the same target grammar:

```python
twin = fq.twin.from_quafu_chip_info(
    chip_info,
    target="quafu:Shenglian",
    qubits=(20, 27),
)
```

The older `QPUDigitalTwin.from_noise_model(...)`,
`QPUDigitalTwin.from_quafu_chip_info(...)`, and
`QPUDigitalTwin.from_quafu_provider(...)` methods used longer, incompatible
parameter names. They were not included in the candidate signature contract,
but their public spelling made them discoverable. This proposal removes them
before v1 freeze instead of preserving two ways to construct the same object.
Provider retrieval remains explicit:

```python
chip_info = provider.fetch_chip_info("Shenglian")
twin = fq.twin.from_quafu_chip_info(
    chip_info,
    target="quafu:Shenglian",
    qubits=(20, 27),
)
```

## Candidate v1 surface

The machine-readable contract now covers:

- every `flagquantum.twin` export;
- all intended public function and method signatures;
- ordered fields and frozen status for every public dataclass;
- all public serialized schema defaults;
- every `TwinEvidenceStatus` literal value;
- the single module-level construction convention;
- the agent, MCP, application-policy, and natural-language exclusions.

The semantic contract proves that provider-neutral construction and prediction
are deterministic and offline, that `TwinExperiment.prepare(...)` freezes the
canonical QASM and physical mapping without submission, and that the removed
constructor shortcuts cannot reappear unnoticed.

## Scientific meaning retained

Twin agreement is `1 - total variation distance` between classical measurement
distributions. It is not state fidelity, amplitude accuracy, or per-shot success
probability. A verified bound applies only to its recorded circuit and physical
mapping. Structural membership without an estimated bound is not an accuracy
claim. Repetition measures observed QPU repeatability and still contains shot
noise.

## Compatibility

The repository is pre-release and the Twin contract explicitly remains a
candidate. Removing the duplicate class construction methods prevents permanent
compatibility debt. The recommended module-level APIs, serialized evidence v1,
serialized validation-series v1, prediction behavior, hardware validation, and
Quafu example remain unchanged.

## Final freeze checklist

- Public names state domain meaning directly.
- One construction grammar is used everywhere.
- The framework/provider/application boundaries are explicit.
- Public signatures, fields, literals, and schema versions are machine checked.
- Offline and live examples are complete and distinguish hardware effects.
- Exact-circuit statistical scope is stated beside every agreement example.
- Focused API and Twin tests, default smoke/unit tests, strict typing, formatting,
  and package checks pass.
- A maintainer explicitly approves changing `candidate_is_frozen_contract` to
  true in a separate reviewed commit.
