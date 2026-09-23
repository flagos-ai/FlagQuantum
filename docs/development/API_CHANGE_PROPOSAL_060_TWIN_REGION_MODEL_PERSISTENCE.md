# API Change Proposal 060: Composed regional Twin model persistence

## Status

**Implementation authorized; additive to the frozen Twin v1 contract.** This
slice persists one composed `TwinRegionModel` so a released regional Twin can
be loaded in another process and passed directly to
`TwinRegionRelease.assess(...)` without retaining or recomposing every source
cell artifact.

## Problem

A released regional Twin exists only as a composition of source artifacts. To
use one after the fact, an application must retain every source cell Twin and
its circuit support, reload them, and recompose them in the exact same ordering
before an assessment can succeed. That has three concrete costs:

- Release artifacts are durable, but the composed model they were qualified
  against is not. Losing any source cell makes a release unusable.
- Recomposing is a repeated cost and a repeated failure mode. Any drift in the
  source artifacts silently changes `TwinRegionModel.identity`, and the release
  then refuses every circuit.
- Applications that "just save the region" in their own JSON create a second,
  unaudited representation of a region, a snapshot, a device profile, or a
  noise model, which is exactly the duplication this repository forbids.

There is no missing measurement or new capability here. The missing piece is
persistence for a model that already exists.

## Decision

- Add `fq.twin.dump_region_twin(region_twin, path)` and
  `fq.twin.load_region_twin(path)`.
- Persist one canonical JSON object with schema
  `flagquantum.twin_region_model_artifact.v1` and exactly two members:
  - `region`: the existing authoritative `TwinConnectedRegion` representation;
  - `twin`: the frozen composed Twin as its existing `TwinSnapshot`
    representation plus the existing canonical `NoiseModel` representation,
    which already carries the authoritative `DeviceNoiseProfile`.
- Reuse the existing serializers unchanged. `TwinRegionModel.identity`,
  `TwinConnectedRegion`, `QPUDigitalTwin`, `TwinSnapshot`, `DeviceNoiseProfile`,
  `NoiseModel`, and the shared create-once writer are not reimplemented, and no
  duplicate region, snapshot, profile, or noise representation is introduced.
- Load strictly: unknown schema versions, missing fields, extra fields,
  non-object members, malformed values, noncanonical payloads, and internally
  contradictory region/twin pairs are refused. Nothing is repaired.
- Loading reconstructs the real `TwinRegionModel`, which reruns its own
  invariants, and then re-derives the artifact and requires byte-equivalent
  canonical content. The composed device profile is bound to
  `flagquantum:twin-region:<region identity>`, so a region whose identity no
  longer matches its composed calibration cannot be accepted as an equivalent
  model.
- Write through the shared create-once policy: mode `0600`, an identical
  repeated write is a no-op, a different or invalid existing artifact is never
  replaced.

## Justification for a new artifact schema

The existing Twin persistence artifact `flagquantum.qpu_digital_twin.v1` stores
one `QPUDigitalTwin` and has no place for the regional structure; the regional
release artifact stores a qualification decision, not the model. Reusing either
would mean overloading a frozen schema with a different object, so a distinct,
versioned artifact schema is the smallest correct option. No new class, result
type, identity concept, or registry is added: the only new public surface is the
two functions, and `TwinRegionModel` itself is unchanged.

## Public API

```python
import flagquantum as fq

fq.twin.dump_region_twin(region_twin, "region-twin.json")
restored = fq.twin.load_region_twin("region-twin.json")
assert restored.identity == region_twin.identity

release = fq.twin.load_region_release("region-release.json")
assessment = release.assess(
    restored,
    circuit,
    physical_qubits=(20, 27, 34),
)

print(assessment.status)
print(assessment.prediction)
print(assessment.reasons)
```

## Artifact contract

```text
flagquantum.twin_region_model_artifact.v1
├── region  flagquantum.twin_connected_region.v1
│   ├── provider, backend_name, captured_at
│   ├── physical_qubits            ordered mapping
│   ├── directed_couplers          directed topology
│   ├── supported_operations
│   ├── maximum_instruction_count, maximum_circuit_depth
│   └── source_snapshot_identities, source_support_identities
└── twin
    ├── snapshot      flagquantum.twin_snapshot.v1
    └── noise_model   flagquantum.noise_model.v1 (incl. device_profile)
```

Every field is required, and the payload must be canonical: sorted keys,
compact separators, one trailing newline, and no extra members.

## Compatibility impact

- Additive only. No existing symbol, signature, default, field, enum value, or
  serialized schema changes, and `TwinRegionModel` keeps its current shape and
  identity computation.
- The Twin v1 contract file gains two public symbols and two signatures. The
  Twin v1 contract status stays `frozen`.
- Existing Twin and regional artifacts remain readable and are unaffected.
- No new dependency, no provider contact, and no change to any runtime,
  planner, compiler, or simulation path.

## Explicit non-goals

- No hardware task receipt, credential, raw provider response, validation
  evidence, confidence level, error bound, release decision, routing
  authorization, or application state is persisted. This is a model artifact.
- No QPU job submission, polling, or provider contact; no model mutation; no
  inference of new couplers, noise, or correlations.
- No Agent, MCP, REST, CLI, or natural-language behavior.
- No provider semantics in core: the persisted target is a plain
  provider/backend string, and Quafu appears only in existing fixtures and
  examples.
- No silent acceptance of unknown future schema versions and no repair of
  tampered payloads.
- No change to `TwinRegionRelease.assess(...)` or to release construction.

## Validation plan

- Round-trip identity and prediction parity for a supported circuit.
- Interoperability: a `load_region_twin` model passed to a loaded
  `TwinRegionRelease.assess(...)` returns `released_exact_circuit`.
- Repeated identical write is a no-op; different or invalid content is refused.
- Refusal of unknown schema, missing fields, extra fields, non-object members,
  noncanonical content, target mismatch, capture mismatch, reordered ordered
  mapping, topology tamper, source identity tamper, and noise-model tamper.
- Focused Twin tests, the full Twin test set, Ruff, Black check, strict mypy,
  public API/architecture/hygiene checks, `pr-default`, and `pr-runtime`.
