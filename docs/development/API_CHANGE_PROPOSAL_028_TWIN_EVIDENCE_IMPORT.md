# API Change Proposal 028: Offline Twin evidence loading

## Status

**Implementation authorized; pending API freeze review.** The user explicitly
authorized the next Twin API slice after Proposal 027: connect externally
produced, frozen validation evidence to the provider-neutral framework without
moving research orchestration into FlagQuantum.

## Problem

`TwinEvidenceEnvelope` has a canonical serialized form, but callers currently
reconstruct it manually. That invites omitted fields, silently ignored schema
changes, and application-specific loaders. FlagQuantum needs one small,
provider-neutral path for restoring evidence that was produced elsewhere.

Historical research artifacts also use identities that are not interchangeable
with FlagQuantum identities. In particular, a physical OpenQASM hash is not a
FlagQuantum IR circuit identity, and a calibration identity is not a frozen
Twin snapshot identity. Import must fail closed rather than infer those links.

## Decision

- Add `TwinEvidenceEnvelope.from_dict(payload)` for strict canonical decoding.
- Add `fq.twin.load_evidence(path)` for offline JSON loading.
- Require the exact `flagquantum.twin_evidence_envelope.v1` field set.
- Reuse all envelope validation for digests, physical mappings, operations,
  bounds, and confidence levels.
- Perform no provider call, submission, polling, training, promotion, routing,
  or application-policy decision.
- Keep Q-ATLAS and other research-specific conversion outside the public API.
  Such workflows may emit the canonical envelope only if they prospectively
  recorded the exact Twin snapshot identity and FlagQuantum IR circuit
  identities before target outcomes became available.
- Provide an offline Q-ATLAS conversion tool as a narrow compatibility bridge.
  It verifies the source audit and a prospectively frozen identity binding,
  derives a conservative 95% radius from observed TV error plus the recorded
  finite-shot tolerance, and refuses to overwrite a different output.
  The prospective binding identifies the frozen validation draft; the later
  audit must prove that it descends from that draft. It cannot refer to an audit
  identity that does not exist until after hardware outcomes arrive.

## Public surface

```python
class TwinEvidenceEnvelope:
    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> TwinEvidenceEnvelope: ...

def load_evidence(
    path: str | os.PathLike[str],
) -> TwinEvidenceEnvelope: ...
```

`load_evidence` is exported only from `flagquantum.twin`; it is not flattened
into the package root.

The internal Q-ATLAS bridge consumes a prospectively frozen object with schema
`flagquantum.q_atlas_twin_evidence_binding.v1`. Its complete fields are the
source validation identity, Twin snapshot identity, QPU and topology
identities, ordered physical chain, source Shadow-Twin candidate identity,
supported operations, maximum instruction count, a complete mapping from each
physical-program identity to its FlagQuantum IR circuit identity, the explicit
pre-outcome flag and zero available outcome count, and its own canonical
SHA-256 identity. This object is an input to the offline tool, not a new public
FlagQuantum type.

## Compatibility

This change is additive. The envelope schema and existing prediction and report
semantics remain unchanged. No compatibility alias or research-specific public
type is introduced.

## Acceptance

- A `to_dict()` payload round-trips with identical envelope identity.
- Files load without network or provider access.
- Missing fields, unknown fields, invalid JSON, wrong schemas, invalid digests,
  and bounds without confidence fail closed.
- Documentation explicitly rejects retrospective identity substitution.
- The Q-ATLAS converter rejects changed audits, failed gates, incomplete program
  mappings, retrospective bindings, and audits that do not descend from the
  frozen validation identity.
