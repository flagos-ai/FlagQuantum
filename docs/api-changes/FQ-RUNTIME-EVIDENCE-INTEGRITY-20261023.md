# Keyless runtime-evidence integrity

Status: approved by the repository owner on 2026-10-23; implemented and merged
in [PR #728](https://github.com/flagos-ai/FlagQuantum/pull/728).

## Problem and user journey

The owner requested completion of issues 708 and 709 and asked to remove the
signing-key requirement for future measurements. The A800 pair is available,
but neither its active environment nor the repository Actions secrets provide
`FQ_EVIDENCE_SIGNING_KEY`. Requiring an unrelated secret prevents an operator
from recording otherwise reproducible measurements.

A future hardware run should emit measurements, their complete provenance and
a canonical checksum without a secret. Reviewers should reproduce that checksum
and inspect the recorded source revision, workload, devices and raw log. This
proves integrity against the recorded checksum; it does not authenticate a
publisher. Trusted repository review and merge control remain the publication
boundary. An attacker able to replace both a payload and its checksum can
recompute the checksum, so SHA-256 is not described as a signature.

## Proposed change

1. Add `flagquantum_runtime_evidence_v2` for keyless envelopes. Keep the five
   unsigned root fields and the complete `RuntimeProvenance` record unchanged.
   Its `integrity` mapping contains exactly `algorithm = "sha256"` and
   `content_sha256`; it has no `signature` field.
2. Compute the digest over the same canonical unsigned field set as v1. Validate
   all provenance, measured fields, rank/device mappings, scope, capacity,
   gradient, optimizer, workload-digest and release-gate conditions independently
   of the integrity algorithm. Missing or incorrect fields still fail closed.
3. Make `create_evidence_artifact(..., signing_key: bytes = b"")` emit v2 when
   the key is empty and retain the exact v1 HMAC behavior when a nonempty key is
   supplied. No root `flagquantum` export is added.
4. Make `verify_evidence_artifact(payload, *, signing_key: bytes = b"")`
   dispatch strictly by schema. V2 requires SHA-256 integrity with no signature;
   v1 continues to require a valid HMAC and a nonempty supplied key. A v1
   artifact is never silently treated as authenticated without its key.
5. Allow `tools/seal_runtime_evidence.py` and the hardware preflight to produce
   v2 without an environment key. A configured key remains an explicit route
   to v1. Update release readers to recognize v2 and verify its checksum and
   the same scientific acceptance conditions.
6. Keep all existing signed artifacts, signatures and source digests byte for
   byte. A keyless re-seal must be explicit and retain the original provenance;
   it must not claim that the original HMAC was verified.

## Compatibility and migration

Existing calls supplying a key keep their signatures, v1 bytes, validation and
authentication guarantees. Existing v1 readers reject the new schema rather
than misread its integrity guarantee. New readers accept both versions, and v1
verification still fails when its key is unavailable.

The two optional defaults are additive call forms. The meaningful contract
change is the availability of integrity-only measured evidence for new release
measurements. API change proposal 065 explicitly excluded integrity-scheme and
schema changes; this proposal supplies the separate decision it required.

No capability is promoted, no existing evidence is rewritten, no rank count or
performance threshold is widened, and no hardware, training or provenance check
is removed. Cancellation of signing is not permission to classify a baseline
failure as a successful sharded completion. Historical full-directory audits
containing v1 artifacts still need their existing verification key; new v2
campaigns do not.

The new format is versioned rather than replacing v1 in place. There is no
removal or deprecation of the signed call form in this change. Release notes
and the evidence runbook will state the authentication difference explicitly.

## Acceptance tests

- New keyless measured runs seal and verify with no environment secret.
- Changes to measurements, provenance, source or log digests fail verification.
- Invalid scope, missing provenance, unknown algorithms and algorithm/schema
  mismatches fail before promotion.
- V1 fixtures are unchanged; correct-key verification succeeds and missing or
  wrong-key verification continues to fail.
- Keyless and signed artifacts face identical MPS/statevector/tensor-network
  scientific acceptance tests; a failed capacity point cannot pass promotion.
- Re-sealing preserves the original source, devices, command and log digest.
- Complete focused evidence/sealer/release-gate tests and current CI/GPU pass.

## Approval

Owner: repository owner requested the removal in the issue-completion chat.
Approval: the user explicitly selected approval and implementation of this exact
proposal in the issue-completion chat on 2026-10-23.
Retain this record with the implementation and its validation.

## Implementation validation

The final focused runtime-envelope, CLI sealer, environment preflight and three
capability-gate suites passed 147 tests. The broader benchmark-contract and audit
run passed 956 tests, with 22 optional skips. Tests exercise both integrity
formats, missing legacy keys, measurement and provenance tampering, keyless
promotion, and refusal to promote an incomplete capacity set. A comparison with
the previous implementation reproduced identical v1 envelopes in all four
scopes. Public API baseline and runtime contract schema checks passed without
changing their snapshots. All 41 applicable GitHub CI checks passed before merge,
including the required GPU lanes; two manual optional checks were skipped.
