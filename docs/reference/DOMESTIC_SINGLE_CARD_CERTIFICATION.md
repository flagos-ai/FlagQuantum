# Domestic single-card certification harness

FlagQuantum provides a vendor-neutral, fail-closed harness for executing the
split real/imag P0-P5 numerical stack on one physical domestic accelerator
through Torch-FL's logical `flagos:0` device. The harness is an evidence
collector. Passing it creates a review candidate; it does not automatically
promote hardware, performance, convergence, distributed, or production claims.

## Ownership boundary

Torch-FL and the machine provisioner own physical-device detection, vendor
runtime integration, and the audit that maps `flagos:0` to the physical card.
FlagQuantum contains no vendor names, SDK imports, or vendor-specific dispatch.
It accepts only a provisioner-owned attestation matching
[`ci/domestic_single_card_attestation.template.json`](../../ci/domestic_single_card_attestation.template.json).

The attestation must identify the vendor, model, architecture, device UUID or
serial, driver, vendor runtime, Torch-FL revision, immutable container digest,
and physical-device probe digest. It must also assert that the internal provider
route was audited, logical-to-physical mapping was verified, and CPU fallback
was neither allowed nor observed. Placeholder or self-unverified input is
rejected before PyTorch or FlagQuantum execution code is imported.

## Acceptance matrix

The immutable contract is
[`domestic-single-card-certification-contract.toml`](../../domestic-single-card-certification-contract.toml).
One run covers:

- P0 split-FP32 forward state evolution;
- P1 expectation values and parameter-shift gradients;
- P2 selective Double-Single reductions;
- P3 full Double-Single state evolution;
- P4 device-generated Double-Single gates and gradients;
- P5 explicit two-word Double-Single SGD at 1, 16, and 64 steps.

Depths 8, 32, and 128 and seeds 0 and 7 are compared with the existing CPU
complex128 diagnostic oracles. The harness checks logical-device residency and
rejects FlagQuantum host fallback. P3's declared CPU complex128 gate encoding
and all CPU complex128 reference calculations remain diagnostics, not hidden
execution fallback. P4 is the device-only gate-generation path used by P5.

## Provision and run

Copy the template outside the repository, replace every placeholder from a
provisioner-controlled runtime probe, and set the two immutable identities used
for cross-checking:

```bash
export FLAGQUANTUM_TORCH_FL_COMMIT=<40-character-revision>
export FLAGQUANTUM_FLAGOS_CONTAINER_IMAGE=sha256:<64-character-digest>
python tools/validate_domestic_single_card.py \
  --attestation /provisioner/domestic-card-attestation.json \
  --output /tmp/flagquantum-domestic-single-card.json
```

The process imports Torch-FL before PyTorch, verifies runtime vendor and device
name against the attestation, requires a clean exact FlagQuantum source
revision, executes the fixed matrix, synchronizes the logical device, and emits
one JSON record containing source, contract, attestation, runtime, memory, and
P0-P5 results. A source archive without Git metadata must provide the exact
revision and clean-tree state through `FLAGQUANTUM_SOURCE_REVISION` and
`FLAGQUANTUM_SOURCE_TREE_DIRTY=0`.

Enable the opt-in integration wrapper on the same host with:

```bash
FLAGQUANTUM_DOMESTIC_ATTESTATION=/provisioner/domestic-card-attestation.json \
pytest tests/test_domestic_single_card_certification.py -v
```

## Claim boundary

A successful raw run may set
`domestic_accelerator_certification_candidate=true` and records that real
domestic hardware was exercised. It must retain all of the following as false:

- `hardware_certification`;
- `convergence_certification`;
- `flagcx_collectives_validated`;
- `scalability_claim_allowed`;
- `performance_claim_allowed`;
- `production_claim_allowed`.

Promotion requires review of the raw result and provisioner attestation,
reproducibility on the locked hardware/software tuple, CUDA non-regression, and
an explicit capability-maturity change. FlagCX multi-card validation is a
separate later gate and cannot be inferred from this single-device result.
