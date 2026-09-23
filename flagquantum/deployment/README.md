# Deployment

Own target-neutral packaging: turn a compiled circuit into a validated
deployment package, a Pauli measurement plan, or a routing-evidence artifact
that a remote target can be handed. External submission, status polling, result
decoding, and provider evidence belong to Remote.

This domain must not own compiler passes, simulation algorithms, credentials, or
provider SDK objects. Everything here is CPU-only and offline: no function in
this package contacts a target.

## Choose an entry point

- `flagquantum.deployment` (`__init__.py`): the supported surface —
  `create_deployment_package`, `validate_deployment_package`,
  `create_pauli_measurement_plan`, `deploy_circuit`, and the counts-to-expectation
  helpers `expectation_z_from_counts`,
  `hamiltonian_expectation_from_counts`, and
  `hamiltonian_expectation_from_grouped_counts`.
- [`cloud.py`](cloud.py): those entry points plus `CloudBackendProfile` and
  `DeploymentPackage`.
- [`routing_evidence.py`](routing_evidence.py): `stable_payload_sha256`,
  `deployment_artifact_sha256`, and the routing-plan validator used before an
  artifact is handed to a target. Module-level, not re-exported.
- [`artifact_dry_run.py`](artifact_dry_run.py): `prepare_artifact_deployment`,
  the offline rehearsal for an artifact that has not been submitted.

A package is identified by the identity recorded in it; a package whose
identity does not match its program is refused rather than repaired.

## Shortest change path

1. Change one of the modules above; `deployment/**` is owned by the `remote`
   team, so start from its assigned branch.
2. Run the deployment checks from the repository root:

   ```bash
   python -m pytest tests/unit/test_deployment_routing_evidence.py \
     tests/unit/test_grouped_count_validation.py \
     tests/unit/test_architecture_boundaries.py -q
   ```

3. For anything that changes a package field or a counts contract, also run the
   default tier, because `flagquantum/twin/` and `flagquantum/runtime/` consume
   these helpers:

   ```bash
   python tools/ci_tier.py pr-default
   ```

Counts validation stays fail-closed: a histogram that cannot be attributed to
the submitted program is refused, never partially interpreted.
