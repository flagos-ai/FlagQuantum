# Deployment Bridge Stage 7 Quafu selection review

Updated: 2026-09-03

## Decision

The owner-supplied labels `provider=Quafu` and `sandbox=Quafu` are recorded. The
canonical Provider namespace is `quafu`, but Stage 7 Entry remains unauthorized
because `Quafu` identifies the platform, not a concrete verified sandbox backend.

Official PyQuafu documentation confirms that Quafu exposes discoverable experimental
backends, requires an API token for device submission and supports real quantum
backend execution. It does not identify a backend named `Quafu` or establish that such
a target is sandbox-only and non-billable. The `testbackend` string in the official
Quafu Runtime example is not sufficient evidence of a supported non-billable QPU
sandbox.

Sources checked:

- <https://scq-cloud.github.io/>
- <https://scq-cloud.github.io/tutorial.html>
- <https://github.com/ScQ-Cloud/pyquafu>
- <https://github.com/ScQ-Cloud/quafu-runtime>

## Direction boundary

This Stage 7 path concerns outbound execution: FlagQuantum would submit an artifact to
a Quafu-controlled sandbox. The existing `flagquantum-quafu-adapter` serves the inverse
integration direction: a Quafu-facing service invokes FlagQuantum simulation or a
digital twin. That adapter cannot qualify or activate the outbound transport, and it
must not contain FlagQuantum core source.

## Missing target evidence

Before Stage 7 Entry can be reviewed, the owner must provide:

- the exact Quafu backend identifier, not the platform name;
- authoritative confirmation that it is sandbox-only and non-billable;
- the approved API endpoint set and backend-discovery rule;
- credential scopes and revocation behavior;
- quota, shots, queue, timeout and retention limits;
- region, jurisdiction, data-handling and incident ownership.

No credential value should be placed in the command, repository, issue or pull
request.

## Next owner action

Once the above evidence exists, record the exact target with:

```text
approve DEPLOYMENT-BRIDGE-STAGE7-TARGET-QUALIFICATION provider=quafu backend=<exact-backend-id> non_billable=true sandbox_only=true evidence=<authoritative-reference>
```

This records target qualification only. It does not authorize implementation,
credentials, network access or submission.
