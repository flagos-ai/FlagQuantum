# Application Services boundary inventory

Status: current boundary after removal of the pre-release Agent facade.

## Decision

`flagquantum.services` is not an agent framework and not a universal facade over
FlagQuantum. It contains only reusable application-level workflows whose
composition adds validation, policy, aggregation, or structured failure behavior.

The old serialized `AgentApplicationService` was removed before release. Protocol
adapters now deserialize their own requests and then:

- call `flagquantum.compiler.optimize`, `fq.plan`, `fq.run`, or another stable API
  directly for a one-step operation;
- call `flagquantum.services` only for a shared multi-step workflow.

## Current surface

| Workflow | Added value | Does not do |
| --- | --- | --- |
| `capabilities()` | aggregates installed backend capabilities and isolates individual discovery failures | device control, target selection, protocol serialization |
| `preflight_execution()` | combines program validation and planning; returns structured blockers | execution, persistence, autonomous decisions |
| `preflight_deployment()` | validates target compatibility, builds and identity-checks a submission package | remote submission, credentials, approval |

The report dataclasses are data returned by these workflows, not a second Core
contract family. Standalone validation is an internal step because exposing it here
would duplicate the owning validation APIs without adding orchestration value.

## Forbidden contents

- pass-through wrappers around stable API calls;
- MCP, REST, gRPC, CLI, or notebook protocol models;
- LLM prompts, agent policies, tenant state, persistence, or credentials;
- numerical kernels, provider SDKs, direct device control, or remote submission;
- generic service registries or managers without a concrete workflow.

## Verification

`tests/team/services/` checks deterministic behavior and operation without MCP
installed. `tests/test_service_preflight.py` covers the public workflows.
`architecture.toml` prevents Services from importing numerical or backend
implementation layers.
