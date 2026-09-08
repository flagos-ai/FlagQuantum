# Execution Provider inventory

Status: in progress

Updated: 2026-09-08

## Responsibility

Execution Providers adapt complete external execution targets. They discover a
target, submit one prepared program, observe the remote task and convert its result
into FlagQuantum-owned data. Vendor SDK objects, credentials, wire formats and
provider-specific status values stop at this boundary.

Execution Providers do not:

- compile or package programs;
- choose an execution target or schedule retries;
- implement simulation numerics;
- manage CPU/GPU devices, streams or communication;
- define user-facing execution semantics.

The intended flow is:

```text
Deployment builds and validates a package
    -> Runtime owns lifecycle policy
        -> Execution Provider adapts one external target
            -> provider-native service or QPU
```

`runtime/executors` are different: they execute already selected internal plans.
`providers/execution` isolates external systems and remote task protocols.

## Maintained implementations

| Module | Role | Evidence level |
| --- | --- | --- |
| `local.py` | In-memory contract fake for deployment package and identity flow | Local tests only |
| `http.py` | Reusable HTTP transport implementation for OpenQASM-style services | Mock transport tests |
| `quafu.py` | Quafu submission, polling, cancellation, result parsing and chip information | Mock contract tests |
| `braket.py` | Amazon Braket profile, preview, submission and result adaptation | Fake SDK tests |
| `result_parsing.py` | Shared remote counts normalization helpers | Unit tests |
| `quafu_calibration.py` | Quafu calibration data to FlagQuantum noise-model conversion | Unit tests |

Tencent, OriginQ, FieldQuantum, Tianyan and Guodun adapters were removed before
release. They had only placeholder or mock-level coverage and are not retained as
compatibility imports.

## Current ownership problem

Concrete adapters live in `providers/execution`, but their shared
`QuantumProvider`, `ProviderTaskHandle` and `DeploymentResult` definitions still
live in `deployment/cloud.py`. Consequently the implementation directory is not yet
the authority for its own provider lifecycle.

This should be corrected without inventing a universal Provider abstraction:

1. keep program construction and validation in Deployment;
2. move the existing provider lifecycle and provider result ownership into
   `providers/execution`;
3. let Runtime own polling, timeout, retry, cancellation and recovery policy;
4. project provider-native results and failures through Core-owned portable
   contracts when that contract is approved.

## Current lifecycle

The shared behavior currently consists of:

1. `discover_backends(n_wires)`;
2. `submit(package) -> ProviderTaskHandle`;
3. `query_status(handle) -> str`;
4. `fetch_result(handle) -> DeploymentResult`;
5. `run(package)` as a convenience path.

The convenience path is not a production Runtime driver. Quafu performs its own
polling, Braket may block through its SDK, and the generic base performs only one
status query. These differences must remain visible until Runtime owns a common
lifecycle policy.

## Known gaps

- Status is still a provider string rather than a closed portable state.
- Provider failures use built-in exceptions without a common retry category.
- `DeploymentResult` and stable `ExecutionResult` are separate result shapes.
- Handles may contain provider-native objects and are not guaranteed portable.
- Package identity covers the submitted artifact, not necessarily a provider's
  later recompilation, physical mapping or calibration snapshot.
- Current tests do not constitute real QPU availability or performance evidence.

These gaps must not be hidden by permissive metadata or silent conversion.

## Completion evidence

Execution Provider convergence is complete only when:

- Simulation and one QPU or remote-service implementation satisfy the same small
  request, status, result and failure conformance suite;
- replacing a provider does not modify Runtime or the user API;
- handles and portable results contain no credentials or live SDK objects;
- backend substitution, bit-order conversion, recompilation and fallback are
  explicit in result evidence;
- `deployment/cloud.py` no longer owns provider lifecycle definitions.

Until then, capability claims remain limited to their recorded test or hardware
evidence. Directory placement alone is not completion evidence.
