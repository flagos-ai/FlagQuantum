# Agent

The Agent package provides deterministic, protocol-neutral validation,
preflight, and application services for automation clients. It does not
implement an autonomous or LLM-backed agent. `AgentApplicationService` exposes
capability discovery, program validation, and execution planning over
FlagQuantum-owned serialized programs and structured results.

This package does not contain an LLM, MCP/REST/gRPC transport, authentication,
tenant or job persistence, provider SDK calls, device control, or numerical
kernels. Those concerns belong to an external gateway or Compute Service and
to the existing FlagQuantum domains. Natural-language reasoning may consume
the structured output, but it cannot replace its validation or planning facts.

## Current path

```text
serialized CircuitIR or ProgramArtifact
  -> AgentApplicationService or preflight API
  -> decode and fail-closed artifact checks
  -> validation or preflight operation
  -> structured capability, validation, or planning dictionary
```

Start in `preflight.py` for direct Python validation and planning, or in
`service.py` for serialized automation requests. Keep request decoding and
protocol-neutral result projection in the service. The current service
intentionally has no `execute` or standalone `explain` method; do not advertise
or add those through a transport shortcut.

## Ten-minute change path

For a small service behavior change, modify `service.py`, add a deterministic
success or fail-closed case, and run:

```bash
python -m pytest tests/team/agent \
  tests/integration/test_cpu_vertical_slice.py -q
python tools/check_architecture.py
python tools/check_dependency_policy.py
```

Preserve operation ordering, input immutability, structured blockers, and
operation without MCP/FastMCP installed. Changes to Core artifacts, public
planning contracts, or cross-repository schemas require their owning contract
process rather than a new Agent-local type.
