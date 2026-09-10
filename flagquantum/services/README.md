# Services

Services contains small, reusable application workflows that add behavior beyond
one stable FlagQuantum API call, such as capability aggregation, execution
preflight, and deployment preflight.

Simple operations do not belong here. Call `flagquantum.compiler.optimize`,
`fq.plan`, or `fq.run` directly. Do not add a service method that only forwards
arguments, renames a result, or converts a protocol payload.

```text
typed Circuit or CircuitIR
  -> validation + planning/deployment preparation
  -> structured report with blockers
```

MCP, REST, CLI, and notebook integrations are edge adapters. They decode their
own requests, call stable APIs directly for one-step operations, and use this
package only for shared composite workflows. Services contains no transport,
LLM, vendor SDK, numerical kernel, authentication, persistence, or autonomous
agent logic.

Start in `preflight.py`. A small behavior change should normally touch that file
and a scenario in `tests/team/services/` or `tests/test_service_preflight.py`.

```bash
python -m pytest tests/team/services tests/test_service_preflight.py -q
python tools/check_architecture.py
python tools/check_dependency_policy.py
```
