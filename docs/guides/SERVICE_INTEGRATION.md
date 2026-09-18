# Service and protocol integration

Use the shortest authoritative path. A one-step operation calls its stable API
directly; a reusable multi-step preflight uses `flagquantum.services`.

```python
import flagquantum as fq
from flagquantum.compiler import optimize

circuit = fq.Circuit(2).h(0).cx(0, 1)
optimized = optimize(circuit)
plan = fq.plan(optimized)
result = fq.run(optimized)
```

Do not add a service wrapper around these calls merely to give MCP, REST, CLI,
or Jupyter another entry point. Each edge adapter decodes its request and invokes
the same stable API.

Services is appropriate when the composition itself is reusable. Execution
preflight combines validation and planning while converting expected failures
to structured blockers:

```python
from flagquantum.services import preflight_execution

report = preflight_execution(circuit, target="expectation")
if report.executable:
    selected_backend = report.selected_backend
else:
    blockers = report.blockers
```

Deployment preflight validates the program against a target and creates and
identity-checks the exact package that may later be submitted. It does not
contact a provider or consume remote capacity.

```python
import flagquantum.deployment as deployment
from flagquantum.services import preflight_deployment

backend = deployment.CloudBackendProfile.simulator(2)
report = preflight_deployment(circuit, backend=backend, shots=1024)
if report.approved_for_submission:
    package = report.package
```

Protocol adapters may serialize `report.to_dict()`. Authentication, approval,
tenant state, job persistence, and paid-resource submission remain responsibilities
of the consuming application, not FlagQuantum Services.

## A worked MCP adapter

[`FlagQuantum/mcp-servers`](https://github.com/FlagQuantum/mcp-servers) is an
MCP server that follows the boundary described above. It is a separate package
with its own release cycle, it is not imported by this repository, and it adds no
dependency here. It decodes its own requests, validates them against this API's
own rules before calling it, and reaches only the stable surface — no private
path, no reimplemented domain logic.

It also stops where an edge adapter must stop: it plans execution but never runs
a circuit, holds no credentials, and does not submit to a provider. That is the
same split [LONG_HORIZON_ARCHITECTURE.md](../architecture/LONG_HORIZON_ARCHITECTURE.md)
draws in Section 7.3 for `LLM / IDE / MCP Host -> thin adapter -> public API`,
and its migration ledger names the exit condition as *no production MCP transport
dependency in this repository* — which is why an adapter of this kind belongs
outside the repository rather than in it.

Read it as one worked example of the shape, not as an endorsement of a particular
protocol or a required component.
