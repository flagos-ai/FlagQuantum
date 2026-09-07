# Agent integration

FlagQuantum keeps language models outside the numerical runtime. An agent may
propose a workload, but deterministic FlagQuantum validation, planning, and
deployment contracts decide whether it can execute.

The deterministic integration API is available from the explicit
`flagquantum.agent` boundary:

```python
import flagquantum as fq
import flagquantum.agent as fqa
import flagquantum.deployment as fqd

circuit = fq.Circuit(2).h(0).cx(0, 1)

validation = fqa.validate(circuit)
preflight = fqa.preflight_execution(circuit, target="expectation")

assert validation.valid
assert preflight.executable
```

Reports provide `to_dict()` for tool protocols and never require an agent to
parse an exception message. Expected planning and deployment failures become
typed blockers with stable error codes. Unexpected programming errors are not
hidden.

## Deployment gate

Deployment preflight creates and identity-checks the exact package that may be
submitted later. It does not contact a provider or consume QPU capacity.

```python
backend = fqd.CloudBackendProfile.simulator(2)
report = fqa.preflight_deployment(circuit, backend=backend, shots=1024)

if report.approved_for_submission:
    package = report.package
```

External agents must require user approval before submitting a paid or scarce
hardware task. They must not rebuild or mutate `report.package` after the
preflight gate.

Product-specific specifications, recipes, prompts, and orchestration belong in
the consuming agent repository. FlagQuantum owns only deterministic validation,
planning, execution, and deployment safety contracts.
