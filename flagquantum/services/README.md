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

`run_managed_quafu_simulator(...)` is the other intentional composite workflow:
it validates the binding between a managed `<device>-sim` target and a
timestamped device noise profile, executes the stable exact/MPS policy, and
returns an auditable calibration-and-routing receipt. It is a service-side
deployment hook, not a replacement for the client's remote Quafu job path.

`run_ground_state_vqe(...)` is the provider-owned variational ground-state
workflow. It accepts a versioned Pauli sum, an explicit fixed-sector ansatz, and
a bounded optimizer configuration. The result includes the convergence trace,
term expectations, final statevector, resources, FlagQuantum package/tool
identity, and a deterministic execution digest. Unsupported Hamiltonians,
sectors, ansatzes, and optimizers return `status="unsupported"`; the workflow
never falls back to another provider. `exact_sector_reference(...)` is a
separate dense reference operation so its evidence cannot be confused with the
variational simulation.

The experimental research-capability surface adds deterministic discovery and
preflight over that bounded workflow. It currently describes exactly three
operation identities: `ground_state.solve.vqe`, `ground_state.solve.exact`, and
`validate.energy_variance`. `search_research_capabilities(...)` accepts only
structured filters; it does not parse natural language or make research
decisions. Validation and estimation run before numerical work and support only
the existing two-qubit local CPU scope. Target facts and fallback authorization
remain owned by Core and Runtime; MCP and other protocol adapters remain outside
this repository.

The first M3 routing slice accepts an explicit `provider_order` for
`ground_state.solve.vqe`. A list with more than one provider is rejected unless
`allow_provider_fallback=True`, and authorized fallback advances only past an
unknown or locally unavailable provider. Execution failures and scientific
validation failures never trigger substitution. The Evidence Bundle retains the
ordered route, every attempt, the selected provider, and whether fallback was
used. Only `flagquantum-native` is installed today; provider registration is not
a public mutable API.

`list_research_schemas()` and `describe_research_schema(schema_id)` publish the
installed JSON Schema Draft 2020-12 documents for those operations. They are
resource lookups, not additional scientific operations. An edge adapter must
serve these documents from installed FlagQuantum and must not keep a copied
schema catalog. Unknown schema identities fail closed, and each lookup returns
an isolated copy so an adapter cannot mutate process-wide contract state.

```bash
python -m pytest tests/team/services tests/test_service_preflight.py -q
python tools/check_architecture.py
python tools/check_dependency_policy.py
```
