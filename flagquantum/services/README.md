# Services

Services contains small, reusable application workflows that add behavior beyond
one stable FlagQuantum API call, such as capability aggregation, execution
preflight, and deployment preflight.

The private `_synthesis` CS-04 trial composes restricted Compiler capture and
arithmetic lowering, explicit input preparation, existing resource estimation,
and bounded named views of existing Runtime results. It does not execute during
synthesis or add a numerical kernel. This trial uses the private construction
seam described in its [boundary record](../../docs/api-changes/FQ-SYNTH-CS04-20261029.md);
it is not a stable Services export or a public `flagquantum.synth` API.
Start with [the executable workflow](../../examples/algorithms/synthesis_trial.py)
and `tests/integration/test_synthesis_python_trial.py`. Supported syntax,
result-identity checks, preparation composition and limitations are in the
[delivery record](../../docs/roadmap/CS04_PYTHON_TRIAL_DELIVERY.md).

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

## Research service development boundary

An agent may write exploratory FlagQuantum code for a new scientific question.
A service is different: it is a repeatable operation with a declared input
schema, supported domain, resource bounds, numerical reference, failure
behavior, and versioned evidence. Generated exploratory code does not become a
service merely because it produced one plausible result. The agent or client
owns natural-language interpretation, task decomposition, session state, and
approval; services own only the bounded scientific operation and its evidence.

Promote an exploratory workflow into this package only when independent
reference cases and tolerances cover its supported domain, unsupported inputs
fail explicitly, and a clean installed distribution reproduces the result.
Keep molecular-model construction separate from a VQE operation until that
construction has its own validated contract. A successful VQE over a supplied
Pauli sum must not be described as end-to-end molecular energy prediction.

The proposed sequence after this initial two-qubit CPU slice is to stabilize
the capability identifiers, schemas, resource estimates, and evidence bundle;
then add scientific scopes one at a time with independent validation. Additional
providers must implement the same contract. Provider fallback remains explicit
and recorded, never a hidden response to execution or validation failure.
This is a development direction, not a claim that those scopes or providers are
available in the current release.

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
