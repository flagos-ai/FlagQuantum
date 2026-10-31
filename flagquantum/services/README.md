# Services

The private `_synthesis_registers.synthesize_registers` workflow admits the
Compiler's owned fresh-register model after external source adaptation. It uses
the same `synthesis._complete` plan/check/Program factory as Python arithmetic,
then the existing `fq.run(program)` and bounded named reader. It performs no
execution or source parsing, and requires an explicit state-memory ceiling.
The public synthesize entry dispatches local PathLike .qmod files and explicitly
marked source text through Ecosystem acquisition into this workflow. Both
profiles share one Program and executor. Native inputs default to a 512 MiB
planned-state buffer ceiling and reject Python preparation arguments.
Start with `tests/integration/test_synthesis_register_workflow.py`; run the
larger original-file acceptance separately with
`python -m pytest tests/integration/test_synthesis_qmod_originals.py -q -s`.
That test retains the original 23/24-qubit dimensions, preflights its bounded
reference buffers, and raises the existing amplitude-read limit explicitly.
See the [CS07-D delivery record](../../docs/roadmap/CS07_QMOD_EXECUTION_DELIVERY.md)
for historical support; the [public source delivery](../../docs/roadmap/CS07_PUBLIC_SOURCE_DELIVERY.md)
records the current entry, common report and remaining release boundaries.
The common report uses register_widths, source_resources and resources. Native
whole-construction costs are not labeled arithmetic_resources.

Services contains small, reusable application workflows that add behavior beyond
one stable FlagQuantum API call, such as capability aggregation, execution
preflight, and deployment preflight.

The `synthesis` workflow composes Compiler capture and CS06-B finite
selection with CS06-A actual execution checks and bounded named reading.
Synthesis plans the selected program once, but never executes. Run the existing
bound product with `fq.run(program)`, then use `program.probabilities(result)`.
No user-managed schedule, physical mapping or bit decoding is needed. This is
the single implementation behind the maintained `flagquantum.synth` facade,
currently an unreleased candidate awaiting independent developer trial records.
The private `_synthesis` module was moved, not wrapped; live callers use the
public facade. Runtime and Compiler gain no reverse dependency on this workflow.
Start with [the executable workflow](../../examples/algorithms/synthesis_trial.py)
and `tests/integration/test_synthesis_python_trial.py`. Supported syntax,
input/output semantics, preparation composition and limitations are in the
[CS06-C delivery record](../../docs/roadmap/CS06_USER_WORKFLOW_DELIVERY.md)
and [private contract](../../docs/api-changes/FQ-SYNTH-CS06-C-20261029.md).
The [run admission record](../../docs/api-changes/FQ-SYNTH-RUN-IMPLEMENTATION-20261029.md)
supersedes its private-entry limitation for this implementation branch. Candidate
scope and exact exports are in the
[governed contract](../../contracts/synthesis-program-run-v1-candidate.json).
`tests/integration/test_synthesis_user_workflow.py` covers the new end-to-end
paths. Final ceilings remain flat synthesis arguments; ConstructionBudget is
separate. SynthesisError exposes checked candidate reports and flat repair
arguments for explicit retry. No alternative constraints object can conflict
with those ceilings.

`_synthesis_execution` is the private CS06-A plan-checking seam. It combines
existing sealed-plan validation, Compiler resource limits, basis membership,
topology and replayed layout evidence. It neither plans nor executes and contains
no numerical kernel. Start with
`python -m examples.synthesis_execution_contract` and
`tests/team/services/test_synthesis_plan_checks.py`, then the real scenarios in
`tests/integration/test_synthesis_execution_contract.py`. The
[contract](../../docs/api-changes/FQ-SYNTH-CS06-A-20261029.md) and
[delivery record](../../docs/roadmap/CS06_EXECUTION_CONTRACT_DELIVERY.md) state the
admitted profile and limits. CS06-C removed the CS04 hash-only admission and
unconditional route refusal. `_synthesis_readout` now consumes only the same
checked execution and its physical named ports; do not add a second independent
validator or extend public Services exports here.

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
