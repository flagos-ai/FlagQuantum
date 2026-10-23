# API Change Proposal 086: Dynamic remote capability negotiation

## Status

**Proposal and candidate-contract baseline approved on 2026-10-23; Runtime and
provider activation are not yet authorized.**

The API owner authorized this decision scope with:

```text
approve API-CHANGE-085-DYNAMIC-REMOTE-CAPABILITY-PROPOSAL
```

The latest `main` already contained two unrelated documents numbered 085 when
this branch was cut. This proposal is therefore numbered 086 to avoid creating a
third ambiguous record. The administrative renumbering does not broaden the
approved scope.

This approval authorizes the architecture and delivery sequence below. It does
not authorize a protected contract edit, Runtime activation, a new provider
target, a network request, a paid task, or a stable hardware-support claim.

The API owner separately authorized delivery-sequence item 1 with:

```text
approve API-CHANGE-086-DYNAMIC-REMOTE-CAPABILITY-CONTRACT
```

That directive authorizes the protected candidate contract and failing-first
conformance baseline only. It does not authorize delivery-sequence items 2-6.

Proposal 084 completed stable local planning and execution of a
`DynamicCircuit` through `fq.plan` and `fq.run`. This proposal governs the next
boundary: deciding whether one explicitly selected remote target can execute the
same program without losing its dynamic semantics.

## User outcome

The intended eventual user workflow keeps the existing execution entry and
requires an explicit provider target:

```python
import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit

circuit = DynamicCircuit(2).h(0).measure(0, classical_bit=0)
circuit.conditional("x", 1, classical_bit=0)

result = fq.run(
    circuit,
    target="amazon-braket:<device-arn>",
    shots=128,
)
```

This code is a target state, not an implemented example. Until the later
activation PR lands, stable `fq.run` must continue to refuse unsupported remote
dynamic targets before submission.

No automatic provider selection is proposed. A local simulator, another cloud,
or a static rewrite must never replace the named QPU.

## Current runnable workflow

The current release can perform a credential-free, non-submitting assessment:

```python
import flagquantum as fq
from flagquantum.deployment import CloudBackendProfile
from flagquantum.dynamic import DynamicCircuit

circuit = DynamicCircuit(2).measure(0, classical_bit=0)
circuit.conditional("h", 1, classical_bit=0)

backend = CloudBackendProfile(
    provider="example",
    name="limited-dynamic-qpu",
    n_qubits=2,
    basis_gates=("x",),
    supports_openqasm=True,
    supports_dynamic_circuits=True,
    max_classical_bits=2,
)
report = fq.experimental.dynamic.assess_dynamic_backend(circuit, backend)

assert not report.compatible
assert report.blockers == ("conditional_gate_not_in_backend_basis:h",)
```

Run the maintained form with:

```bash
python -m examples.local.dynamic_deployment_preflight
```

It prints an incompatible verdict and the same blocker. It does not discover a
device, use credentials, or submit a task.

## Problem

The repository has most pieces of an experimental remote path, but their
current agreement is too coarse for a stable entry:

1. `CloudBackendProfile.supports_dynamic_circuits` is one boolean. It cannot
   distinguish mid-circuit measurement, active reset, repeated measurement,
   condition width, condition values, supported conditional gates, disjunction,
   or returned mid-circuit data.
2. `assess_dynamic_backend` combines that boolean, capacity, basis gates,
   Program IR round-trip, and dialect export. A true boolean can therefore look
   stronger than the provider facts that justified it.
3. `DynamicFeatureSet` already models the semantic feature vector used by local,
   Qiskit Aer, and Braket IQM conformance, but provider discovery does not make
   that vector an evidence-bearing target fact.
4. The Braket adapter has an IQM dialect, dynamic-qubit-group checks, sealed
   packaging, dry run, and mocked task lifecycle. No retained real-QPU task
   proves that this checkout executed the path.
5. Stable remote `fq.run` supports existing static provider routes. It does not
   expose an Amazon Braket dynamic target, and a local conformance result cannot
   establish remote hardware support.

The consequence is two opposite risks: accepting an incompatible paid task, or
building a second capability model beside FlagQuantum's existing target
capability and dynamic conformance models.

## Decision

### 1. Capability is a feature vector, not a boolean

`supports_dynamic_circuits` remains readable for compatibility and coarse
discovery, but it is never sufficient to authorize packaging or submission.
The remote match must cover only features the current `DynamicCircuit` can
express:

| Feature | Requirement source | Target fact |
| --- | --- | --- |
| Mid-circuit measurement | Program instructions | Supported or explicitly unsupported |
| Active reset | Program instructions | Supported or explicitly unsupported |
| Repeated measurement | Ordered measurement writes | Supported or explicitly unsupported |
| Classical feed-forward | Conditioned instruction | Supported or explicitly unsupported |
| Conditional gate set | Conditioned opcodes | Set inclusion |
| Maximum condition width | Condition clauses | Positive limit or unknown |
| Accepted condition values | Condition clauses | Set inclusion |
| Disjunctive conditions | More than one clause | Supported or explicitly unsupported |
| Returned mid-circuit data | Requested result contract | Supported, unsupported, or not exposed |

Provider-specific geometry, such as IQM dynamic qubit groups and controller-to-
target restrictions, stays in the dialect/provider validator. It must not be
flattened into a universal boolean or invented for providers that do not expose
it.

Loops, nested conditions, conditional measurement, and conditional reset are
not added speculatively. The current construction contract does not express
them. A later public construction proposal must land before they become remote
requirements.

### 2. Reuse the existing authorities

No new public capability class, planner class, provider registry, or result type
is introduced.

- Circuit IR remains the semantic source of program requirements.
- `DynamicFeatureSet` remains the internal feature vocabulary and conformance
  oracle during this delivery.
- Core Target Capabilities remains the long-term requirement/snapshot authority;
  the implementation proposal must either map the dynamic vector into its
  reviewed vocabulary or provide an explicitly temporary codec with an owner,
  removal condition, and target version.
- `CloudBackendProfile` remains the provider-neutral deployment projection.
- The provider adapter remains authoritative for vendor declarations, dialect
  legality, transport, task lifecycle, and result decoding.
- `ExecutionResult` remains the stable normalized return type.

Provider SDK objects and vendor response payloads must stop at Remote. Core,
Runtime, and the public result must not depend on Amazon Braket, IBM, IQM,
Quantinuum, Quafu, or Azure types.

### 3. Separate declaration, verification, and certification

A capability observation has three distinct meanings:

- **Declared:** a provider-owned device description states the feature. This can
  admit offline matching and packaging when the declaration is fresh and scoped
  to the exact device.
- **Verified:** a controlled provider task demonstrates that the submitted
  feature behaved as required on the named target. This can support an
  experimental execution claim for that target and observation window.
- **Certified:** retained, independently reviewable evidence meets the release
  policy. Only this level can support stable provider promotion.

Missing, stale, ambiguous, unmeasured, or not-exposed mandatory facts fail
closed. Marketing documentation, provider family names, simulator success, and
SDK syntax are not substitutes for a device-scoped fact.

### 4. Match before every irreversible boundary

The required order is:

1. validate Circuit IR ordering and measurement-to-condition def-use;
2. lower and restore the internal Program IR exactly;
3. derive the dynamic requirement vector from the program;
4. resolve one explicit target identity and one fresh provider snapshot;
5. match generic dynamic features and capacity;
6. legalize, route, and lower into the named provider dialect;
7. re-match the routed artifact and provider-specific restrictions;
8. seal the deployment package and identity;
9. run the provider's non-submitting dry run; and
10. submit exactly once only when every prior stage is compatible.

Any refusal consumes no RNG state, creates no provider task, incurs no charge,
and produces no partial stable result. No later stage may erase an earlier
blocker.

### 5. Preserve one execution entry

The eventual stable route extends `fq.run`; it does not add
`fq.run_dynamic_remote`, a provider-specific root function, or a second public
plan type. The existing keywords `target`, `shots`, and `name` are reused.

The first positive route may support an explicit Amazon Braket IQM target only.
That target value is additive stable behavior and therefore requires a separate
implementation authorization and candidate contract before activation.

`fq.experimental.dynamic.assess_dynamic_backend` and `run_dynamic` remain
available during the migration. They are not promoted by this proposal. Direct
user access to `run_dynamic` cannot enter deprecation until provider and
diagnostic consumers have replacements and a documented release window begins.

### 6. Normalize remote results without inventing data

The remote route returns `fq.ExecutionResult` and records the exact target,
deployment identity, provider task identity, requested/completed shots, and
dynamic capability evidence used for admission.

Final samples or counts are normalized only when the provider returns them. If
the provider does not expose mid-circuit outcomes, the result records that they
are unavailable and does not synthesize a measurement record. A target that
cannot satisfy an explicitly required result form is rejected before submission.

Provider-native task objects, feedback keys, controller groups, and raw result
payloads remain outside Stable Core.

## Initial provider boundary

Amazon Braket IQM is the first positive implementation candidate because the
repository already contains one service-owned adapter, an IQM dialect, dynamic
feature conformance, dynamic-qubit-group validation, package identity, and a
mocked task lifecycle. This is implementation reuse, not a support claim.

Quafu and Azure remain negative or unknown dynamic candidates until their exact
backend declarations and adapters expose sufficient facts. Their current static
remote routes must reject a dynamic program before compilation or submission.
IBM and Quantinuum hardware capabilities are relevant future targets, but no
FlagQuantum adapter or root execution behavior is inferred from vendor support.

## Compatibility

The proposal itself changes no export, signature, default, exception behavior,
Circuit IR schema, ExecutionPlan schema, DeploymentPackage schema, or result
field. Local `fq.plan` and `fq.run` behavior remains protected by proposals 062
and 084 and their stable contracts.

The later remote activation is additive but protected because it extends the
accepted behavior of stable `fq.run`. It must have a candidate contract,
executable semantic checks, zero-change behavior for existing static remote
targets, and explicit API-owner approval.

An unknown dynamic target continues to fail rather than run locally. A provider
that previously accepted static circuits must not start accepting dynamic
circuits merely because its coarse profile boolean is true.

## Delivery sequence

1. **Candidate contract and failing-first conformance.** Freeze the feature
   vector, evidence states, blocker categories, no-submit guarantee, result
   normalization, and proposed explicit target spelling. No transport call.
2. **Requirement and snapshot bridge.** Derive requirements from Circuit IR and
   project provider-owned facts without a second public capability model. Prove
   unknown, unsupported, stale, and not-exposed facts fail closed.
3. **Stable facade preflight.** Route remote dynamic requests through the same
   preflight from direct and inspectable paths. Existing Quafu and Azure routes
   prove rejection before compiler, package, and provider calls.
4. **Experimental Braket IQM vertical slice.** Reuse the existing dialect,
   package, dry run, submit, task, and result boundaries behind the candidate
   behavior. Mocked transport proves exactly-once submission and identity.
5. **Live evidence and performance.** Run an explicitly authorized bounded task
   only when credentials and device access exist. Record task identity,
   capability snapshot, artifact identity, shots, result, elapsed time, and
   limitations. Offline evidence alone cannot pass this gate.
6. **Promotion decision.** Promote only the exact verified provider/feature
   slice. Keep other targets experimental, unsupported, or unknown by name.

Every implementation PR must include a runnable user-facing example and must
not require credentials or paid hardware in its ordinary test path.

## Acceptance

- [x] The API owner approved the proposal scope; the administrative renumbering
      from the requested 085 label to 086 is recorded.
- [x] A protected candidate contract receives separate explicit authorization.
- [x] Dynamic program requirements are derived from IR rather than caller or
      provider claims.
- [x] One coarse boolean cannot make a target executable.
- [x] Missing, stale, unknown, unmeasured, unsupported, and not-exposed facts
      produce distinct actionable blockers.
- [ ] Direct and inspectable remote paths use the same match and target identity.
- [ ] Routing and dialect lowering cannot weaken requirements.
- [ ] Incompatible Quafu, Azure, non-IQM Braket, and incomplete IQM profiles
      create no task.
- [ ] A mocked Braket IQM path submits exactly once and retains package/task
      identity through `ExecutionResult`.
- [ ] A live claim cites retained real-QPU evidence or remains explicitly absent.
- [ ] Static remote and stable local execution retain current behavior and
      performance.
- [ ] Root exports and existing serialized schemas remain unchanged.

## Non-goals

- Automatically selecting a provider or silently falling back to simulation.
- Claiming that every IQM, IBM, Quantinuum, Quafu, Azure, or Braket device
  supports the same dynamic semantics.
- Adding loops, nested control flow, conditional measurement, or conditional
  reset to `DynamicCircuit`.
- Promoting dynamic noise, gradients, distributed dynamic execution, provider
  dialects, native task objects, or `DynamicExecutionResult`.
- Merging `DynamicCircuit` into `Circuit`.
- Treating a simulator, dry run, successful compilation, or provider
  declaration as real-QPU execution evidence.
