# API Change Proposal 087: Dynamic remote stable-facade preflight

## Status

**Proposal approved on 2026-10-24. Implementation, provider activation,
networking, and task submission require separate authorization.**

The API owner authorized this decision scope with:

```text
approve API-CHANGE-087-DYNAMIC-REMOTE-STABLE-FACADE-PREFLIGHT-PROPOSAL
```

This authorization approves the ownership, ordering, fail-closed behavior,
negative provider matrix, and verification requirements in this document. It
does not authorize production-source changes, a protected contract edit, a
provider activation, network access, a mocked or live task submission, or a
hardware-support claim.

API Change 086 froze the dynamic remote capability contract and implemented the
internal requirement/snapshot bridge. This proposal governs delivery-sequence
item 3 from that decision: make the stable remote execution facade and the
inspectable assessment path use one fail-closed admission decision before any
irreversible provider boundary.

This proposal does not authorize a live device call, a paid task, a provider
support claim, or the experimental Braket IQM vertical slice.

## Current user workflow

The supported credential-free workflow remains the existing assessment example:

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
    supports_dynamic_circuits=True,
)
report = fq.experimental.dynamic.assess_dynamic_backend(circuit, backend)

assert not report.compatible
assert report.blockers == ("conditional_gate_not_in_backend_basis:h",)
```

Run the maintained form with:

```bash
python -m examples.local.dynamic_deployment_preflight
```

This workflow does not use credentials, discover hardware, construct a provider,
compile a provider artifact, or submit a task.

## Intended stable-facade behavior

The eventual stable workflow keeps the existing `fq.run` entry and requires an
explicit target:

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

This is a target-state example, not implemented behavior. The preflight stage
defined here must still refuse every remote dynamic target unless a later,
separately approved provider slice supplies all required facts and submission
boundaries.

## Problem

The repository now has a canonical dynamic requirement vector and a pure matcher,
but the stable remote facade does not yet consume that decision. Leaving the two
paths separate creates four risks:

1. `fq.run` could enter compiler, package, credential, or provider code before a
   dynamic incompatibility is known.
2. The inspectable assessment and direct execution paths could report different
   blockers for the same program and target facts.
3. A provider's coarse `supports_dynamic_circuits=True` flag could accidentally
   bypass missing detailed facts.
4. An existing static Quafu, Azure, or Braket route could receive a dynamic
   program even though no dynamic provider slice has been admitted.

The next stage must close those gaps without adding another public planner,
capability model, provider registry, or result type.

## Decision

### 1. One internal preflight decision

Runtime owns one internal preflight operation that accepts:

- validated ordered Circuit IR;
- the explicit provider and device identity parsed from the stable target;
- one device-scoped `TargetCapabilitySnapshot`; and
- the operation mode, which is assessment or execution admission.

It returns the existing provider-neutral compatibility report plus the canonical
requirement identity, snapshot identity, exact target identity, and ordered
blockers needed by later internal stages. These values remain internal. Users do
not construct fingerprints, evidence records, requirement sets, or snapshots.

The operation reuses API Change 086's requirement/snapshot bridge and matcher.
It must not implement a second feature derivation or blocker vocabulary.

### 2. Stable direct and inspectable paths agree

Two callers consume the same preflight decision:

- the current inspectable
  `fq.experimental.dynamic.assess_dynamic_backend(...)` workflow; and
- remote dynamic dispatch reached through stable `fq.run(...)`.

For the same Circuit IR and equivalent target facts they must produce the same
compatibility verdict, target identity, and ordered blocker categories. The
experimental report may expose diagnostic detail that stable `fq.run` expresses
as an actionable `CapabilityError`; it may not apply different admission rules.

`fq.plan` remains the backend-neutral local execution planner in this stage. This
proposal does not add `target`, provider credentials, or provider snapshots to its
signature, and it does not pretend that a local `ExecutionPlan` authorizes a
remote task. Inspectability is supplied by the existing assessment path until a
separate remote-plan artifact is justified and approved.

### 3. Fail before irreversible or provider-owned work

Remote dynamic dispatch performs preflight before:

1. provider SDK import or construction;
2. credential lookup;
3. discovery or network access;
4. provider compiler or dialect lowering;
5. deployment-package creation or sealing;
6. provider dry run;
7. task creation, submission, polling, or charging; and
8. random-number consumption or partial result creation.

A refusal is deterministic for the same program and snapshot. It raises a
FlagQuantum capability error containing the exact target and ordered blockers.
It never silently selects a simulator, another provider, a static rewrite, or a
local execution path.

### 4. Static provider routes remain closed to dynamic programs

This stage admits no positive remote dynamic provider. It must prove the
following target families fail before provider-owned work:

| Target family | Required preflight result |
| --- | --- |
| Quafu | Unsupported or missing detailed dynamic facts; no provider call |
| Azure | Unsupported or missing detailed dynamic facts; no provider call |
| Amazon Braket non-IQM | Unsupported target family; no provider call |
| Amazon Braket IQM with incomplete facts | Precise missing/unknown/not-exposed blockers; no provider call |
| Unknown provider or backend | Unresolved exact target identity; no fallback |

Existing static circuits continue through their current remote routes. A static
provider regression test must prove that the dynamic guard does not change their
target parsing, compilation, package identity, submission count, or result
normalization.

### 5. Routing cannot weaken requirements

This stage installs the first, pre-provider match only. It preserves the full
requirement identity for the later routing and dialect stage. No compiler or
provider projection may remove a requirement or turn a blocker into acceptance.

The post-routing match remains required by API Change 086 and will be activated
with the separately approved provider vertical slice. Until then, the absence of
that positive provider slice means remote dynamic execution remains closed.

### 6. No public API or schema expansion

This stage adds no root export, stable parameter, provider-specific option,
public class, result field, enum value, target alias, or serialized schema.

It reuses:

- `fq.run` as the stable execution facade;
- `DynamicCircuit` and Circuit IR as the semantic program source;
- `DynamicFeatureSet` as the internal feature vocabulary;
- Core `RequirementSet` and `TargetCapabilitySnapshot` as the capability
  authorities;
- `CloudBackendProfile` as the provider-neutral deployment projection; and
- `fq.ExecutionResult` as the eventual normalized result.

Provider SDK types and raw provider payloads remain outside Core and Runtime.

## Error contract

The stable refusal must be actionable without exposing internal machinery. Its
observable contract is:

- exception family: `CapabilityError`;
- exact explicit target included in the message or structured context;
- one or more deterministic blocker codes from the protected candidate
  vocabulary;
- failure before compiler, package, provider, network, task, or RNG activity;
  and
- no partial `ExecutionResult`.

Tests must cover at least missing, stale, unknown, unmeasured, unsupported,
not-exposed, wrong-target, capacity, conditional-gate, condition-width,
condition-value, disjunction, reset, repeated-measurement, and requested
mid-circuit-data blockers where the program can express the requirement.

## Implementation boundary

The implementation PR is expected to modify Runtime-owned dynamic remote
admission and the narrow stable remote dispatch boundary. It may add private
fakes and scenario-oriented tests. It must not change provider transport,
credentials, task lifecycle, public contracts, or provider package ownership.

If implementation requires coordinated edits across four or more primary
domains, work stops for boundary review rather than adding cross-layer wrappers.

## Verification plan

The implementation stage must provide:

1. a contract fake proving the same consumer can swap between an accepted
   snapshot and each fail-closed snapshot without changing the consumer;
2. equivalence tests between inspectable assessment and stable direct dispatch;
3. no-call sentinels around compiler, package, provider constructor, credential,
   discovery, network, submit, and RNG boundaries;
4. scenario tests for Quafu, Azure, non-IQM Braket, incomplete IQM, unknown
   provider, and wrong-device snapshots;
5. static remote and local DynamicCircuit regression tests;
6. deterministic blocker ordering and identity round trips; and
7. a runnable, credential-free user example showing the current inspectable
   refusal workflow.

The smallest focused tests run first, followed by the Runtime integration tier
and the protected public API checks. No live credential or device is part of the
ordinary test path.

## Performance budget

Preflight is linear in the number of instructions plus the size of the declared
feature sets. It must not perform network I/O, provider import, compilation, or
package construction. The implementation PR must retain a reproducible local
measurement and set a regression budget only after measuring the merged-base
and candidate paths on the same host.

The budget protects ordinary local and static remote users: a static program
must pay only the constant-time dynamic-program classification needed to bypass
this path.

## Delivery sequence

1. **This proposal.** Freeze ownership, ordering, failure semantics, negative
   provider matrix, and verification requirements.
2. **Stable-facade preflight implementation.** Connect the API Change 086 bridge
   to inspectable assessment and stable remote dispatch, with no positive
   provider submission path.
3. **Performance and regression evidence.** Measure preflight overhead and prove
   local DynamicCircuit plus existing static remote behavior remains unchanged.
4. **Experimental Braket IQM proposal.** Separately authorize provider dialect,
   sealed package, dry run, mocked exactly-once submission, task identity, and
   result normalization.
5. **Live evidence.** Run only under an explicit bounded authorization when
   credentials, device access, pricing, and evidence retention are known.
6. **Promotion decision.** Promote only the exact provider/device/feature slice
   supported by retained evidence.

### Preflight performance budget

The maintained `dynamic_remote_preflight` runner measures the pure fail-closed
preflight at 2, 32, and 256 dynamic instructions, plus the static-circuit type
classification that bypasses this path. Circuit construction is outside the
timed region, evaluation time is fixed so identities are reproducible, and the
runner performs no credential, provider, network, or task operation.

The preflight p95 budget is a linear envelope of 10 ms fixed allowance plus
0.55 ms per instruction. This gives limits of 11.1, 27.6, and 150.8 ms for the
three reference cases. Static classification has a separate 1 ms p95 limit. The
envelope reflects repeated local tail-latency observations while still detecting
fixed-cost and superlinear regressions; it is local non-release evidence, not a
QPU latency or provider-performance claim. CI validates the runner, schema, and
budget arithmetic but does not use shared-runner wall time as a release gate.

Reproduce the reference observation without adding generated output to the
source repository:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
flagquantum-benchmark run dynamic_remote_preflight \
  --feedback-pairs 1 16 128 --warmup 5 --iterations 101 \
  --static-iterations 1001 --json-output \
  ./dynamic_remote_preflight.json
```

## Acceptance

- [x] API owner approves this proposal and its no-submit boundary.
- [x] Direct stable dispatch and inspectable assessment use one matcher.
- [x] Equivalent inputs produce the same target identity and blocker ordering.
- [x] Quafu, Azure, non-IQM Braket, incomplete IQM, and unknown targets create no
      compiler, package, provider, network, or task activity.
- [x] The coarse dynamic-support boolean cannot admit execution.
- [x] Static remote and stable local DynamicCircuit behavior remains unchanged.
- [ ] Routing cannot weaken or erase the preserved requirement identity.
- [x] Root exports, stable signatures, and serialized schemas remain unchanged.
- [x] The PR includes a runnable credential-free user example.
- [x] Reproducible preflight and static-classifier performance evidence passes
      its local non-release budget without provider activity.

## Non-goals

- Submitting to a real or mocked provider.
- Claiming Amazon Braket IQM or any other QPU is supported.
- Adding a provider-specific stable API or another public planner.
- Adding provider or credential parameters to `fq.plan`.
- Automatically selecting a provider or falling back to simulation.
- Merging `DynamicCircuit` into `Circuit`.
- Promoting `assess_dynamic_backend`, `run_dynamic`, provider dialects, native
  task objects, dynamic noise, gradients, or distributed dynamic execution.
- Adding loops, nested control flow, conditional measurement, or conditional
  reset.
