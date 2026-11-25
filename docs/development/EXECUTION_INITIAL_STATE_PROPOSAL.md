# Program definitions and execution inputs

Status: approved for the specified implementation on 2026-11-12; see the
[explicit approval record](../api-changes/FQ-EXECUTION-INITIAL-STATE-20261112.md).
The bounded implementation is available locally; release certification is not
claimed. On
2026-11-12 the user confirmed the work order: separate program definitions from
runtime inputs, converge execution-argument rules, then use high-level function
semantics in optimization. That work-order direction alone did not approve a new public signature or
serialized schema; the later explicit approval covers the exact delta below.

This revision withdraws the earlier draft choices to add `initial_state` to
public `fq.plan`, bind plans to amplitude hashes, select a plan version in
advance, or make all plans accept arbitrary input by default. Those choices were
unapproved and unimplemented; no historical approval is being revoked. The
specific input/replay contract below was subsequently approved; the earlier
work-order instruction alone did not approve it.

## Ownership and existing implementations

| Concern | Existing implementation | Responsibility to preserve |
| --- | --- | --- |
| Operations, registers and effects | Circuit, CircuitIR, internal Core program model | Program definition; Compiler transforms it |
| External quantum state | Circuit inputs and direct statevector execution | Per-execution data admitted by Runtime and applied by Simulation |
| State preparation | Gates, channels and synthesis InputState | Program semantics, including preparation cost |
| Gate parameter bindings | Circuit.bind_parameters and instruction parameters | Mathematical inputs, using existing binding mechanisms |
| Compile-time constants | Synthesis capture and specialization | Compiler inputs; changed constants may require resynthesis |
| Backend, device, precision and policy | ExecutionOptions, RuntimeConfig, options_resolver | Runtime resolves execution choices against program constraints |
| Measurements, reset, feedback and syndrome rounds | Ordered program instructions and QEC workflows | In-program effects, not initial-state reassignment |
| Numerical state and caches | Circuit caches and numerical executors | Execution storage with explicit invalidation and gradient behavior |
| Results and evidence | ExecutionResult and existing artifacts | Record actual execution and outputs |

These are responsibility boundaries, not authorization to remove existing
fields. CircuitIR currently owns precision and batch constraints; preserve them
while separating per-call data. Reuse the existing
[Core program model](IR_IMPLEMENTATION_STATUS.md),
[options resolver](../../flagquantum/runtime/options_resolver.py),
[planner](../../flagquantum/runtime/planner/README.md),
[accepted-plan executor](../../flagquantum/runtime/plan_execution.py) and
[synthesis workflow](../../flagquantum/services/synthesis.py). Do not introduce a
second IR, resolver, registry, request envelope or numerical engine.

## First correction and current supported workflow

Circuit currently holds both a program and live inputs. Circuit.to_ir preserves
operations, dtype, shape and runtime metadata, but not those input tensors.
The shared admission correction in this working tree addresses issue #1129 by
refusing unsupported custom states before lossy conversion and compilation.
It is not yet a merged or released change.

```python
import torch
import flagquantum as fq
from flagquantum.errors import CapabilityError

circuit = fq.Circuit(
    1,
    inputs=torch.tensor([0, 1], dtype=torch.complex128),
    dtype=torch.complex128,
)
torch.testing.assert_close(
    circuit.state().reshape(-1),
    torch.tensor([0, 1], dtype=torch.complex128),
)
try:
    fq.run(circuit)
except CapabilityError as error:
    assert "custom initial states" in str(error)
else:
    raise AssertionError("unsupported input must not execute as zero")
```

`fq.plan` and Circuit.run share that admission guard. Explicit zero tensors must
also be refused. Direct Circuit.state and its existing input gradients remain
supported. Gate-based preparation and default-zero plan/JSON replay retain their
behavior. The path is `single_device_fast_path`; this correction adds no backend
support or state-upload capability. Keep it separate from experiments and future
Stable Core changes.

## Initialization is also a compiler assumption

[remove_zero_state_resets](../../flagquantum/compiler/optimization/zero_state_reset.py)
removes leading resets because current CircuitIR execution starts from zero.
For input `|1>`, reset must produce `|0>`; removing it instead preserves `|1>`.
Passing an external state only after compilation therefore changes semantics.

Future external input must reach admission before compilation, representation
selection, memory estimation and numerical execution. Review initialization,
wire permutation and clean-ancilla assumptions at that boundary. Preserve
zero-input optimizations on the existing default path; do not reinterpret all
plans as arbitrary-input programs or globally disable those optimizations.

## Approved ordinary entry and explicit-plan limits

The implemented bounded user workflow is one call:

```python
result = fq.run(program, initial_state=state)
```

The approved public delta is one keyword-only
`initial_state: torch.Tensor | None = None` appended to fq.run. None retains
default initialization. Do not add live state to ExecutionOptions or require
users to construct input specifications. No fq.plan, Circuit.run, Module or QEC
signature change is included. Preserve Circuit(inputs=...).state; no constructor
deprecation is proposed. Supplying both construction inputs and the run
argument must raise TypeError rather than choose a hidden precedence.

The first implemented numerical slice accepts an ordinary static Circuit and
unbatched CPU PyTorch statevectors, without shots or noise, producing full
amplitudes or probabilities. Ordinary unitary opcodes are admitted using the
existing operator schema; channels, dynamic instructions, structured operations
and explicit matrix operations are excluded initially. CircuitIR and
precompiled sources are excluded from external-input admission in this slice:
an already optimized program may have lost operations under zero-input
assumptions. Existing CircuitIR workflows without external input are unchanged.

fq.run(program) already plans internally. fq.plan produces an accepted executable
contract, and fq.run(plan) validates and executes it without replanning.
Circuit.runtime_plan is a separate selection explanation. Neither makes an
existing zero-input plan compatible with arbitrary external data.

Existing zero-input plans retain their external-input rejection. The
variable-input contract below is admitted only on the dedicated path approved
by the API owner; it does not upgrade existing plans automatically.

## Approved input, plan and replay contract

The [machine-readable proposal](../../contracts/execution-initial-state-proposal.json)
records the exact public delta, support boundary, failure categories and delivery
gates. Implementation is authorized by the linked approval record. The
[review-only fake](../../tests/unit/test_execution_initial_state_contract_fake.py)
reuses existing planning, option resolution, serialization validation and
statevector computation to exercise input binding and replay. It does not
install an alternative public API or numerical engine. The real binder
replacement and public execution/artifact workflows are covered in
[the conformance suite](../../tests/api_contract/test_execution_initial_state.py).

The implemented extension reuses the existing versioned extension mechanism inside
the circuit ExecutionPlan schema version 1.0. It adds no top-level fields, plan
class, public input-specification type, amplitude storage or amplitude hash:

```json
{
  "namespace": "flagquantum.execution",
  "kind": "initial_state",
  "version": "1.0",
  "identity": "<existing canonical_hash of the payload below>",
  "payload": {"representation": "statevector"}
}
```

The existing plan identity already includes the complete extension list.
Dimensions, precision and device derive from the existing program and decision
fields. The first slice permits exactly this one extension; combining it with
noise is not admitted. The plan parser dispatches extension kinds instead
of interpreting every extension as a noise model. Older readers must reject the
unknown extension explicitly. Existing zero-input plans retain their identity,
schema and execution behavior.

Supported syntax within the stated scope:

```python
result = fq.run(circuit, initial_state=psi)
next_result = fq.run(result.plan, initial_state=phi)
restored = fq.ExecutionPlan.from_json(result.plan.to_json())
replayed = fq.run(restored, initial_state=psi)
```

Changing compatible amplitudes is a new computation under the same plan identity;
replay validates and executes without planning or compiling. A marked plan
without initial_state raises ValidationError. Supplying initial_state to an
existing zero-input plan raises CapabilityError, without silently replanning.

### Validation, ownership and gradients

Accept exact torch.Tensor and torch.nn.Parameter types with dense strided CPU
storage, shape `(2**n_qubits,)`, MSB ordering and complex64 or complex128 matching
resolved precision exactly. Noncontiguous input is accepted. Unsupported tensor
subclasses, layouts and devices raise CapabilityError; malformed shape, dtype,
nonfinite values and nonunit norm raise ValidationError. Non-tensor input or
conflicting construction/run input spellings raise TypeError. Do not coerce,
renormalize, downcast, change requires_grad or detach the numerical input.

Validate squared norm against one with absolute tolerance 1e-5 for complex64
and 1e-12 for complex128, and zero relative tolerance. Validation uses detached
chunks of at most 4096 elements; execution receives one contiguous,
differentiable clone per call. The caller owns the original; later mutation
cannot alter the admitted clone. Concurrent mutation during admission is
unsupported. Neither live input nor its graph is retained in the plan.

First-order native PyTorch gradients must reach both input and live gate
parameters. For a new direct request, infer require_gradients from these tensors
only when the call leaves it unspecified, through the existing resolver's
policy_options. Explicit False conflicting with a gradient-bearing input or
parameter raises ValidationError. A gradient-bearing input supplied to an
accepted non-gradient plan raises CapabilityError without replanning. JSON
restoration does not restore an original gate-parameter autograd graph.
Gate values remain part of the sealed program fingerprint. Replay rejects
changed live gate bindings before copying or executing the input. A training
loop that updates gates runs the source Circuit again; input-only changes may
reuse an accepted compatible plan.

### Compilation and memory

Reuse planner.plan_advanced with optimize=False for this external-input route.
Its assumption is an arbitrary full-register state, with no introduced work
qubits. This conservative first slice admits no clean-ancilla optimization;
later optimization needs explicit arbitrary-input legality evidence. Preserve
the current optimized default-zero route.

Before cloning, check the minimum budget of plan.state_bytes plus
input.numel()*input.element_size(). Caller-owned storage is excluded from that
runtime budget. This preflight does not certify bounded kernel workspace,
autograd storage or process RSS. Insufficient budget raises CapabilityError
before the copy. Do not cache the input clone across executions.

### Persistence and compatibility

Plan JSON stores the input requirement, not amplitudes. Exact reproduction
requires the caller to supply the same input and required environment. The
existing .fqx result record must preserve the extension and outputs without
storing initial amplitudes or autograd graphs. Loading never executes; the
existing loaded result.plan remains None, and Experiment.execution_plan is the
historical record. The caller may restore an ExecutionPlan from that record
and explicitly supply the input again.

The existing zero-input plan-only .fqx route writes only the program and drops
the execution record. For a marked external-input plan this would lose its
required input semantics, so this slice raises SerializationError before
writing in that case. Existing zero-input artifact behavior is unchanged.
Artifact round trips and rejection before writing are covered by the public
conformance suite; the review-only input fake alone does not establish them.

The keyword is an additive API change. Existing users need no source migration;
Circuit(inputs=...).state remains available, and existing plans are never
automatically upgraded. External-input plans require a reader that implements
this extension. The explicit approval precedes implementation; a release note and reviewed
candidate contract record the exact change. Historical snapshots are preserved.
The release version is to be assigned by the release owner at implementation.

## Execution arguments and function semantics: next stages

Keep resolve_execution_options as the single resolver. Preserve its reviewed
precedence: call options, runtime policy, program constraints, RuntimeConfig,
then defaults, followed by constraint validation. fq.run(target=...) names a
remote destination; ExecutionOptions.target names an output objective. Document
those meanings without silently renaming stable fields.

Inventory ordinary Circuit, synthesis Program and LindbladPlan bindings as
mathematical inputs, synthesis constraints, execution choices or already sealed
decisions. Preserve each product's existing override checks until its owning
contract explicitly permits another binding. Converge one verified workflow at
a time instead of replacing these guarantees with a permissive dispatcher.

The current root entry already distinguishes the following cases:

| Input to fq.run | Current binding/override rule | Existing owner to reuse |
| --- | --- | --- |
| Circuit or CircuitIR | Resolve admitted execution options; compose planning and execution | Runtime planner and options_resolver |
| ExecutionPlan | Execute the accepted contract; refuse semantic overrides | execution_plan_contract and plan_execution |
| Synthesis Program | Validate its selected product; reject all root execution overrides | services.synthesis.Program and its checked execution plan |
| LindbladPlan | Execute the sealed plan; reject all root execution overrides | lindblad plan and its Runtime executor |
| IsingBraidProgram | Use native planning; no remote target, direct-shot or noise-model route | Runtime native Ising planner |

The first argument-convergence candidate is the ordinary local Circuit/IR route:
follow existing program precision/batch constraints and resolver provenance from
call to accepted plan to result. Keep synthesis and native-plan restrictions
intact. Their relaxation needs its own user workflow and domain contract review;
it is not a necessary side effect of supporting external statevector inputs.

For function-level optimization, extend the existing Core program model and
private structured Compiler path. Preserve typed inputs/outputs, source origins,
effects and resource lifetimes through the point where implementation choices
are made. Use existing arithmetic candidates and allocation traces. The first
study should vary one implementation or compute/uncompute choice while keeping
function, preparation, outputs, cleanup and execution constraints fixed. Verify
outputs and work-register cleanliness independently, and measure resources and
execution, including cases with no improvement. A logical gate tally alone does
not establish physical execution cost.

## Delivery gates

| Order | Deliverable | Acceptance |
| --- | --- | --- |
| 1 | Shared initial-state admission and corrected boundary documentation | Early rejection, preserved direct state/gradients, gate preparation and default plan replay |
| 2 | Reviewed external-input run contract | Concrete input, compilation, result, replay, mutation, memory and gradient behavior; API-owner approval; required Integration contract, fake and conformance tests |
| 3 | Complete CPU external-input workflow | Independent amplitude/gradient references, explicit unsupported failures and preserved no-input correctness/performance |
| 4 | One execution-argument workflow converged | Existing resolver reused, supported equivalent spellings agree, sealed constraints preserved |
| 5 | One function-level optimization decision | Function/effects/cleanup preserved and honest measured resource/execution comparison |

Dynamic external input needs its own complete path: fresh state and classical
history per shot, ordered collapse and feedback, local ancilla reset and repeated
syndrome rounds. Logical preparation and encoding remain in the program.
Specialized QEC samplers retain their lifecycle and decoder contracts. Unsupported
inputs fail before numerical allocation or RNG draws. Density matrices,
stabilizer/Pauli inputs, MPS/TN and distributed execution need their own
representation-specific admission and evidence.

Use existing admission, execution-plan, options, zero-state reset, dynamic and
QEC tests first. Run required default/runtime tiers for production changes and
report failures without weakening checks. Add no tests solely for the wording
or existence of this document.
