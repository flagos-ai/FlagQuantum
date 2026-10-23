# Proposal 085: Native Ising braid programs through shared planning and execution

## Status and approved decisions

**Approved for staged implementation on 2026-10-23.** The owner
explicitly replied "Approve the contract and conditional merge of the
documentation" (English rendering) to the five-decision request for PR #758.
This authorizes the decisions and baseline-first delivery below. It separately
authorizes normal merge of #758 only after its latest applicable checks,
complete CI/GPU workflows and repository rules pass; later PRs retain their own
merge-authorization gates. This document changes no API, IR profile, serialized
schema, numerical behavior or capability status.
The inspected baseline is main `d658df0dfea36fe9ae9929c9f5715f617e74461e`.
Proposals 079 and 083 authorize ordered matrix coordinates and terminal
instruments; they do not authorize a native anyon program or changes to
`fq.plan`, `fq.run` or `ExecutionPlan`.

The approved decisions are:

1. Add an experimental, Core-owned `IsingBraidProgram` for four sigma anyons
   with fixed total vacuum charge, one explicit fusion tree and gauge, and
   ordered adjacent exchanges. Preserve that meaning in the existing internal
   ProgramRecord rather than converting it to a one-qubit circuit.
2. Admit this frontend through existing `fq.plan` and `fq.run`, using the same
   ExecutionOptions, ExecutionPlan and ExecutionResult objects. Make circuit
   analysis optional for this native plan instead of inventing qubit counts.
3. Add native-only executable plan 1.1 and internal physical-source encoding
   1.1. Keep CircuitIR 1.0, circuit execution plans 1.0 and all Lindblad plans
   unchanged, including old bytes and identities.
4. Execute the exact finite unitary sequence through existing SuperOperator
   algebra, owned by Simulation, and reuse Runtime terminal instruments for
   nondestructive charge observations. No physical time or Hamiltonian is
   inferred from a braid ordinal.
5. Deliver a replaceable Integration baseline before Core, Simulation and
   Runtime implementations. Each PR requires its own merge authorization,
   runnable example and complete applicable verification.

The explicit owner decision above supplies implementation approval; document
review or merge alone would not supply it. Follow
[public API protection](PUBLIC_API_PROTECTION.md) and the
[shared M0-M4 roadmap](../roadmap/FLAGOS_ALIGNED_RELEASE_TRAIN.md).

## Why this is the next boundary

The roadmap requires a small braiding-and-fusion-measurement scenario. Existing
`examples.constrained_basis_reference` supplies a caller-owned two-dimensional
matrix and literal fusion-path labels. Its logarithm witness is valid as a
matrix reference, but does not preserve a native braid word, certify fusion
rules or make its numerical time a braid duration.

The internal ProgramRecord already carries typed native physical context and
operations for Rydberg programs. This is the canonical source boundary to
extend. Runtime's current ExecutionPlan instead decodes CircuitIR and requires
CircuitAnalysis, whose fields include qubit count, qubit usage and gate widths.
Neither silently inserting a native source into that schema nor declaring one
encoded qubit resolves the distinction. RuntimePlanContract is an existing
lossy audit projection; it cannot replace an executable sealed program.

The proposal deliberately reviews this cross-domain boundary before numerical
implementation. It adds no second IR, plan class, execution manager, registry,
parameter language, trainer, numerical engine or production dependency.

## Physical and mathematical scope

There are exactly four identical sigma anyons, initially ordered `a,b,c,d`, with
fixed total charge `1`. Ordered left-associated fusion paths are
`(("1", "sigma", "1"), ("psi", "sigma", "1"))`. These are constrained coordinates,
not a tensor factorization into four qubits. The Ising rules are
`sigma x sigma = 1 + psi`, `sigma x psi = sigma`, and `psi x psi = 1`.

Use the gauge and counter-clockwise exchange convention of
[Nayak et al., section II.A and equation (50)](https://arxiv.org/abs/0707.1889):

```text
F = [[1, 1], [1, -1]] / sqrt(2)
R = diag(exp(-i*pi/8), exp(3i*pi/8))
B0 = R, B1 = F^dagger R F, B2 = R
```

Clockwise exchanges use the adjoints. Appending `exchange(i)` followed by
`exchange(j)` acts as `Bj @ Bi` on a column state. Generators refer to adjacent
positions in the current ordering; every exchange swaps those positions in the
worldline ordering, regardless of direction. The source retains the complete
word. It does not identify braid words solely by their resulting density map:
two unitaries differing by a global phase have the same density action.

The model is an ideal topological operation sequence. It has no path geometry,
elapsed time, microscopic Hamiltonian, dissipative dynamics, control gradient,
hardware adapter or inferred calibration. A terminal pair-charge QND instrument
does not remove anyons or change their number. Destructive fusion, non-vacuum
total sectors, additional anyons, Fibonacci models, Majorana parity feedback,
GPU/distributed execution and universal-Ising gate-set claims are outside this
slice. Unsupported requests fail explicitly rather than selecting a circuit
backend or a Lindblad logarithm automatically.

## Frontend and Core source

Only `flagquantum.experimental.simulation` gains the feature export. The
Core-owned implementation lives beside the existing native frontend; it imports
no Runtime, Simulation, Compiler, provider or accelerator implementation.
The proposed signatures are:

```python
# Proposed surface only; unavailable on the inspected baseline.
class IsingBraidProgram:
    def __init__(self, *, initial_channel: Literal["1", "psi"] = "1") -> None: ...
    def exchange(self, left: int, *, clockwise: bool = False) -> "IsingBraidProgram": ...
    @property
    def basis_labels(self) -> tuple[tuple[str, ...], ...]: ...
    def projector(self, channel: Literal["1", "psi"]) -> torch.Tensor: ...
    def to_ir(self) -> ProgramRecord: ...
```

`exchange` appends one operation and returns the builder. `to_ir()` returns an
immutable snapshot of the existing internal ProgramRecord; subsequent builder
edits cannot alter an earlier plan. This does not publish a new public IR or
admit arbitrary ProgramRecord objects as root execution inputs. The projector
helper returns a fresh CPU complex128 mathematical tensor in canonical path
order, not a measurement executor. Its channel refers to the first pair in the
current positional ordering and the declared fusion tree, not an arbitrary pair
of the initial anyon IDs.

Add typed frozen Ising context and exchange data to the existing physical
profile unions. Context fields are `anyon_ids=("a","b","c","d")`,
`total_charge="1"`, `fusion_tree="left_associated"`, `gauge="nayak_ising"`, and
`initial_channel`. An exchange carries `left` and `clockwise`; its opcode is
`ising.exchange`. The graph has the same one-main-function, one-entry-block,
return-only structure as the existing bounded native profile. It has no SSA
qubit values, parameters, control-flow edges or timed-control fields. Existing
`physical.invalid_context`, `physical.invalid_operation` and
`physical.unsupported_structure` diagnostics remain the admission boundary.

The existing native-source codec emits `flagquantum.program_ir` version `1.1`
only for this profile. Its outer fields remain `schema`, `version`, `functions`
and `physical_context`. Context has exactly the fields above plus
`kind="ising_braid"`. Operations retain the existing operation envelope;
physical data has exactly `kind="exchange"`, `left` and `clockwise`.
All unused operand/result/parameter/condition lists are empty and literal is
null. Unknown fields, mixed profiles and unsupported versions are refused.
Rydberg source encoding remains exactly 1.0. Existing SSA kinds and the
unimplemented Quantum/Target level exits do not change.
Admit the new profile by evolving the existing internal multi-level candidate
from 0.4 to 0.5, retaining its previous approvals, diagnostics and authorities;
do not establish a second Program IR contract or verifier.

## Shared planning and result contract

Append IsingBraidProgram to the accepted program types of existing root and
Runtime planning/execution entries. Add no keyword, root export or plan class.
Only the typed frontend and its validated snapshot reach this route; generic
duck-typed `to_ir` objects do not receive native authority. Rydberg programs
continue through their approved Lindblad route and remain refused here.
Circuit conversion, compilation, OpenQASM, Deployment and Remote must refuse
this native profile before lowering it to an empty/encoded circuit or invoking
a service. Add refusal consumers to the Integration baseline and retain any
existing generic native-source refusal that already satisfies them. If an
owner needs a profile-admission change, deliver that change in its own domain
PR before enabling the affected path, not as an unreviewed Runtime dependency.
The existing Lindblad facades also refuse this discrete frontend or its source
with CapabilityError before Rydberg-specific field access: the Ising profile is
not a Hamiltonian input. Preserve existing malformed-source IR diagnostics.
Include these refusal consumers before activating the new frontend, rather
than letting a newly valid physical context reach Rydberg-only assumptions.

The native route accepts the following ExecutionOptions values. Existing
malformed-option validation remains unchanged. A valid unsupported choice raises
CapabilityError before plan sealing or numerical execution.
Reuse the current options resolver and its call, runtime-policy, program and
runtime-config precedence. Supply the fixed complex128 precision as a program
constraint; do not implement a parallel resolver. An explicit complex64
demotion retains the resolver's existing ValidationError rather than being
silently promoted or recategorized as an unsupported backend.

| Field | Native values |
| --- | --- |
| mode | unset, auto, density_matrix; resolved to density_matrix |
| backend | unset, auto, pytorch; resolved to pytorch |
| device | unset, auto, CPU; resolved to cpu |
| target | unset, auto, state, expectation; auto resolves to expectation when outputs are supplied, otherwise state |
| precision | unset or complex128; resolved to complex128 |
| batch_size | unset or 1; resolved to 1 |
| shots | unset; sampling is a separate explicit terminal-instrument call |
| seed | any currently valid seed or unset; retained in the plan but unused by deterministic unitary execution |
| require_gradients | unset or false; discrete braid words have no differentiable control input |
| allow_approximate, allow_backend_fallback | unset or false; there is no approximation or alternate route |
| memory_limit_bytes | existing positive-integer validation; fail if the planned state does not fit |

The state estimate is 64 bytes for one 2-by-2 complex128 density matrix, not a
hard process-memory bound. No noise_model, remote compiler/target/target_qubits,
direct shots, or alternate per-run setting is accepted for this native slice.
An expectation target without output requests raises ValidationError. An
explicit state target may still include expectation outputs; both are returned
with the exact axes below, with no sampling inferred from either target.
Executing a sealed plan retains existing refusal of per-run overrides and
environment-incompatible plans; no silent replan or fallback is allowed.

Outputs may be absent or use existing `fq.expectation(matrix, name=...)` requests
with finite CPU complex128 Hermitian 2-by-2 matrices in canonical fusion order.
Reuse current OutputRequest comparison/hash, matrix validation and naming rules
from proposal 080. Nonempty ordered request sequences are supported; named
outputs must be unique under the current rules. Trainable matrices, Pauli
descriptors, local qubit selectors and other output kinds are refused here.
The exception does not widen dense-output support for CircuitIR or Remote.
Snapshot source and output matrices at planning; later mutations to the builder
or the caller's matrices must not change a sealed plan or its execution.

Reuse ExecutionPlan with only its `analysis` annotation widened to
`CircuitAnalysis | None`: native plans have analysis=None, layers=(),
shardable_wires=(), world_size=1 and state_mode=recommended_mode=density_matrix.
Circuit plans retain their real CircuitAnalysis and all existing field values.
No native source is reported as zero qubits or as one encoded qubit. Adapt native
plan consumers explicitly; do not weaken circuit invariants to accommodate it.
The existing audit projection uses the sealed identity and braid-word length
for this route, not the legacy qubit-derived identity or a physical duration.
Existing circuit audit projections remain unchanged.

For this native route, `ExecutionPlan.summary()` retains its current keys:
`depth` and `n_instructions` are the braid-word length, while `n_wires`,
`two_qubit_gates` and `multi_qubit_gates` are null. Single-device distribution
semantics, false scalability/release claims and empty shardable wires remain
explicit. No new summary field is needed to recover native meaning: use the
sealed source. `calibrated_cost()` refuses this profile with CapabilityError
until a native cost model is independently approved; circuit calibration must
not interpret braid length as its measured gate-count features. The result
summary schema remains unchanged.

ExecutionResult is unchanged. Native `state` has shape `(1,2,2)` in canonical
fusion coordinates; each expectation measurement has shape `(1,)`, kind
expectation and empty qubits. Names use existing measurement metadata.
`value` has shape `(1,k)` for k requested expectations and is None without
outputs. `samples` is None. There is no time or braid-trajectory axis. The
result retains the exact executed plan and ordinary accuracy/runtime evidence.
Nondestructive charge sampling is explicit `sample_terminal(result.state[0],
instrument, shots=..., seed=...)` and returns existing MeasurementResult.

## Executable plan compatibility and failure behavior

Only native Ising plans emit `flagquantum.execution_plan` version `1.1`.
They reuse existing top-level fields and add one `outputs` ordered array.
`program` contains the physical-source 1.1 record above. Each output record has
exactly `kind="expectation"`, `name` (string or null) and `operator`, using the
existing Core tensor/scalar JSON encoding. Reuse that encoder/decoder and the
shared canonical JSON/hash helpers; add no second tensor codec.

Decision fields remain the existing mode/backend/device/target/batch_size/
precision/world_size/state_bytes/recommended_mode/require_gradients/
allow_approximate/allow_backend_fallback/memory_limit_bytes fields, with the
resolved values above. Requested and resolved options retain their existing
schema. The extensions array is empty. Existing environment/compiler identity
and exact-plan preflight rules apply. The program fingerprint covers the full
native source, the options fingerprint covers resolved options, and the complete
plan identity seals all fields including ordered outputs and their names and
matrix bytes. For native 1.1 only, define that identity as the existing canonical
hash of the complete payload with the identity field removed. The program
fingerprint is the canonical hash of its encoded source; option, environment
and compiler fingerprints retain their existing definitions. The existing 1.0
identity preimage and public EXECUTION_PLAN_VERSION constant remain unchanged.
This extends the existing helper by version, not a separate identity framework.
Validate encoded output shape and dtype before allocating through the existing
Core decoder. Equal density actions do not collapse different source identities.

The plan decoder recognizes 1.0 and 1.1 explicitly. A 1.1 plan requires exactly
the native source, fields and support decisions in this proposal; it cannot
carry a circuit or arbitrary physical profile. Old readers reject 1.1 instead
of interpreting it as CircuitIR. New readers retain byte-identical old 1.0
circuit plan round trips. Lindblad 1.3 through 1.7 and CircuitIR 1.0 remain
unchanged. No existing payload is migrated implicitly.

| Input or failure | Contract |
| --- | --- |
| Wrong Python type, including boolean exchange index | TypeError before append |
| Index outside 0,1,2 or unknown channel string | ValueError before append |
| Malformed internal source | Existing IRValidationError with physical diagnostic |
| Valid unsupported execution/output request | CapabilityError before sealing/execution |
| Corrupt, unknown or contradictory saved plan | Existing ExecutionPlanContractError categories; fail before numerical work |
| Non-finite/non-Hermitian output or a square matrix of the wrong dimension | ValidationError before sealing; reuse existing matrix checks, no silent casting or projection |

Existing OutputRequest factory errors still occur at construction: malformed
Python types raise TypeError, and a non-square/empty matrix or a local selector
on a matrix expectation raises ValueError. Unsupported dtype/device or a
trainable matrix raises CapabilityError on this new planning route. None of
these changes alter construction or execution errors on existing routes.

`clockwise` requires an actual bool. Frontend constructor/channel arguments
require strings and known channel literals. Source verifier and plan reader
validate these rules independently of the builder. Snapshot equality and hash
are structural; operation order, direction and initial channel matter. Existing
ProgramRecord hashes and source encodings remain unchanged for old profiles.

## Proposed user workflow

The following is the post-implementation target, **not runnable today**:

```python
import flagquantum as fq
from flagquantum.experimental.simulation import IsingBraidProgram
from flagquantum.runtime import sample_terminal

program = IsingBraidProgram().exchange(1)
effect = program.projector("psi")
plan = fq.plan(program, outputs=fq.expectation(effect, name="psi"))
result = fq.run(plan)
print(f"Psi probability: {result.expectation('psi')[0].item():.8f}")

restored = fq.ExecutionPlan.from_json(plan.to_json())
replayed = fq.run(restored)
instrument = {channel: (program.projector(channel),) for channel in ("1", "psi")}
assert replayed.state is not None
observed = sample_terminal(replayed.state[0], instrument, shots=4096, seed=7)
print("Fusion labels:", observed.metadata["outcome_labels"])
```

Expected probability is 0.50000000 and labels are `['1', 'psi']`. An inverse
exchange restores the original density. The empty braid word preserves the
chosen initial channel. Shot counts are environment-dependent and must not be
treated as cross-version exact constants.

The following current-API example is runnable now. It proves reuse only; the
caller still owns the fusion model and no native program or plan is created:

```python
import math
import torch
from flagquantum.operators import SuperOperator
from flagquantum.runtime import sample_terminal

dtype = torch.complex128
f = torch.tensor([[1, 1], [1, -1]], dtype=dtype) / math.sqrt(2)
r = torch.diag(torch.exp(torch.tensor([-1j, 3j], dtype=dtype) * math.pi / 8))
exchange = f.mH @ r @ f
initial = torch.diag(torch.tensor([1, 0], dtype=dtype))
state = SuperOperator.from_kraus([exchange]).apply(initial)
instrument = {
    "1": (torch.diag(torch.tensor([1, 0], dtype=dtype)),),
    "psi": (torch.diag(torch.tensor([0, 1], dtype=dtype)),),
}
result = sample_terminal(state, instrument, shots=4096, seed=7)
print("Probabilities:", [round(p, 8) for p in result.metadata["probabilities"]])
print("Fusion labels:", result.metadata["outcome_labels"])
```

Actual output on Python 3.12 / Torch 2.13.0:

```text
Probabilities: [0.5, 0.5]
Fusion labels: ['1', 'psi']
```

## Delivery and acceptance gates

1. **Integration baseline, after explicit contract approval.** Extend existing
   protected candidates rather than adding a parallel protection framework.
   Consumers use a replaceable fake implementing the reviewed frontend, plan,
   result and refusals. Capture old circuit, native-source and Lindblad payload
   bytes before implementation. Fixed independent oracles must not change when
   fake factories are replaced. A baseline does not establish native support.
2. **Core source.** Implement the frontend, immutable typed profile, verifier and
   versioned codec. Check fixed fusion rules, sector/tree/gauge, identity,
   direction, word order and worldline permutation. Prove old source/hash and
   public CircuitIR compatibility; keep new execution refused until admission.
3. **Simulation kernel.** Reuse existing SuperOperator actions for the finite
   ideal sequence. Compare complex F/R matrices, generators, both braid
   relations, distant commutation, inverse and full complex density against
   independent references at absolute tolerance 1e-12, rtol=0. Validate
   projective density semantics separately from unitary global phases. No
   alternative integrator or gate-backend fallback is admitted.
4. **Runtime planning before activation.** Implement native-only plan 1.1,
   immutable source/output snapshots, shared options, result axes, exact-plan
   identity and preflight. Prove direct run, planned run and JSON replay agree;
   distinguish malformed input, unsupported capability and saved-plan drift.
   Every native plan consumer must handle absent circuit analysis deliberately.
5. **End-to-end replacement and examples.** Bind unchanged baseline consumers to
   real implementations, sample both initial channels and nontrivial braid
   words, compare QND effects and unread measurement back-action. Check private
   RNG replay and binomial uncertainty separately from numerical error. Publish
   the runnable native example only when the complete path is real.

Run supported-Python focused tests, default tier, affected Runtime integration,
strict type/architecture/API checks and full applicable CI/GPU before each
authorized merge. Do not lower tolerances or replace independent references to
obtain green checks. Full M3 completion, hardware and all other computing
routes retain their separate gates.
