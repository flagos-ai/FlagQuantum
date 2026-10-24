# Proposal 089: unread Ising pair-charge observation

Status: **draft for review; neither contract nor implementation approved**.
The [M3 acceptance review PR](https://github.com/flagos-ai/FlagQuantum/pull/791)
identified the native intermediate-observation gap. Its document approval or
merge does not approve this change. No protected API is implemented here.

## Five decisions requiring explicit approval

1. Add `IsingBraidProgram.dephase_pair(self) -> IsingBraidProgram` in its existing
   experimental namespace. This appends one **unread**, ideal QND observation of
   the charge of positions 0/1 in the current fusion tree. No outcome is selected
   or returned. Zero or one such operation is permitted in the initial profile.
2. Represent it in the existing ProgramRecord as `ising.dephase_pair`, with
   empty operands/results/parameters/conditions and `physical_data=None`.
   Introduce no IR, generic instrument record, controller or registry. Core owns
   source validation and encoding; Simulation applies the fixed existing Kraus
   channel; Runtime retains its existing planning/execution/result authority.
3. Only sources containing this operation use native source 1.2 and native
   execution plan 1.2. Existing braid-only source/plan 1.1 and all circuit,
   Rydberg and Lindblad artifacts retain their exact bytes and meanings.
4. Preserve root `fq.plan/run`, ExecutionOptions, ExecutionPlan, ExecutionResult
   and MeasurementResult signatures and fields. The result state stays
   `(1,2,2)` CPU complex128; it may now be mixed. No mid-program sample, branch,
   classical bit, event timestamp or feedback trace is fabricated. Existing
   terminal sampling remains a separate explicit call after execution.
5. After approval, first establish unchanged independent consumers with a
   replaceable baseline and old-byte captures; then deliver Core, Simulation
   and Runtime-owned changes. Each PR needs its own merge authorization and
   complete applicable CI/GPU, rather than a blanket implementation/merge grant.

## Why this bounded first slice

[Proposal 085](API_CHANGE_PROPOSAL_085_ISING_BRAID_PROGRAMS.md) admits only ideal
braids. [Proposal 083](API_CHANGE_PROPOSAL_083_TERMINAL_INSTRUMENT_SAMPLING.md)
returns terminal outcome indices and explicitly does not implement intermediate
state updates. A caller can currently demonstrate the channel mathematically,
but cannot retain its placement between braids in a native source or saved plan.

Qubit dynamic measurement/feedback already exists in
[dynamic execution](../../flagquantum/runtime/dynamic/execution.py), with
[classical/feedback result projection](../../flagquantum/runtime/dynamic/result.py)
and the approved [unified execution contract](API_CHANGE_PROPOSAL_084_DYNAMIC_EXECUTION_UNIFIED_RUN.md).
That executor assumes tensor-product qubit pure states and classical bits. A
fusion density is not a qubit statevector or a synthetic bit-register event.
This proposal reuses ProgramRecord and canonical root result authorities while
leaving that separate dynamic execution and Remote workstream unchanged. It
must not add a second dynamic runner. A future sampled native instrument must
review its join to those existing authorities before choosing a new surface.

The first choice here is the unread exact channel, **not** the more extensive
sampled-branch API. That choice keeps a complete native source/plan/channel/
continuation path without guessing the later classical-record contract.
Sampled branches and feedback remain unfinished roadmap work, not completion
claims for this slice. If that ordering is unsuitable, reject or amend these
five decisions before implementation.

## Exact physical and source semantics

Keep the four sigma anyons, total vacuum, left-associated fusion tree, fixed
Nayak gauge and two ordered fusion coordinates from proposal 085. Positions 0/1
mean the **current first pair**, consistent with `program.projector(channel)`,
not the initially named anyon IDs after worldline exchanges.

At the operation boundary, apply

```text
P1 = diag(1,0), Ppsi = diag(0,1)
J1(rho) = P1 rho P1, Jpsi(rho) = Ppsi rho Ppsi
D(rho) = J1(rho) + Jpsi(rho)
```

Simulation reuses `SuperOperator.from_kraus([P1,Ppsi]).apply(rho)` and the existing
braid actions, in source order. No Hamiltonian, integrator, time coordinate or
hidden measurement draw is introduced. Runtime does not implement that algebra.
The channel preserves trace and charge populations and removes off-diagonal
fusion coherence. It does not condition on a result or renormalize a branch.
A zero-weight branch contributes the zero matrix without division.

No atom/trap identity is introduced into the Ising context; existing anyon IDs
and exchange permutations are preserved. Extending observation to native atom
presence, nonideal efficiency or arbitrary Kraus branches requires another
concrete contract, including the physical-loss/detected-loss distinction.

## Proposed user code (unavailable before implementation)

```python
import flagquantum as fq
from flagquantum.experimental.simulation import IsingBraidProgram

program = IsingBraidProgram().exchange(1).dephase_pair().exchange(1, clockwise=True)
output = fq.expectation(program.projector("psi"), name="psi")
plan = fq.plan(program, outputs=output)
result = fq.run(fq.ExecutionPlan.from_json(plan.to_json()))
# Proposed: result.state[0] == I/2; result.expectation("psi")[0] == 0.5.
# Omitting dephase_pair restores the initial channel and gives P(psi)=0.
```

The word has two exchanges and one unread channel. Any existing source-depth
summary must count the channel as an operation, without claiming a qubit gate
count or physical duration. Circuit analysis stays absent. The method has no
callback, Kraus argument, outcome argument, options, implicit seed or timing
parameter. Supplying arguments raises TypeError. Calling it a second time raises
ValueError before mutating the builder; no observation is silently dropped.

## Runnable current-API reference

This mathematical composition runs **today**; it is not native intermediate
observation support, a contract baseline or the proposed API's implementation:

```python
import math
import torch
import flagquantum as fq
from flagquantum.experimental.simulation import IsingBraidProgram
from flagquantum.operators import SuperOperator

program = IsingBraidProgram().exchange(1)
rho = fq.run(program).state[0]
projectors = [program.projector(c) for c in ("1", "psi")]
unread = SuperOperator.from_kraus(projectors).apply(rho)
fusion = torch.tensor([[1,1],[1,-1]], dtype=torch.complex128) / math.sqrt(2)
phases = torch.exp(torch.tensor([-1j,3j], dtype=torch.complex128) * math.pi/8)
braid = fusion.mH @ torch.diag(phases) @ fusion
continuation = SuperOperator.left_right_multiply(braid.mH, braid)
with_observation = continuation.apply(unread)
without_observation = continuation.apply(rho)
print(round(float(with_observation[1,1].real), 12))
print(round(float(abs(without_observation[1,1].real)), 12))
print(torch.allclose(with_observation, torch.eye(2, dtype=rho.dtype)/2,
                     atol=1e-12, rtol=0))
```

Expected executed output: `0.5`, `0.0`, `True`. The fixed F/R matrices above are
an independent reference, not a new production kernel. The proposed native
program must reproduce this complete observation-plus-continuation effect.

## Encoding, equality and compatibility

The new operation's source record uses all the same Operation fields and a
literal JSON null `physical_data`; the opcode fully specifies the fixed channel.
A new physical-data type would carry no additional physical input and is not
justified. The native codec must special-case this declared null, never broadly
accept null for existing exchange or Rydberg controls.

Only a source containing exactly one channel may declare source 1.2, and its
execution plan must declare native 1.2. A 1.1 source with the new opcode, a 1.2
source without it, extra fields, conditions, operands or duplicate observations
must fail strict decoding/verification. A rehashed payload cannot evade semantic
validation. Saved-plan inconsistency remains a SerializationError under the
existing reader contract; malformed builder arguments remain TypeError or
ValueError as stated above. No API snapshots are regenerated to mask refusals.

The existing full-source and output snapshots, plan identity and environment/
compiler checks apply. Reordering observation and exchanges changes the source
and plan identity even if a particular input has equal output. Old 1.1 artifacts
must not be relabeled, migrated or regenerated. Existing old-version readers
refuse 1.2. The shared root circuit and dynamic-program paths gain no admission
for native source dictionaries, dense outputs or arbitrary physical records.

## Execution limits and explicit non-admission

Keep current native Ising options/refusals: CPU complex128 density, one declared
initial channel, no gradients, no noise model, no shots request on `fq.run`, no
Remote/Deployment, no cost calibration masquerading as circuit analysis, no
fallback and no quantum-memory scaling claim. Existing matrix-output validation
and allocation preflight continue unchanged. Terminal sampling may be called
explicitly on the final state using its existing private-seed replay contract;
those samples are final observations, not a record of the intermediate channel.

No mid-program branch probability or selected outcome is returned. A future
sampled choice would require explicit `p_j=trace(J_j(rho))`, a positive
normalization denominator, private RNG ownership, recorded labels and a precise
result projection; zero-probability forced branches must be refused. Those are
future required decisions, not fields to add speculatively in this slice.

## Acceptance and delivery gates

1. Before code changes, capture old source/plan bytes, identities and result
   axes. Establish replaceable consumers at 1e-12 absolute error, rtol=0, with
   fixed independent F/R and Kraus references. Do not change the oracles when
   replacing factories with the real implementation.
2. For both initial channels, compare full complex density for identity,
   prefix/channel/inverse, noncommuting prefix/suffix words, eigenstate
   zero-weight branches and terminal-output probabilities. Verify unread
   continuation equals the weighted sum of conditional continuations, retains
   trace, removes coherence and consumes no RNG.
3. Prove source order, fresh snapshots, worldline/position meaning, strict
   rejection of second observations/extra data, source/plan version binding,
   and exact numeric direct/planned/JSON replay. Preserve all old payload bytes
   and existing dynamic, circuit and Lindblad consumers.
4. Deliver Core source/codec changes, Simulation channel composition and Runtime
   admission separately after the approved shared baseline. No new controller,
   statevector fallback, solver, event registry or result schema is allowed.
5. Run supported-Python focused tests, default tier, affected integration,
   strict type/architecture/API checks and complete latest CI/GPU before each
   individually authorized merge. Proposal approval does not authorize merges.

This draft introduces no implementation and asserts no contract approval.
