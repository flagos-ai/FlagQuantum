# API Change: Channel instruction parameters on the public `fq.Circuit` surface

## Status

Implemented for review on 2026-10-02 with explicit API-owner authorization. This
proposal adds declared parameter names to four existing channel opcodes and
extends `flagquantum.noise` with the reader that turns them into Kraus
operators. It adds no new opcode, no new root-level function, no new
`ExecutionOptions` field, and no change to `IR_VERSION`.

## Decision and authorization

The Stable Core surface changes in two places.

1. `OperatorSchema.parameters` for `bit_flip`, `phase_flip`, and `depolarizing`
   becomes `("probability",)`; for `amplitude_damping` it becomes `("gamma",)`.
   These opcodes were registered as channels with an empty parameter tuple, so
   the public gate method derived from the schema could not accept the value the
   channel is defined by.
2. `flagquantum/core/ir.py` requires those parameters on an `Instruction` that
   names a channel and does not carry a materialized Kraus tuple.

The owner of the affected surface is the integration maintainer, because the
change spans `flagquantum/core` (registry and IR), `flagquantum/circuit.py`
(public entry point), `flagquantum/noise` (factories and the new reader),
`flagquantum/runtime/planner` (route selection), and
`flagquantum/simulation/statevector` (the matrix field is no longer always one
tensor). `python tools/check_team_scope.py --team integration` passes for the
complete change set.

## Problem and affected user journey

Before this change, a user could not write a channel into a circuit:

```python
fq.Circuit(1).depolarizing(0, 0.1)
# TypeError: depolarizing accepts 1 wire(s) and 0 parameter(s)
```

The only supported routes were a `NoiseModel` rule matched to a gate name, or a
raw `Instruction` with an explicitly built Kraus tuple. The first cannot express
a channel that is not attached to a gate; the second is not reachable from
`fq.Circuit`'s public methods at all, because `Circuit.gate` takes `matrix=` and
the dynamic gate methods pass no matrix.

The machinery for the missing route already existed. `flagquantum.noise` had
four factories; `KrausChannel` already carried `(name, value)` pairs in
`parameters`; the serialized plan already recorded
`{"channel": {"kraus": ..., "name": ..., "parameters": {"probability": ...},
"schema": "flagquantum.kraus_channel.v1"}}`; and the density-matrix executor
already read `instruction.metadata["is_channel"]`. The compiler's own lowering
path already wrote `params=` for a channel. Only the four registry rows and the
gate-method assembly step were missing, so the capability was unreachable from
the user's side of the boundary.

A second defect followed from the first. Two entry points answered "is this
program noisy?" differently. `NoiseModel.add` wrote a channel into the IR; a
channel written directly into the circuit did not exist. When the first route
was removed from the picture, the planner's noise policy consulted only
`noise_model`, so:

- `mode="auto"`, `mode="density_matrix"`, and the default route executed the
  inline program on a representation that cannot hold a channel; and
- `mode="mps"` **returned the noiseless number** instead of failing.

The second is a silent wrong answer, which the repository forbids outright.

Repairing that exposed the same mistake a further level out. The question "does
this program need a noisy route?" was still answered from the `noise_model`
argument in three more places, each below the one before it:

- `plan()` built the noisy execution plan, and the MPS trajectory bond budget,
  only when a model was passed;
- `execute_plan` switched to the `noisy_mps` executor only when a model was
  passed; and
- `plan_from_dict` rebuilt the noisy execution plan only from a recorded
  extension, which an inline channel does not have.

Each is a case of reading the argument instead of the program. The third is the
sharpest: `plan_to_json` then `plan_from_dict` is the supported way to move a
plan between processes, and an inline-channel plan came back as a noiseless one.
The measurements under Evidence were taken before and after each repair.

## Evidence

Recorded against the worktree that carries this change, with
`PYTHONPATH=<workspace>/.venv/lib/python3.12/site-packages:$PWD` and the
workspace virtual environment.

Entry point, before and after:

```text
before  fq.Circuit(1).depolarizing(0, 0.1)
        TypeError: depolarizing accepts 1 wire(s) and 0 parameter(s)
after   fq.Circuit(1).depolarizing(0, 0.1).to_ir().instructions[0]
        Instruction(name='depolarizing', params={'probability': 0.1},
                    metadata={'is_channel': True}, matrix=<4 Kraus tensors>)
```

The six declarations of "differentiable channel and gate names" in the
repository were read with an AST scan before the change. They disagree today --
`OperatorSchema.differentiable` names 14 opcodes, `gradients.py:17` names 6,
`gradients.py:18` names 3, and three further modules repeat the 6-name or 3-name
list. This proposal does not touch that disagreement. It records that the
parameter-shift path is numerically correct for all 14 differentiable opcodes
against `torch.autograd`, so the defect is a duplicated constant and not a wrong
gradient, and it stays out of this change's scope.

Numeric agreement of the new entry point, against the analytic value of each
channel rather than against a recorded run:

```text
depolarizing(0.1) on |0>            P = [0.9333, 0.0667]   1 - 2p/3, 2p/3
h then amplitude_damping(0.2)       P = [0.6000, 0.4000]   (1+g)/2, (1-g)/2
bit_flip(0.25)                      P = [0.7500, 0.2500]   1 - p, p
phase_flip(0.25) on |+>             P = [0.5000, 0.5000]   unchanged populations
```

Route agreement, `torch.allclose(..., atol=1e-6)`:

```text
fq.Circuit(1).x(0).depolarizing(0, 0.1)
fq.Circuit(1).x(0) with NoiseModel().add("x", depolarizing_channel(0.1))
```

Planner behaviour after the change, for
`fq.Circuit(1).h(0).depolarizing(0, 0.1)`:

```text
default, auto, density_matrix   executes; decision mode = density_matrix
mps, tensor_network, statevector
    ValidationError: stable noisy execution supports mode='auto' or mode='density_matrix'
stabilizer                      CapabilityError: mode='stabilizer' samples
                                measurement outcomes; it cannot serve
                                measurement kind(s) probabilities
```

The stabilizer refusal is reported before the representation refusal, because
the stabilizer engine already owns a more specific statement of the same
obstacle and a caller asking for that mode should read the error about that
mode.

The one route that admits a channel without `mode="density_matrix"` is
`mode="auto"` with `allow_approximate=True` and a memory limit the density
matrix cannot meet. Measured on `fq.Circuit(6).h(0).cx(0,1)...cx(4,5)` with
`depolarizing(0, 1.0)` appended, `memory_limit_bytes=16384`, `shots=200`,
`seed=7` (the bond budget this limit resolves to is 2, the smallest that
represents the state):

```text
                                plan decision mode   noisy_execution_plan        counts
before  inline channel          mps                  None                        only 0**6 and 1**6
        NoiseModel route        mps                  mps / quantum_trajectory    4 outcomes
after   inline channel          mps                  mps / quantum_trajectory    4 outcomes
```

`depolarizing(p)` keeps `(1 - p) rho` and adds `p/3` of each Pauli, so at
`p = 1.0` no branch leaves the diagonal pair alone: seeing only `0**6` and
`1**6` is the noiseless run. The two rows above are the same program written two
ways, so they must agree, and after the change they do.

Plan portability, same program, `plan_to_json` then `plan_from_dict`:

```text
                                restored noisy_execution_plan   distinct outcomes at seed 4
before  inline channel          None                            2
after   inline channel          mps / quantum_trajectory        4
```

Agreement between the original plan and the restored one, over 4000 shots:

```text
inline channel    original 4 outcomes, restored 4 outcomes, identical counts
NoiseModel route  original 4 outcomes, restored 4 outcomes, identical counts
```

A model rule and an inline channel are compared by the program they produce, not
by where the channel appears in the source. `NoiseModel.add(name, channel)`
inserts the channel after **every** instruction with that name, so the two
spellings agree only when the same instruction is targeted once. Measured on the
same 6-qubit ladder at `p = 0.5`, `shots=1000`, `seed=4`, all four on
`mps / quantum_trajectory`:

```text
inline channel after x(0)          [..., x, depolarizing]        4 outcomes, P(0**6)=0.148
model rule on x                    [..., x, depolarizing]        4 outcomes, P(0**6)=0.148
inline channel after h(0)          [h, depolarizing, cx...]      2 outcomes, P(0**6)=0.000
model rule on h                    [h, depolarizing, cx...]      2 outcomes, P(0**6)=0.000
```

The first pair and the second pair are instruction-for-instruction identical, and
so are their counts. The two pairs differ because the programs differ:
`depolarizing` on the control of a GHZ before the entanglers is a dephasing
channel on `|+>`, and every branch of it then entangles to `|0**6>` or `|1**6>`,
so the diagonal keeps only two entries instead of four. A comparison that
matched an inline channel at one position against a model rule at another was
measuring the position, not the route.

Refusals of a malformed channel, all reproduced:

```text
depolarizing(0)                     ValidationError: depolarizing requires parameter(s): probability
depolarizing(0, rate=0.1)           ValidationError: depolarizing does not accept parameter(s): rate; accepted: probability
amplitude_damping(0, probability=0.1)
                                    ValidationError: amplitude_damping does not accept parameter(s): probability; accepted: gamma
depolarizing(0, 1.5)                ValidationError: probability must be between 0 and 1.
depolarizing(0, -0.1)               ValidationError: probability must be between 0 and 1.
depolarizing(0, "0.1")              TypeError: channel 'depolarizing' parameter 'probability' must be a real number, got str
depolarizing(0, True)               TypeError: channel 'depolarizing' parameter 'probability' must be a real number, not a bool
depolarizing(0, float("nan"))       IRValidationError: channel 'depolarizing' parameter 'probability' must be a finite real number, got nan
depolarizing(0, 0.1, 0.2)           TypeError: depolarizing accepts 1 qubit(s) and 1 parameter(s)
depolarizing(0, matrix=<kraus>)     ValidationError: Channel 'depolarizing' derives its Kraus operators from its declared parameters; pass parameters instead of a matrix.
channel_from_parameters("nope", {})
                                    ValidationError: unknown channel 'nope'; channels with a native factory: amplitude_damping, bit_flip, depolarizing, phase_flip
Instruction(name="depolarizing", wires=(0,))
                                    IRValidationError: opcode 'depolarizing' is missing parameter(s): probability
```

Two refusals are deliberately split between two owners. The registry rule
(`_normalize_angle`) asks only that a value be a finite real number, because a
rotation angle may be any real number; the `[0, 1]` limit belongs to the channel
factory, which is where the number's physical meaning lives. A string, a bool,
and a non-finite value are refused by the registry, so
`depolarizing(0, "0.1")` is refused by the same owner that refuses
`ry(0, "0.3")`; an out-of-range float is refused by the factory. Before that
split was enforced, a string reached `torch.as_tensor` and surfaced as
`TypeError: new(): invalid data type 'str'`, naming neither the channel nor the
parameter, and `True` was read as the probability one.

Channel-parity audit, from W0 item `N7-11`. Every claim below is reproducible
with the command in the same row.

| Claim | Result | Command |
| --- | --- | --- |
| All 4 channel opcodes are writable from `fq.Circuit` | pass | `pytest tests/unit/test_channel_instruction_parameters.py -q` |
| The declared name equals the factory keyword | pass | same |
| The value survives the IR JSON round trip | pass | same |
| The compiler lowering writes the same instruction as the inline form | pass | same |
| An inline channel is planned as a noisy program | pass | same |
| An amplitude representation refuses a channel | pass | same |
| The population matches the analytic channel | pass | same |
| The `NoiseModel` route and the inline route agree | pass | same |
| An inline channel reaches the `auto` + `allow_approximate` MPS route | pass | same |
| An inline channel survives `plan_to_json` / `plan_from_dict` | pass | same |
| A materialized channel without declared parameters still loads | pass | same |
| The operator manifest is current | pass | `python tools/operator_manifest.py --check` |
| All twelve fail-closed repository checkers pass | pass | `python tools/check_*.py` |

## Decision Candidates

**A. Declare the parameter in the operator registry and require it on an
unmaterialized channel instruction. (Recommended, implemented.)**

The registry is already the single statement of what an opcode takes; a channel
factory and a schema that disagree are then refused at the boundary rather than
at execution. The serialized plan already carried the value, so nothing new
crosses the artifact boundary. `IR_VERSION` stays `1.0` because no field is added
or reinterpreted: a channel instruction written by the compiler already had
`params`, and one written before this change already had the Kraus operators.

Cost: `Instruction.matrix` is a tuple of tensors for a channel and a single
tensor for a gate. Every consumer of that field must read it as a value rather
than assume one tensor. Six modules read it; two of them
(`simulation/statevector/fixed_layer_cpu.py`,
`simulation/statevector/clifford_matching.py`) tested
`instruction.matrix.requires_grad` and raised
`AttributeError: 'tuple' object has no attribute 'requires_grad'` on a channel
circuit. Both now share one helper. The remaining four already iterate a
sequence or never see a channel.

**B. Add a `Circuit.channel(...)` method that takes a `KrausChannel`. Rejected.**

It puts a second vocabulary beside the gate methods for no gain: the opcode name
already exists, the factory already exists, and `fq.Circuit(1).depolarizing(0,
0.1)` is the spelling a PennyLane or Qiskit user already knows. It would also
make the channel the only instruction whose parameters are not declared in the
registry, which is the defect being repaired.

**C. Keep the schemas empty and accept any keyword on a channel method.
Rejected.** `amplitude_damping(0, probability=0.1)` would then fail inside
`torch.as_tensor` or, worse, be silently accepted as the damping rate.

**D. Leave the planner keyed on `noise_model`. Rejected.** `mode="mps"` on an
inline channel returns the noiseless number. A silent wrong answer is not an
acceptable default, and the repository's non-negotiable rules forbid it.

## Prohibited Practices

1. Do not make a channel instruction's `matrix` field a single tensor by
   broadcasting the Kraus tuple. The tuple length is the channel's branch count.
2. Do not silently ignore a channel instruction on a representation that cannot
   hold it. Refuse it, as `mode="mps"` and `mode="statevector"` now do.
3. Do not accept a channel keyword the schema does not declare, and do not
   accept a declared one under a different name.
4. Do not move the `[0, 1]` range check into `_normalize_angle`. A rotation angle
   is any real number, and a shared rule that refuses `ry(0, 3.5)` is wrong.
5. Do not require declared parameters on a channel instruction that already
   carries its Kraus operators. `IR_VERSION` is an exact-match pin, so such a
   payload cannot be rewritten and must stay loadable.
6. Do not let a channel and an explicit `matrix=` define the same instruction.
   Refuse the conflict.
7. Do not reorder the stabilizer refusal behind the generic representation
   refusal, or a caller asking for `mode="stabilizer"` reads an error about a
   different mode.
8. Do not decide whether a program is noisy from the `noise_model` argument.
   The program carries the channels, and a rule keyed on the argument admits
   the same program on a representation that cannot hold it. The predicate is
   `flagquantum.runtime.planner.carries_noise_channels` over the program.
9. Do not rebuild a saved plan's noisy route from the recorded extension list
   alone. An inline channel is in the program, so the restore path reads the
   program too.

## Compatibility

This is an additive success path plus a fail-closed correction.

- **Adding.** A channel written in a circuit now executes. Nothing that worked
  before stops working: the `NoiseModel` route is unchanged and produces the
  identical instruction.
- **Changing.** An `Instruction` that names a channel and carries neither
  parameters nor Kraus operators is now refused. No producer in the repository
  built one: the compiler lowering, the QEC sampling path, and the testing
  helper all write one or the other, and each was updated to write `params`
  where it previously wrote only `matrix`.
- **Correcting.** A program that carries a channel and selects `mode="mps"`,
  `mode="tensor_network"`, or `mode="statevector"` now fails instead of
  returning a noiseless result. The refusal names the two supported modes.
- **Correcting.** The `mode="auto"` + `allow_approximate=True` route now runs
  the trajectory representation for an inline channel, as it already did for the
  equivalent `NoiseModel` route. The two wrote the same program and returned
  different numbers.
- **Correcting.** `plan_from_dict` now restores the noisy execution plan of an
  inline-channel plan. A plan saved to JSON and read back used to execute the
  noiseless program, and executing a plan is the supported way to run a workload
  in another process.
- **Not changing.** `IR_VERSION` stays `1.0`; the plan schema is unchanged; no
  root export is added or removed; no `ExecutionOptions` field is added.
  `flagquantum.noise` gains two exports, `CHANNEL_FACTORIES` and
  `channel_from_parameters`, and `flagquantum.noise.carries_noise_channels` is
  added to the planner's public surface.

The four channel opcodes remain in `[unsupported] flagquantum_opcodes` of
`contracts/pennylane-interop-contract.toml`. That entry states what the
PennyLane adapter can lower, not what FlagQuantum can execute, and the adapter
still converts gates only. `tools/check_pennylane_interop_contract.py` passes
unchanged.

## Acceptance Tests

Focused, all in `tests/unit/test_channel_instruction_parameters.py` (37 tests):

- the declared name on each of the four channel schemas equals the factory
  keyword;
- the value is accepted positionally and by name and reaches the same
  instruction;
- a value under another channel's name, a missing value, and an undeclared
  keyword are refused with the name and the accepted set;
- a probability outside `[0, 1]`, a string, a bool, and a non-finite value are
  refused by the owner that should refuse them;
- a `matrix=` for a channel is refused as a conflict;
- a second positional value is refused with the arity;
- the instruction carries the value and the `is_channel` marker;
- the value survives the IR JSON round trip byte for byte;
- the compiler lowering writes the instruction the inline form writes;
- an inline channel is planned as `density_matrix` with `channel_count == 1`;
- `mps`, `tensor_network`, and `statevector` refuse it;
- an inline channel reaches the `auto` + `allow_approximate` MPS route, and the
  counts are not the noiseless pair;
- an inline channel survives `plan_to_json` and `plan_from_dict` with its noisy
  execution plan and its identity intact;
- the population equals the analytic value of each channel;
- the `NoiseModel` route and the inline route agree within `1e-6`;
- a materialized channel without declared parameters still loads;
- a channel with neither parameters nor operators is refused.

The two route tests were confirmed to fail against the unfixed planner and
restore path and to pass against the fixed one, so they are not restatements of
the implementation.

Regression, updated rather than weakened. Five pre-existing tests built a
channel instruction through the internal constructor with a `matrix=` and no
`params`; they are legal and were left as they are. Five sites that built one
with neither were given the parameters they describe:
`tests/unit/test_stabilizer_execution_mode.py:108`,
`tests/team/simulation/test_stabilizer_engine.py:333`,
`tests/unit/test_statevector_forward.py:573`,
`tests/unit/test_noisy_statevector_numerics.py:24`, and
`tests/hybrid_compiler/test_target_legalization.py:170`.

Broad tier:

```text
python -m pytest -m "smoke or unit" -q
2 failed, 5460 passed, 222 skipped, 2128 deselected, 10 warnings in 230.17s
```

Both failures pre-date this change and were confirmed by a stash comparison:
`tests/unit/test_evidence_revisions.py::test_checked_in_evidence_accounts_for_every_recorded_revision`
and `tests/unit/test_native_cpu_adjoint.py::test_compact_cx_runtime_threshold_and_rollback`.

Repository gates:

```text
python tools/operator_manifest.py --check                    # pass, after regeneration
python tools/docs_source_of_truth.py                         # pass
python tools/public_api_snapshot.py                          # public API migration baseline passed
python tools/check_docs_links.py                             # 497 markdown files, all links resolve
python tools/check_repository_language.py                    # pass
python tools/check_repository_hygiene.py                     # pass
python tools/check_capability_maturity.py                    # pass
python tools/check_team_scope.py --validate                  # pass
python tools/check_pennylane_interop_contract.py             # pass, unchanged
python tools/check_cirq_interop_contract.py                  # pass
python tools/check_qiskit_interop_contract.py                # pass
python tools/check_braket_interop_contract.py                # pass
python tools/check_cudaq_export_contract.py                  # pass
python tools/check_interop_capability_gap_matrix.py          # pass
python tools/check_legacy_root_api_usage.py                  # pass
```

## Open Questions

1. **Trajectory representations — resolved for `auto`, open for an explicit
   request.** `mode="mps"` with a channel still reads the specific refusal
   `stable noisy execution supports mode='auto' or mode='density_matrix'`,
   because naming a representation whose executor cannot consume a channel is a
   request the caller can make differently. The `auto` route with
   `allow_approximate=True` now builds the MPS trajectory representation for an
   inline channel exactly as it did for a `NoiseModel`, and a saved plan
   restores it. Whether an explicit `mode="mps"` should be admitted once the
   trajectory representation is reachable for inline channels is a separate
   question about the mode vocabulary, and it is left open.
2. **Sampling-axis breadth.** The inline form covers the four opcodes with a
   native factory. `flagquantum.noise` exposes sixteen callables, including
   thermal relaxation, readout error, and coherent overrotation. Which of those
   should become opcodes rather than `NoiseModel` rules is a separate proposal;
   a channel with parameters that are not scalars (a relaxation pair, a rate
   matrix) needs its own decision about what a serialized parameter is.
3. **Parameter expressions.** A channel parameter accepts a bound real scalar
   through the registry's rule. A trainable channel probability would need
   `differentiable = True` on the channel schema and a gradient path through the
   Kraus operators, which does not exist. Noisy gradients remain unsupported.
4. **The duplicated differentiable-name constants.** Six declarations of the
   differentiable gate set disagree, as recorded under Evidence. Consolidating
   them is a change to `flagquantum/gradients.py` and three simulation modules
   and is deliberately not folded into this proposal.
5. **`mode="stabilizer"` with a readable price.** The stabilizer engine refuses
   a channel with its own message. Whether it should be able to execute a Pauli
   channel natively, which is the one channel class a stabilizer representation
   can represent exactly, is a capability question for the simulation team.

## Owner and approvals

- **Owner:** integration maintainer, 2026-10-02.
- **Affected surfaces:** `flagquantum/core/operator_schema.py`,
  `flagquantum/core/ir.py`, `flagquantum/circuit.py`,
  `flagquantum/noise/{channels,__init__}.py`,
  `flagquantum/compiler/noise.py`, `flagquantum/qec/sampling.py`,
  `flagquantum/testing/correctness.py`,
  `flagquantum/runtime/planner/{__init__,noise_selection}.py`,
  `flagquantum/runtime/{plan_execution,execution_plan_contract}.py`,
  `flagquantum/simulation/statevector/{fixed_layer_cpu,clifford_matching}.py`,
  `docs/operator_manifest.json`, `capability-maturity.toml`,
  `docs/guides/NOISY_SIMULATION.md`.
- **Team scope:** `python tools/check_team_scope.py --team integration --files`
  over the complete change set reports `team ownership policy passed`.
- **Approval required from:** the API owner for the registry and IR change; the
  integration maintainer for the planner and simulation change. Both are the
  same review on this change.
