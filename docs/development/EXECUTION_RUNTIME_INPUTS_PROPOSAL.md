# Named runtime inputs for execution

Status: **explicitly approved by the user on 2026-11-12; bounded implementation
available locally as an experimental candidate**. This is not release admission.
The review fake remains executable evidence.

Owner: Integration for admission and the public boundary; Runtime for the
follow-up implementation. Path classification: `single_device_fast_path`.

The user requested a complete angle-binding slice after discussing the separation
of programs, runtime arguments, initial quantum states and execution options.
The user subsequently approved this concrete proposal. The approval covers the
additional keyword and plan extension described below; it does not authorize a
release or broader input support.

## Problem and intended behavior

FlagQuantum already has `fq.Parameter`, parameter expressions,
`Circuit.bind_parameters`, a versioned CircuitIR, reusable Module parameter slots,
an execution-option resolver and executable ExecutionPlan contracts. Users can
bind a new Circuit and run it today. That does not establish that a sealed plan
can accept a new angle without compilation, planning, stale values or loss of
the caller's PyTorch graph.

Add one named argument binding entry to `fq.run`, using existing symbolic
parameters as the declaration. Do not add a Parameter subclass, input registry,
public binding object, second IR, or input-shape declaration language.

## Approved public delta

Append this keyword-only parameter **after `initial_state`**, preserving all
existing parameters, order, defaults, exports and result fields:

```python
inputs: Mapping[str, float | Tensor] | None = None
```

This approved signature delta is implemented. The complete supported workflow is:

```python
import torch
import flagquantum as fq

angle = fq.Parameter("theta")
program = fq.Circuit(1, dtype=torch.complex128).ry(0, angle)
theta = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)

first = fq.run(program, inputs={"theta": theta})
first.statevector()[0, 1].abs().square().backward()
# theta.grad == sin(0.4) / 2, within float64 tolerance.

second = fq.run(first.plan, inputs={"theta": 0.8})
# second.statevector() == [[cos(0.4), sin(0.4)]], within float64 tolerance.
```

The shortest **existing, executable public workflow** remains:

```python
import torch
import flagquantum as fq

program = fq.Circuit(1, dtype=torch.complex128).ry(0, fq.Parameter("theta"))
theta = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
result = fq.run(program.bind_parameters({"theta": theta}))
result.statevector()[0, 1].abs().square().backward()
torch.testing.assert_close(theta.grad, torch.sin(theta.detach()) / 2)
```

The addition would remove explicit per-call Circuit binding from that workflow
and admit compatible value changes through one accepted plan. It does not claim
that the current binding workflow already avoids planning.

`fq.plan` has no signature or behavior change in this slice. An input
plan is obtained from the first successful `fq.run(..., inputs=...)`. Existing
plans without the new extension cannot accept `inputs`; run the symbolic source
with inputs to create an input-capable plan. Neither `Circuit.run` nor Module,
training, QEC, synthesis, remote, checkpoint or deployment APIs change.

## Meaning and first supported slice

`inputs` binds only declared runtime values. `initial_state` retains its separate
meaning as the entry quantum state. `options` and `outputs` retain their existing
responsibilities. In particular, synthesis's `inputs={"x": synth.UInt(2)}` is a
type declaration, not this runtime value mapping. Coherent quantum-register
preparation is not converted to classical substitution.

The first slice accepts ordinary static Circuit sources and compatible marked
ExecutionPlans, exact unbatched CPU PyTorch statevector execution, and full-state
or probability outputs. Runtime parameters are scalar real gate angles;
`ParameterExpression` uses the existing add/subtract/multiply/negate operations.
Constants and symbolic dependencies stay in the existing IR. Optimization is
disabled on this route until value-independent transforms are separately proven.
No new work qubits may be introduced. Default-zero execution remains the only
initial-state mode in this slice.

External initial states, explicit matrices, structured and dynamic operations,
noise, shots, accelerators, distributed execution, other engines, synthesis
Program sources, observables and remote execution are excluded. Explicitly
combining non-None `inputs` and `initial_state` must fail before planning. A later
change may compose the two contracts, including joint-state and workspace
preconditions; no implicit composition is admitted here.

Only finite Python `float`, exact dense strided CPU scalar `torch.Tensor`, or
exact `torch.nn.Parameter` values are accepted. Scalar tensors must be float32
for a complex64 plan and float64 for a complex128 plan, after the existing option
resolver has selected precision. Python floats are converted at that precision
and checked again for overflow. Booleans, integers, complex values, vectors,
tensor subclasses, wrong precision and implicit device transfer are rejected.
The narrow float rule is local to this new entry; existing parameter-binding APIs
retain their broader behavior.

The mapping must have string keys and exactly cover the source's symbolic names.
There are no remembered bindings or implicit per-parameter defaults. Parameter
objects with the same name refer to the same declared parameter. Static tensor
literals in the template, including expression leaves, are excluded to prevent
capturing previous-run storage or graphs in the reusable plan.

Binding must validate both supplied values and evaluated expressions before
numerical execution; finite inputs can produce non-finite expressions. Runtime
inputs cannot change wires, topology, operator arity, control flow, register
widths or resource constraints.

## Plans, gradients and ownership

Reuse outer ExecutionPlan schema/version 1.0 and its existing extension field:

```json
{
  "namespace": "flagquantum.execution",
  "kind": "runtime_inputs",
  "version": "1.0",
  "identity": "<existing canonical hash of payload>",
  "payload": {"parameters": ["theta"]}
}
```

Names are sorted and unique and must exactly match the symbolic IR. Shape,
precision, device and gradient permission are derived from the sealed decision;
do not duplicate them in another schema. Exactly one extension is admitted in
this slice. Readers that do not implement it reject it explicitly.

The plan stores the symbolic program and execution decision, not supplied angle
values, input tensors or autograd graphs. Compatible value changes preserve its
identity. A per-attempt binding materializes the numerical parameters without
mutating the source or the plan. Runtime must validate that live layer structure
still matches the sealed symbolic template before binding. Incompatible plans
fail rather than silently replan.

For direct `run(program, inputs=...)`, use the existing resolver policy to infer
gradient need from supplied tensors when not explicitly specified. Explicit
`require_gradients=False` conflicting with differentiable inputs is a validation
failure. A non-gradient replay plan refuses gradient-bearing inputs. A gradient
plan accepts both differentiable and ordinary values. The fake takes this
decision explicitly; public conformance tests verify production inference.

Reuse caller tensor objects for binding and preserve first-order native PyTorch
autograd. Do not detach, convert a tensor to a Python float, retain it in a plan,
or set `requires_grad` on behalf of the caller. Independent runs must not share
mutable binding state. Reusing an nn.Parameter after an optimizer update must
work without retaining a previous backward graph. Concurrent caller mutation
during a run is unsupported, as with ordinary PyTorch model inputs.

The plan's state estimate is not a bound on all autograd or process memory.
Preserve existing resource checks; make no additional peak-memory claim.

## Persistence and compatibility

JSON replay restores the symbolic program, requirements and decision; the caller
must supply all inputs again. It never restores a previous autograd graph or
silently substitutes the last bound value. Source/static-constant changes create
a new plan; a compatible runtime angle change does not.

Production JSON acceptance now validates this approved extension and restores
the symbolic requirements. The existing FQX workflow preserves the input
requirement with a result, and refuses a marked plan-only save before writing
because the program-only writer would discard the requirement. Loading never
executes. Public scenario tests verify these behaviors independently of the
review-only JSON helper. Old readers explicitly reject the new extension.

`inputs=None` must leave current paths and serialized artifacts unchanged.
Historical public contracts and API snapshots are preserved. The approved
implementation adds only a narrowly authorized signature delta and extension
admission, with conformance tests. No deprecation or compatibility adapter is
needed for this additive proposal. No release or capability promotion follows
from approval.

## Review evidence and implementation sequence

The [executable fake](../../tests/unit/test_execution_runtime_inputs_contract_fake.py)
uses the real planner once per preparation, the real IR/codec and statevector
engine, and the same numerical consumer with Core substitution or existing
Runtime parameter slots. It exercises analytic gradients, entanglement,
optimizer updates, multiple live forward graphs, compatible input changes,
review-only JSON replay, invalid inputs and tampered plans. Replay tests replace
compiler and planner entries with failing sentinels.

This proves that existing components can express the proposed slice. It does
not prove production `fq.run` support, backend replacement, performance gains,
FQX integration, environment admission or release readiness. Two binders share
the existing expression evaluator; the independent numerical references are
analytic state and gradient formulas.

The integration contract was recorded before implementation as
`67573f2067b1948e8e07eee1ff297e1c059b09a5`, in the local review repository. Runtime
then implemented `_runtime_inputs.py` behind the existing `run` path. It delegates
substitution and evaluated-angle checks to Core rather than adding an evaluator.
The numerical consumer remains unchanged; conformance tests replace the reviewed
binder with the real binder, verify public inference and persistence, and forbid
compiler/planner entry during accepted-plan replay.

The implementation adds no dependency, public type, registry or second IR. It
preflights the state memory minimum before planning, validates exact gate
parameter keys and scalar expressions, and rejects unsupported environments.
Broader representations and composition with external initial states remain
separate contract changes. Timings have not established a speedup; replay's
no-replanning behavior is verified by failing sentinels, not inferred from time.

## Verified review results

On 2026-11-12, the fake passed 26 scenarios. The focused selection including
existing parameter-binding, parameter-mapping and public external-initial-state
tests passed 88 tests. Black, Ruff, scoped strict mypy, public-API protection,
architecture and team-ownership checks passed. The existing public example above
was executed and its analytic gradient checked. SHA-256 verification confirmed
that all 26 files in the preceding initial-state delivery were unchanged.

The default smoke/unit selection was also attempted with `--maxfail=2`: 5,510
passed, 205 skipped and two failed in 218.31 seconds. Both failures were the QFT
Hadamard/phase-fusion apply-count cases already reproduced on the original source
revision during the initial-state delivery. The run stopped before reaching the
new unit-test file, which is covered by the separate focused selection. This is
not a full green tier. Integration was not rerun for this test/documentation-only
proposal; prior socket/Gloo environment limitations remain unresolved. No
production behavior or performance claim follows from these review results.

## Implemented candidate verification

The approved path is implemented locally. Public conformance has 42 scenarios;
combined public/fake, initial-state, plan, options, artifact, dynamic-default,
synthesis-admission and parameter-binding regression selection passed 267 tests.
After the final scalar-template and memory guards, a focused 157-test selection
passed again. Strict scoped mypy for the new runtime and public scenarios, general
typing for the edited API/codec/artifact/checker files, Black, Ruff, public-API,
architecture, dependency, capability and generated-document checks passed.

Default compatibility was compared against the preceding initial-state delivery
for 1/8-qubit Bell circuits in complex64/128: plan JSON and direct/replay output
bytes matched exactly. That preceding reader rejects the new extension with
`extension_incompatible`. This is bounded compatibility evidence, not a general
performance result.

Broader tiers were attempted with `--maxfail=2` and explicit 300-second budgets.
Default smoke/unit reported 5,510 passed, 205 skipped and the two previously
reproduced QFT fusion-count failures in 280.04 seconds. Integration reported 139
passed, 56 skipped and two setup errors from C10d localhost socket denial
(errno 1, Operation not permitted) in 158.73 seconds. Both subprocesses exited
normally within their budgets. The integration selection reached all new public
scenarios successfully; the default selection stopped before the new unit fake.
These are not green full tiers or release certification. The final focused
selection covers the guards added while the broader tiers were running.
