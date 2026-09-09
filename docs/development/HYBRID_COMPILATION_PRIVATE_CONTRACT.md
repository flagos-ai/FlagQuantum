# Private hybrid compilation contract

Status: Phases 1-5 implemented and verified under the repository owner's
2026-09-09 direction to record and execute the Python-first hybrid compilation
plan.

## Purpose

Authorize the smallest Compiler-owned semantic core needed to represent the
first hybrid quantum-classical vertical slice without changing a stable public
API or creating a second execution contract.

This contract is deliberately narrower than the long-horizon multi-level IR
architecture. It authorizes only the semantic objects that are exercised by the
first nested-loop, data-dependent-gate, scalar-expectation scenario.

## Existing contracts reused

- Core-owned `CircuitIR` remains the quantum-region handoff.
- Existing compiler entry points continue to consume and return `CircuitIR`.
- Existing Runtime entry points continue to consume `CircuitIR` or their
  existing execution-plan type.
- Existing local statevector execution remains the first numerical target.
- Existing tensor and result types remain authoritative at the execution edge.
- Existing extension discovery and lifecycle remain the only plugin authority.

No new public artifact, execution result, backend, plugin registry, or root
export is authorized.

## Newly authorized private responsibility

`flagquantum.compiler._hybrid` may own a private structured program
representation containing:

- typed SSA-like values;
- immutable operations, blocks, and regions;
- function arguments and one return;
- scalar/tensor constants, indexing, comparison, addition, and remainder;
- structured `if` and `for` regions with explicit yields;
- ordered quantum effects;
- parameterized RX, RY, CX, embedding, and scalar expectation operations;
- deterministic semantic identity and source-located diagnostics;
- a fail-closed verifier.

These types are compiler implementation details. They must not be re-exported
from `flagquantum`, added to the Stable Core manifest, serialized as a public
schema, or accepted by `fq.run`/`fq.plan` during Phase 1.

## Dependency direction

The Phase 1 package may depend on Python standard-library modules and stable
Core semantic types needed for later `CircuitIR` lowering. It must not import
Runtime, Simulation, Compute, Remote, Ecosystem, provider SDKs, or optional
frameworks.

PyTorch is not required by the Phase 1 semantic core. Framework tensor values
enter only in later capture and execution phases behind separately tested
adapters.

## Required invariants

The Phase 1 implementation must prove:

1. immutable deterministic representation;
2. definitions dominate uses within the accepted single-function profile;
3. values have one definition and type-consistent uses;
4. branch result arity and types match;
5. loop-carried value arity and types match initialization and yield;
6. ordered quantum effects cannot be duplicated, reused, or dropped;
7. unsupported operation names and attributes fail closed;
8. source locations do not affect semantic identity;
9. no current stable import, signature, default, schema, or behavior changes;
10. the package remains unused by the default compiler and runtime until the
    later vertical-slice gates authorize integration.

## First-profile exclusions

Phase 1 does not authorize:

- Python AST capture;
- `while`, recursion, exceptions, mutation, or general Python semantics;
- measurement-produced control flow;
- stateful simulator sessions;
- lowering or execution;
- differentiation implementation;
- public API or default-path integration;
- MLIR, LLVM, QIR, timing, pulse, provider, or deployment behavior;
- Catalyst or FlagQuantum-authored C++.

## Entry gates

Implementation begins only when:

- the authoritative Integration baseline is identified;
- migration targets current vNext authority rather than restoring a historical
  worktree, package, or layer stack;
- team-scope preflight passes for all proposed implementation and test files.

These gates are satisfied. Phase 1 implementation proceeds in the private
Compiler namespace on the Integration baseline. The historical `_compiler`
tree remains reference evidence only.

## Phase 1 acceptance

- focused construction and negative-verifier tests pass;
- semantic identity is deterministic under repeated construction;
- source relocation does not change identity;
- all declared invalid value/effect/control-flow cases are rejected;
- compiler package boundaries pass the architecture checker;
- default smoke/unit tests pass;
- protected public API comparison reports no hybrid additions;
- the Compiler README contains a ten-minute internal contributor path.

Phase 1 is complete and authorizes Phase 2 capture design review; it does not
automatically authorize capture, execution, gradients, or public claims.

## Phase 2 restricted capture authorization

Phase 2 may add private `capture_source` and `capture_function` entry points
under `flagquantum.compiler._hybrid`. Capture parses Python source with `ast`;
it must not execute the function or evaluate runtime tensor predicates.

The accepted subset is limited to local-name assignment, structured
`if`/`elif`/`else`, `for` over `range`, iteration over a statically ranked tensor
axis, `enumerate` over such an axis, tensor dimension/extraction, scalar
addition/remainder/comparison, and the first-profile quantum operations. A
single scalar expectation return is required.

Local assignments are straight-line temporaries in their current region.
Writing an outer classical binding inside a branch or loop is rejected until a
later contract adds explicit classical region-carried results; such writes must
never be approximated by retaining the pre-region value.

Mutation, `while`, `break`, `continue`, Python `yield`, exceptions, context
managers, comprehensions, generators, imports, global/nonlocal state, arbitrary
calls, host scalar extraction, detach, NumPy conversion, recursion, and
unsupported quantum operations fail closed with source-located diagnostics.

Phase 2 does not authorize specialization, `CircuitIR` lowering, execution,
autodiff, framework graph capture, or default-path/public integration.

Phase 2 acceptance is complete. The captured golden source preserves nested
tensor-axis loops and data-dependent quantum gate selection, and the prohibited
construct suite fails with source-located diagnostics. Phase 3 specialization
and `CircuitIR` lowering remain separately gated.

## Phase 3 specialization and lowering authorization

Phase 3 may interpret a verified `HybridProgram` against one runtime input set
to select structured branches and unroll bounded loops into an ephemeral quantum
trace. The trace is not a restored `QuantumIR` and is not serialized or exposed;
it exists only to lower the selected path into the existing Core-owned
`CircuitIR`.

Parameterized gates lower first to a `CircuitIR` template containing stable
Core `Parameter` slots. Actual scalar tensor views are retained separately by
reference and may be bound into a circuit without `.item()`, detachment, NumPy
conversion, or scalar copying. Angle embedding lowers to one RX operation per
declared wire. Program structure, input signature, selected quantum structure,
and parameter values remain distinct identity dimensions; parameter values
must not affect reusable structure identity.

A bounded LRU structure cache is permitted only when hit, miss, and eviction
are observable in the returned private result. Unsupported input signatures,
non-scalar predicates, invalid loop bounds, invalid wires, and unsupported
dynamic effects fail before Simulation.

Phase 3 does not authorize Runtime integration, numerical execution, VJP,
public/default-path integration, or performance claims.

Phase 3 acceptance is complete: positive, negative, zero, and mixed gate
selection match hand-built circuit structures; late-bound tensor views preserve
their autograd edges; template identity excludes values; bounded cache events
are observable; and the existing Compiler accepts both templates and bound
circuits. That compiler-owned handoff is the input to the separately reviewed
Phase 4 contract below.

## Phase 4 local CPU forward authorization

Phase 4 may complete the forward-only vertical slice by encoding the captured
single expectation in the existing Core `MeasurementNode` representation on the
same `CircuitIR` that already carries the lowered instructions. Each Pauli term
uses the same `fq_output_index`, so the existing `ExecutionResult.expectation()`
accessor returns their sum as one scalar per batch item. The existing
`ObservableNode` terms remain available as semantic circuit annotations; they
do not constitute a second execution request or result type.

The handoff is exactly:

```text
Compiler-owned specialization and lowering
  -> Core-owned CircuitIR with MeasurementNode requests
  -> existing Runtime plan and one-attempt execution
  -> existing Simulation local CPU statevector
  -> existing ExecutionResult.expectation()
```

Runtime must not import `flagquantum.compiler._hybrid`. It accepts only the
Core-owned artifact and remains responsible for planning, execution lifecycle,
measurement dispatch, and result assembly. Simulation remains responsible for
statevector evolution and Pauli expectation numerics. No new executor,
execution result, public export, serialized public schema, or backend-selection
rule is authorized.

Phase 4 is restricted to analytic, forward-only, local CPU statevector
execution with `single_device_fast_path` semantics. It does not claim a VJP,
gradient preservation through the runtime, finite-shot behavior, noise,
accelerator support, distributed execution, performance improvement, or
production support. The existing public/default `fq.run` and `fq.plan` paths
must not import or invoke hybrid capture or specialization when hybrid
compilation is unused.

Phase 4 acceptance requires all golden positive, negative, zero, and mixed
branch patterns to match an explicit hand-built circuit reference, including
the grouped scalar expectation and execution metadata. Compiler specialization
errors must remain compiler errors, while planning/execution failures retain
their existing Runtime or Simulation exception ownership.

Phase 4 acceptance is complete. The compiler-produced `CircuitIR` passes through
the existing compiler, Runtime planner, and local CPU statevector path; all four
golden branch classes match the explicit reference and return one grouped
expectation per batch item. The result reports `local_statevector` and
`single_device_fast_path`. No Runtime, Simulation, root API, public contract, or
default-path implementation changed. Unknown observable axes now fail during
private Program IR verification rather than acquiring an accidental execution
meaning. That forward handoff is the input to the separately reviewed Phase 5
contract below.

## Phase 5 local CPU VJP authorization

Phase 5 may connect the bound `CircuitIR` to the existing Runtime-owned
statevector reverse executor. The accepted observable profile is a sum of one
or more single-wire Pauli-Z terms. Runtime may extend its internal reverse
executor from one Z wire to several Z wires by evaluating the circuit once and
constructing one adjoint seed equal to the sum of the term adjoints. Numerical
gate derivatives and statevector primitives remain Simulation-owned.

The gradient chain must preserve the original PyTorch scalar tensor views used
by lowering. Backward therefore accumulates from the existing private adjoint
result through those views into the original `weights` and `data` tensors;
host scalar conversion, detachment, NumPy conversion, or replacement by copied
leaf tensors is forbidden.

Control-flow derivatives are branchwise. Backward follows only the gates
selected during the corresponding forward specialization; comparison
predicates themselves are not differentiated. Equality at an ordered
comparison threshold is a non-smooth control boundary. Specialization must
record such boundaries and, when smooth gradients are explicitly required,
fail with a Compiler-owned diagnostic before lowering or execution.

Acceptance compares the adjoint VJP against dense PyTorch statevector autograd
and central finite differences away from branch boundaries. Parameter shift is
not an execution fallback. A deterministic optimization check must decrease a
seeded objective over multiple steps. The reverse result summary must identify
the adjoint method, observable profile, selected wires, and local
`single_device_fast_path` semantics.

This authorization does not cover higher-order derivatives, finite-shot or
noisy gradients, general Pauli strings or weighted Hamiltonians, accelerators,
distributed-gradient claims, performance claims, a public hybrid API, or a
change to the default `fq.run`/`fq.plan` path.

Phase 5 acceptance is complete. The existing statevector adjoint executor now
supports a sum of single-wire Z terms using one forward state and one combined
adjoint seed while preserving its original single-wire call form. Positive,
negative, and mixed branch paths match both dense PyTorch statevector autograd
and central finite differences for the original `weights` and `data` tensors.
A six-step deterministic optimization trajectory decreases monotonically.
Ordered-comparison equality boundaries are recorded and fail before lowering
when smooth gradients are required. The result summary exposes
`statevector_adjoint`, the accepted observable profile, selected wires, and
local backward semantics. Those results form the input to the separately
reviewed Phase 6 contract below.

## Phase 6 PyTorch graph-compilation authorization

Phase 6 may wrap one already specialized `CircuitIR` quantum region in a
private functional PyTorch custom operator. The operator receives every scalar
parameter tensor and the deterministic `CircuitIR` JSON as explicit inputs.
The template records the exact parameter order in
`hybrid_parameter_order`; a process-global program registry, opaque integer
handle, implicit mutable cache, or duplicated circuit representation is not
permitted.

Runtime owns the operator because it invokes the existing Runtime statevector
reverse executor. Compiler remains responsible only for specialization,
lowering, and deterministic template construction. The operator must register
a FakeTensor implementation and a first-order Autograd formula. The Autograd
formula calls a separate functional backward custom operator so AOT Autograd
can retain the whole supported region in a `fullgraph=True` graph. Backward
saves the explicit parameter tensors and immutable circuit text, then
rematerializes the existing statevector-adjoint execution context. This is a
correctness-first implementation, not a claim that quantum state replay has
been eliminated.

The supported profile remains analytic local CPU statevector execution, scalar
`float32` or `float64` parameters, and a sum of unit single-wire Pauli-Z terms.
All parameters must have one dtype and device and must exactly match the
template slots. Unsupported devices, shapes, observables, missing slots, or
extra slots fail closed before simulation. Only the selected quantum region is
compiled; this phase does not claim general dynamic Python capture, predicate
derivatives, higher-order gradients, accelerator or distributed execution, or
finite-shot/noisy gradients.

Acceptance requires `torch.library.opcheck`, numerical `gradcheck`, eager versus
compiled forward and first-order-gradient parity, and successful
`torch.compile(..., fullgraph=True)` execution with AOT Autograd. The custom
operator remains private: it is not exported from the package root and does
not alter `fq.run`, `fq.plan`, `fq.Module`, backend selection, or the default
non-hybrid path. Compile and execution timings may be recorded separately, but
no performance benefit is claimed.

Phase 6 acceptance is complete for this bounded profile. The serialized
template carries an explicit parameter order, the functional forward and
backward operators have FakeTensor coverage, `opcheck` and `gradcheck` pass,
and eager/AOT-compiled forward and first-order gradients agree under
`fullgraph=True`. Original tensor views still receive gradients. The backward
path deliberately rematerializes the statevector-adjoint context, and no
performance, public API, accelerator, distributed, or broader control-flow
claim is made.

## Phase 7 dynamic measurement-session authorization

Phase 7 may add a separately bounded measurement-feedback profile without
changing the analytic custom operator from Phase 6. Source capture represents
`qp.measure(wires=...)` as a `quantum.measure` operation that consumes and
returns the linear quantum effect and also produces a boolean SSA value. A
program may use that value directly as an `if` condition and return the
measured bit. This makes the data dependency explicit in Program IR rather
than evaluating a quantum measurement during Compiler specialization.

Compiler lowers both sides of the structured branch into the existing Core
`CircuitIR`: the measurement becomes an existing dynamic `measure`
instruction, and gates in the true and false regions carry conditions for the
same classical bit. Runtime, not Compiler, owns state collapse, the per-shot
classical register, conditional continuation, seeding, trajectory selection,
statistics, and result construction. Simulation continues to own statevector
gate primitives; this phase reuses the existing dynamic measurement
implementation and adds no numerical kernel. The first profile supports only
no-input programs, top-level measurement, direct measurement-bool conditions,
H/X/CX gates, and local CPU `complex64` or `complex128` execution.

The private Runtime entry point validates the compiler marker and all dynamic
metadata before constructing a transient `DynamicCircuit` and calling the
existing dynamic trajectory executor. Every shot starts from an independent
initial state, maintains its own collapsed quantum state and classical
register, follows its measured branch, and contributes one final sample. A
fixed integer seed must reproduce measurement bits, branches, and final
samples. The returned `DynamicExecutionResult.classical_bits` contains the
source-visible measurement values; metadata identifies which classical bit
was returned by the program.

Stochastic gradients are explicitly unsupported and fail closed when any
trainable tensor reaches the session. There is no parameter-shift,
score-function, straight-through, or silent deterministic-gradient fallback.
This authorization also excludes runtime inputs, measurement inside a branch,
arbitrary boolean expressions over measurement results, loops containing
measurements, reset, durable sessions across calls, `torch.compile`, finite-shot
gradient claims, accelerators, distributed execution, remote providers, and
performance claims.

Phase 7 acceptance requires exact `CircuitIR` lowering, observation of both
branches, shot-wise agreement among the measured bit, conditional gates, and
final samples, seeded reproducibility, equivalent reference and batched
trajectory semantics, and negative tests for invalid shot counts, classical
read-before-measurement, conditional measurement, and trainable parameters.
The entry points remain private and do not change the root API or the default
`fq.run`/`fq.plan` path.

Phase 7 acceptance is complete for this bounded profile. The compiler emits a
single Core `CircuitIR`; the existing Runtime dynamic trajectory machinery
performs state collapse and feedback; both reference and batched strategies
preserve the returned measurement semantics; and every excluded gradient or
control case fails closed. This result is a semantic vertical slice, not a
claim of general dynamic-program compilation or production dynamic-QPU
support.
