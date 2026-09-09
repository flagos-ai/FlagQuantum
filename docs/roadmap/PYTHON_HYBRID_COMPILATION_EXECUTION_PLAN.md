# Python-first hybrid quantum-classical compilation execution plan

Status: **Phases 1-8 bounded vertical slices complete**
Owner: Compiler, with Integration approval for cross-domain contracts
Initial target: local CPU `single_device_fast_path`
Implementation language constraint: no FlagQuantum-authored C++ in Phases 0-6
External compiler constraint: Catalyst is out of scope for this plan

## 1. Outcome

Deliver a FlagQuantum-owned path that captures, validates, specializes, lowers,
executes, and differentiates a bounded hybrid quantum-classical Python program.
The first complete scenario is a parameterized cost function containing nested
loops and data-dependent branches that choose quantum gates, followed by an
expectation value and a first-order gradient.

The first accepted vertical slice must prove this flow:

```text
restricted Python function
  -> private hybrid Program IR
  -> verified structured control flow
  -> selected quantum operation trace
  -> existing CircuitIR
  -> existing compiler and local statevector simulation
  -> scalar expectation
  -> VJP back to the original tensor parameters
```

The work is not complete when new IR classes exist. It is complete only when the
scenario above runs end to end, fails closed outside its declared subset, and
passes independent forward and gradient differentials.

## 2. Product and architecture constraints

This plan obeys the following repository constraints:

- `CircuitIR` remains the stable public exchange representation and the source
  consumed by existing simulation paths.
- The hybrid IR is private until a separate public API proposal is approved.
- Existing `fq.run`, `fq.plan`, `fq.Module`, `fq.compile`, result types, and
  serialized schemas do not change in the initial phases.
- The existing local CPU and single-device paths must not incur hybrid compiler
  overhead unless the new path is explicitly selected.
- Compiler captures and transforms programs; Runtime owns execution lifecycle;
  Simulation owns numerical kernels; Ecosystem owns external-framework adapters.
- Unsupported control flow, mutation, operations, gradients, measurements,
  shapes, and effects fail before execution whenever they are statically known.
- No performance, scale, hardware, or production claim is made from CPU
  conformance tests.
- No duplicate plugin registry, backend manager, execution result, or public IR
  is introduced.

## 3. Explicit non-goals

The following are excluded from the first six phases:

- Catalyst integration or compatibility claims;
- a C++ compiler, C++ runtime, native device ABI, or Python-free executable;
- general Python compilation;
- arbitrary Python objects, reflection, generators, exceptions, I/O, networking,
  or mutation inside captured functions;
- provider submission, QPU credentials, queue management, or cloud execution;
- QIR, pulse, timing, or controller firmware generation;
- distributed scalability claims;
- finite-shot, noisy, or mid-circuit-measurement gradients;
- higher-order differentiation;
- automatic public API changes.

These exclusions are capability boundaries, not silent fallbacks.

## 4. Existing assets and authoritative boundaries

### 4.1 Current vNext assets

The current repository already provides:

- stable `CircuitIR`, instructions, measurements, observables, serialization,
  and content identity in `flagquantum/core/ir.py`;
- target-independent optimization and topology-aware circuit compilation in
  `flagquantum/compiler`;
- local statevector, MPS, tensor-network, and selected gradient executors;
- `fq.Module` with PyTorch-first training behavior;
- a JAX kernel path exposed through `torch.autograd.Function`;
- an internal `ProgramArtifact` candidate envelope;
- a compiler extension boundary that currently consumes and returns `CircuitIR`.

None of these types is to be duplicated under a new name.

### 4.2 Historical research implementation used as evidence

Repository history and the retained workspace copy contain a former private
compiler tree with:

- immutable types, values, operations, blocks, regions, and modules;
- deterministic identities and frozen attributes;
- operation schemas and a fail-closed verifier;
- def-use and qubit-lifetime analyses;
- pass management, static canonicalization, decomposition, placement, routing,
  emitters, target capabilities, TargetIR, and executable artifact prototypes.

It was removed from the current branch because it was disconnected from the
public/default product path. It is evidence for semantics and negative tests,
not a source tree, branch, or class hierarchy to restore. Migration means
re-expressing only required behavior inside current vNext authorities; copying
the former `_compiler`, `QuantumIR`, `TargetIR`, artifact, provider, or runtime
layers is prohibited.

### 4.3 Missing capability

The missing capability is not another static circuit IR. It is the program-level
semantics that preserve tensor values and structured classical control around
quantum effects:

```text
function / call / return
tensor argument / extract / shape
arithmetic / comparison
if / for / while / yield
quantum region / operation / expectation
forward context / VJP
```

## 5. Target architecture

```text
Python source and tensor inputs
           |
           v
Hybrid frontend capture
  AST structure + framework tensor values
           |
           v
Private Program IR
  functions / blocks / regions / typed SSA values / effects
           |
           +--> verifier and capability diagnostics
           |
           v
Partial evaluation and specialization
  static Python values resolved; runtime tensor decisions retained
           |
           v
Quantum-region lowering
  selected operations + late-bound parameter references
           |
           v
Existing CircuitIR and Compiler
           |
           v
Existing Runtime and Simulation
           |
           +--> forward value
           |
           v
Quantum VJP + classical VJP
           |
           v
Gradient in the originating PyTorch graph
```

This is a semantic migration, not a restoration of the historical layer stack:

```text
historical capability                 vNext authority
---------------------                 ---------------
program-level SSA/control/effects  -> flagquantum.compiler._hybrid
static quantum region              -> existing flagquantum.core.ir.CircuitIR
optimization/routing/target emit   -> existing flagquantum.compiler
execution lifecycle                -> existing flagquantum.runtime
numerical state and kernels        -> existing flagquantum.simulation
```

The first phase adds only the missing program-level semantics. A separate
`QuantumIR` or `TargetIR` may be introduced later only if an existing vNext type
cannot express a verified requirement and a new contract is approved.

The architecture has two execution modes:

1. **Path-specialized static region**: evaluate runtime classical branches,
   record the selected quantum operations, lower the resulting region to
   `CircuitIR`, and cache the compiled execution shape.
2. **Structured session**: preserve control-flow boundaries and execute quantum
   regions in sequence. This mode is reserved for later measurement feedback;
   it is not required for the first vertical slice.

## 6. Private Program IR semantic minimum

### 6.1 Types

The first profile requires:

```text
BoolType
IndexType
ScalarType(dtype)
TensorType(dtype, rank, static-or-symbolic dimensions)
QubitRefType
ObservableType
ExpectationType(dtype)
EffectTokenType
```

Tensor values retain shape, dtype, device class, and origin. Conversion to a
Python `float`, `.item()`, NumPy scalar, or detached host value is forbidden for
runtime parameters.

### 6.2 Operations

The first profile requires only the operations used by the golden scenario:

```text
program.func
program.return
tensor.dim
tensor.extract
arith.constant
arith.add
arith.rem
arith.cmp
scf.for
scf.if
scf.yield
quantum.angle_embedding
quantum.rx
quantum.ry
quantum.cx
quantum.expectation
gradient.vjp
```

Operation names are illustrative internal names. They do not create a public
MLIR dialect or compatibility promise.

### 6.3 Values and effects

- Every value has one deterministic definition and typed uses.
- Branch result shapes and types match at joins.
- Loop-carried values have explicit initial, iteration, and result values.
- Quantum operations are ordered through a quantum effect token or an
  equivalent verified linear-state rule.
- A quantum value/effect cannot be duplicated across branches.
- Runtime parameter references remain late-bound through circuit lowering.
- Source locations are diagnostic metadata and do not change semantic identity.

### 6.4 First-profile verifier

The verifier rejects:

- use before definition and duplicate definition;
- operand/result type mismatch;
- branch signature mismatch;
- invalid loop bounds or loop-carried values;
- non-tensor runtime predicates;
- unsupported Python side effects;
- conversion of differentiable tensor parameters to host scalars;
- unsupported quantum operations or observables;
- qubit/effect reuse, loss, or ambiguous branch merge;
- gradients requested through unsupported measurements or dynamic effects;
- implicit backend or gradient-method fallback.

## 7. Frontend capture boundary

### 7.1 Supported Python subset

The initial frontend accepts:

- a decorated function with positional tensor inputs;
- local scalar/tensor assignments;
- `if`/`elif`/`else` with tensor-scalar predicates;
- `for` over `range` and statically ranked tensor axes;
- nested supported control flow;
- supported arithmetic, comparison, indexing, and FlagQuantum calls;
- a single scalar expectation return.

### 7.2 Unsupported subset

The frontend rejects with source-located diagnostics:

- object or container mutation visible outside a region;
- `break`, `continue`, `yield`, exceptions, context managers, comprehensions,
  generators, recursion, and dynamic imports;
- file, process, network, logging, printing, and global-state effects;
- `Tensor.item()`, `float(Tensor)`, NumPy conversion, or detach on a required
  gradient path;
- data-dependent tensor rank or result signature;
- calls without a registered capture rule.

### 7.3 Capture implementation sequence

The first frontend uses Python `ast` to preserve native control-flow structure.
It does not execute runtime tensor predicates during capture. PyTorch FX/Dynamo
integration follows only after Program IR semantics and direct execution are
proven. This avoids making a framework graph a second source of quantum truth.

## 8. Lowering and execution

### 8.1 Partial evaluation

Compile-time constants, static ranges, and source structure are resolved once.
Runtime tensor values remain typed parameters. Constant folding must never read
a runtime tensor value through `.item()`.

### 8.2 Path-specialized lowering

For the initial profile, a runtime invocation evaluates structured classical
predicates and records the selected gate sequence. The trace is lowered into the
existing `CircuitIR` using late-bound tensor expressions or parameter slots.

The cache separates:

```text
program structure identity
input signature identity
selected quantum structure identity
parameter values
```

Parameter values never enter the reusable structure identity unless explicitly
declared static. Branch-dependent structures may produce multiple bounded cache
entries. Cache exhaustion fails or evicts visibly; it must not silently compile
without accounting.

### 8.3 Existing compiler and simulation reuse

The lowered `CircuitIR` passes through the existing compiler. The first runtime
target is local CPU statevector. MPS, tensor network, GPU, JAX, distributed, and
provider paths are excluded until the same scenario is independently accepted
for those modes.

This path is classified as `single_device_fast_path`.

## 9. Gradient contract

### 9.1 Required invariant

For every differentiable runtime input, the compiler preserves this chain:

```text
source tensor
  -> Program IR SSA value
  -> gate parameter slot
  -> quantum expectation
  -> quantum VJP
  -> classical VJP
  -> source tensor gradient
```

No source-to-IR, IR-to-circuit, runtime, result, or framework adapter may detach,
copy through a scalar host value, or replace an unsupported derivative with zero.

### 9.2 Initial method

The first profile computes a VJP for a scalar analytic expectation on a pure
statevector. The implementation may use an existing FlagQuantum reverse/adjoint
executor where its semantic contract matches the selected circuit. Parameter
shift is allowed only as an explicitly selected reference method, never as a
silent performance fallback.

### 9.3 Control-flow differentiation

Reverse execution follows the branch selected by the forward invocation and
reverses loop iterations in execution order. A predicate such as `p > 0` is not
differentiated as a smooth function. Gradients are branchwise and the switching
boundary is reported as non-smooth.

### 9.4 PyTorch integration

After the direct internal path is accepted, the quantum region is registered as
a functional PyTorch custom operation with:

- an explicit schema and no hidden input mutation;
- FakeTensor/meta behavior;
- saved forward context;
- a registered VJP/backward rule;
- `opcheck` and `gradcheck` coverage;
- `torch.compile` full-graph and graph-break tests.

## 10. Golden scenario

The first scenario is semantically equivalent to:

```python
def cost(weights, data):
    angle_embedding(data, wires=range(4))
    for layer in weights:
        for j, p in enumerate(layer):
            if p > 0:
                rx(p, wire=j)
            elif p < 0:
                ry(p, wire=j)
        for j in range(4):
            cx(j, (j + 1) % 4)
    return expectation(z(0) + z(3))
```

Acceptance requires:

- positive, negative, zero, and mixed branch patterns;
- repeated invocation with different values and the same signature;
- branch structure cache identity and bounded cache behavior;
- deterministic IR identity independent of source-location movement;
- forward agreement with an explicit hand-built `CircuitIR` reference;
- gradient agreement with central finite differences away from branch boundaries;
- gradient agreement with the existing uncompiled FlagQuantum path where
  equivalent;
- optimization steps that decrease a seeded differentiable objective;
- explicit rejection at `p == 0` when a test asks for a smoothness guarantee;
- no change in existing non-hybrid `fq.run`, `fq.plan`, or `fq.Module` behavior.

## 11. Execution phases and gates

### Phase 0 — reconcile and freeze the vertical slice

Deliverables:

- current/removed implementation inventory;
- authoritative ownership map;
- exact golden scenario and unsupported boundary;
- reuse/subtraction decision for each old compiler component;
- baseline tests for existing circuit compile, statevector forward, and gradient;
- API impact statement.

Exit gate:

- the current compiler-extension/public API work is committed or otherwise
  cleared from the integration worktree;
- Integration approves the private cross-domain contract;
- no public schema or root export is changed;
- Compiler and Runtime owners agree on the `CircuitIR` handoff and VJP boundary.

### Phase 1 — private Program IR semantic core

Compiler-owned files only. Deliver immutable values, operations, blocks,
regions, types, schemas, deterministic identity, source diagnostics, and the
first-profile verifier.

Exit gate:

- positive construction tests;
- negative verifier tests for every declared invariant;
- deterministic identity tests;
- contributor golden path under ten minutes;
- no import from Runtime, Simulation, provider, or ecosystem modules.

### Phase 2 — restricted Python capture

Deliver AST capture for the supported subset and source-located rejection for
the unsupported subset.

Exit gate:

- golden source captures without evaluating runtime predicates;
- all prohibited constructs fail closed;
- equivalent formatting and local renaming preserve semantic identity where
  names are not semantically observable;
- capture does not modify globals or user tensors.

### Phase 3 — path specialization and CircuitIR lowering

Deliver partial evaluation, runtime branch selection, gate tracing, parameter
slots, structure cache, and lowering to the existing `CircuitIR`.

Exit gate:

- all golden branch patterns match hand-built `CircuitIR` references;
- parameter values remain differentiable tensors;
- structure identities exclude late-bound values;
- existing compiler passes accept the lowered output;
- unsupported dynamic effects fail before simulation.

### Phase 4 — local CPU forward vertical slice

Contract-first cross-domain work. Runtime accepts the compiler-produced existing
artifact at an internal boundary and invokes existing statevector simulation.

Exit gate:

- scalar expectation matches the explicit reference for all branch patterns;
- errors retain compiler/runtime/simulation ownership;
- no default path or backend fallback changes;
- local fast-path regression budget is met when hybrid compilation is unused.

### Phase 5 — first-order VJP vertical slice

Deliver quantum VJP, classical chain rule, branch/loop reverse semantics, and
finite-difference differential tests.

Exit gate:

- gradients match numerical references away from branch boundaries;
- zero/detached gradients fail tests;
- gradient method and limitations are observable;
- a seeded optimizer decreases the objective over multiple steps.

### Phase 6 — PyTorch graph compilation

Deliver the functional custom operation, Autograd registration, FakeTensor
contract, and `torch.compile` integration.

Exit gate:

- eager and compiled forward/gradient parity;
- `opcheck` and `gradcheck` pass;
- no unexpected graph break in the supported scenario;
- compile latency, cache hit, steady-state step time, and memory are reported
  separately without a performance claim unless gates are met.

### Phase 7 — dynamic measurement session

This phase requires a separately approved Runtime/Simulation contract. Add
mid-circuit measurement values, stateful sessions, conditional continuation,
shot semantics, and stochastic-gradient policy.

It is not a prerequisite for claiming the bounded classical-data-dependent
hybrid control-flow profile from Phases 0-6.

### Phase 8 — parameterized bounded dynamic session

Extend the private Phase 7 session with non-trainable scalar, index, and bool
runtime inputs. Specialize positive bounded `range` loops, lower RX/RY values
through explicit Core `Parameter` slots, and keep parameter values outside the
template identity.

Exit gate:

- equal structure with different parameter values has one template identity;
- bound values preserve the original scalar tensor objects;
- loop bounds are visible and limited by a configurable unroll ceiling;
- eager dynamic shots respond to parameter changes under deterministic oracles;
- trainable inputs, unbound templates, tensor inputs, and loop measurements
  fail closed;
- no finite-shot gradient, graph-compilation, accelerator, distributed, or
  performance claim is introduced.

## 12. File and team ownership plan

Expected Compiler-owned implementation (files are added only when their phase
needs them):

```text
flagquantum/compiler/_hybrid/
  README.md
  model.py
  schemas.py
  verifier.py
  capture.py
  specialize.py
  lowering.py
  diagnostics.py
```

Expected shared tests:

```text
tests/hybrid_compiler/
  test_program_ir.py
  test_verifier_negative.py
  test_capture.py
  test_lowering.py
  test_forward_vertical.py
  test_gradient_vertical.py
  test_torch_compile.py
```

Runtime and Simulation changes must be separate domain-owned changes after an
Integration-owned contract is approved. Core types or public APIs are not to be
changed by a team branch.

## 13. Verification ladder

For each phase:

1. run the smallest new focused unit tests;
2. run existing compiler tests;
3. run existing statevector and gradient tests when the handoff is touched;
4. run `python tools/check_team_scope.py` for the owning team;
5. run `python tools/check_architecture.py`;
6. run `python tools/ci_tier.py pr-default`;
7. run `python tools/ci_tier.py pr-runtime` for vertical execution changes;
8. compare the protected public API snapshot without regenerating it.

All commands must use the repository-supported Python environment and bounded
timeouts. Empty test selections are not evidence.

## 14. Performance evaluation

The first performance report separates:

- source capture time;
- Program IR verification and lowering time;
- branch specialization time;
- structure-cache hit rate;
- circuit compile time;
- statevector forward time;
- VJP time;
- end-to-end eager step time;
- end-to-end compiled steady-state step time;
- peak host and device memory.

Compilation benefit is reported only after amortization:

```text
break_even_steps = compile_time / (eager_step_time - compiled_step_time)
```

No kernel, distributed, or capacity benefit may be attributed to hybrid
compilation without direct evidence.

## 15. Risks and controls

| Risk | Control |
| --- | --- |
| Another disconnected compiler tree | Every phase ends in the same golden vertical slice; unused abstractions are removed |
| Public API drift | Private namespace first; separate proposal before any root export |
| Gradient detachment | SSA origin tracking, forbidden scalar extraction, VJP and optimizer tests |
| Branch-cache explosion | Bounded cache, visible eviction/failure, structure/value identity separation |
| Python interpreter overhead | Region batching first, PyTorch compilation after semantics pass |
| PyTorch becomes the quantum truth | Program IR owns quantum semantics; PyTorch owns classical graph execution |
| Silent unsupported behavior | Capability verifier and typed diagnostics before execution |
| Local-path regression | Hybrid path is opt-in and benchmarked separately |
| Old code volume returns | Migrate behavior by need; do not copy provider/deployment machinery |

## 16. Stop and review conditions

Stop implementation and return to Integration review if:

- a protected public API or serialized Core schema must change;
- the vertical slice requires modifications in four or more primary domains;
- runtime tensor values cannot remain attached through current `CircuitIR`;
- the existing compiler-extension contract conflicts with hybrid artifacts;
- the local CPU path slows when hybrid compilation is unused;
- an unsupported feature would require silent fallback;
- a second source of truth for circuits, results, backends, or plugins appears.

## 17. Current progress

- [x] Workspace and repository-rule inspection
- [x] Current static compiler and extension-boundary inspection
- [x] Historical multi-level IR implementation located
- [x] Historical core IR tests sampled successfully
- [x] Static versus dynamic Program IR boundary identified
- [x] Detailed execution plan recorded
- [x] Existing compiler-extension/public API worktree changes cleared
- [x] Phase 0 private semantic contract approved conditionally
- [x] Machine-readable entry gates added
- [x] Whole-tree restoration explicitly rejected
- [x] Migration mapped onto current vNext authorities
- [x] Integration team-scope preflight passed for the Phase 1 slice
- [x] Phase 1 private semantic slice implemented and verified
- [x] Phase 2 private capture contract authorized
- [x] Phase 2 restricted Python capture implemented and verified
- [x] Phase 3 private specialization/lowering contract authorized
- [x] Phase 3 path specialization and `CircuitIR` lowering verified
- [x] Phase 4 existing `CircuitIR`/`MeasurementNode` handoff contract authorized
- [x] Phase 4 local CPU forward vertical slice verified
- [x] Phase 5 local CPU VJP contract authorized
- [x] Phase 5 branchwise adjoint VJP vertical slice verified
- [x] Phase 6 functional PyTorch custom-op contract authorized
- [x] Phase 6 FakeTensor, Autograd, opcheck, gradcheck, and fullgraph path verified
- [x] Phase 7 Runtime/Simulation contract authorized for a bounded profile
- [x] Phase 7 measurement-value, conditional-session, shot, and gradient-policy slice verified
- [x] Phase 8 parameter-slot and bounded-loop contract authorized
- [x] Phase 8 non-trainable parameterized dynamic-session slice verified
