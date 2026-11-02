# Proposed reusable Module lowering boundary

Status: **boundary explicitly approved by the user on 2026-11-02; implementation pending**. This is the review packet for
[#918](https://github.com/flagos-ai/FlagQuantum/issues/918) slice B
([#928](https://github.com/flagos-ai/FlagQuantum/issues/928)); it does not authorize
a new backend default or whole-training-step compilation in #936.

## Approved decision

The user approved one narrow Runtime-to-Compiler dependency for the existing Module builder
capture/lowering entry, replacing `Module._compile_builder_program`, after the
contract-only change lands. Runtime currently prohibits Compiler dependencies;
its domain instructions and dependency check must be amended in that reviewed
contract change, before implementation. Simulation continues to consume Core
contracts only. Do not route the dependency through a registry or another domain.

Keep the public PyTorch default and explicit JAX policy. A default JAX switch
would lose currently supported higher-order Torch derivatives. Keep explicit
precision, devices, representations, public signatures, policy serialization and
checkpoint schemas. No migration or release-version change is needed for these
private implementation changes. Any automatic policy or default change needs a
separate API proposal with owner approval under PUBLIC_API_PROTECTION.md.

Alternatives are to retain the current duplicate capture, or optimize only the
Torch path. Neither resolves the separate JAX re-execution of the Python builder.
The proposed boundary replaces capture; it does not add a second Module, quantum
IR, backend registry or planner. Review owners are Integration/API, Compiler,
Simulation and Runtime. The user approved the boundary in this packet; production replacement and CI review remain required. This is not approval of automatic selection or whole-step compilation.

## Producer and consumer responsibilities

Compiler captures the original builder's tensor arithmetic using the existing
Torch FX binding graph, emits canonical Core CircuitIR with existing named
Parameter placeholders, and records the ordered dynamic gate parameter slots.
It must not extend ParameterExpression to encode arbitrary Torch arithmetic.
Constants, gate order, wires, observable and metadata retain canonical semantics.
The binding graph is an ephemeral compiler artifact, not another serialized IR.

Runtime owns one bounded current-signature artifact, tensor binding, derivative
demand, invalidation, diagnostics and lifecycle. Simulation accepts canonical IR
and current tensor bindings; it must not import Compiler artifact types. JAX
lowers that same canonical template rather than invoking the original Python
builder again. Compiler owns the private capture result: canonical `template: CircuitIR`,
`binding_graph: torch.fx.GraphModule`, ordered `slots: tuple[tuple[int, str], ...]`
and the existing Circuit constructor properties. The private capture function
takes a builder callable and current flat tensor inputs, with named-parameter
order and input-presence flags, and returns that capture result. Runtime alone
retains it; Simulation receives only canonical IR plus tensor values. The test
fake demonstrates this data boundary without adding a production registry or
public protocol.

Re-use requires builder identity, topology, parameter names/order, binding
shape/dtype/device, declared batch, observable, precision and execution policy to
match. Changed values reuse the artifact. Changed signatures, Module.to(), policy
changes and checkpoint reload invalidate it. No per-step/value-keyed cache.
Every bound angle is validated before execution, including zero observables and
warm calls. Non-finite refusals retain the current error category; correctness
debug retains its documented diagnostic behavior.

Diagnostics and checkpoints describe the values actually consumed, even after
in-place optimizer/input updates. Deferred snapshots own their necessary values
and retain grad/inference context. They must not retain arbitrarily large unused
feature buffers or live autograd graphs. Checkpoint load preserves the existing
representative-execution requirement and optimizer ownership.

## Replacement acceptance before production implementation

`tests/unit/test_module_program_contract_fake.py` swaps only the private builder
adapter. Both implementations run the same ordinary public Module consumer.
The fake captures canonical symbolic IR once per builder generation and binds
new values without adding a production type or changing consumers. It checks
changing trainable features, shared nonlinear parameters, declared batches,
nonuniform output cotangents, independent analytic values/gradients and Adam
updates. It is a boundary feasibility test, **not** a compiled executor or a
performance result; it still uses the current builder capture and validation.

Production replacement conformance must additionally cover all supported gate
families/modes, all derivative demands, higher derivatives on the Torch route,
precision/device changes, checkpoint/optimizer recovery, hooks, unsupported
capture/operations, optional dependencies, bounded cache growth and cold/hot
capture counts. Compare old and replacement implementations through the same
public consumer. Unsupported capture keeps the current supported route with an
explicit reason; no hidden device transfer, precision change or silent fallback.

Whole-step compilation in #936 is a separate boundary review: it must preserve
caller-owned objectives/optimizers, zero_grad/execute/objective/backward/step,
callbacks, hooks, optimizer state and detached diagnostic results. This packet
makes no claim that arbitrary Python objectives or optimizers can be compiled.

## Performance acceptance and closure

Retain the existing proposed budgets, all failures and untouched controls.
Compare independent fresh-process trials, cold first result, warm calls,
realistic batches, fixed-step training/recovery, memory and supported hardware.
Record the actual native extension, backend, derivative, precision, JIT and
fallback. Public and external whole-JAX training are separate comparisons.
A faster private kernel or this contract fake cannot close #918. Remaining
selection/adoption (#930), whole-step (#936), batch/hardware (#929) and budget
losses need explicit owners and retained evidence before closure.
