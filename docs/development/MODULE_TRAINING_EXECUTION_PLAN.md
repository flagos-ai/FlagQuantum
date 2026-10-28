# Ordinary Module execution: implementation and acceptance plan

Status: proposed architecture work under [#918](https://github.com/flagos-ai/FlagQuantum/issues/918).
Only derivative-demand execution described below is implemented in this change.
Batch semantics and automatic backend selection remain proposals requiring
API-owner review. This document grants no approval and freezes no new contract.

## User outcome

An ordinary PyTorch training loop should obtain an effective supported execution
path without private imports, manually built bridges or circuit-specific flags:

```python
import torch
import flagquantum as fq

def build(theta, features):
    return fq.Circuit(2).ry(0, theta[0] + features[0]).cx(0, 1)

model = fq.Module(build, 1, init=torch.tensor([0.2]))
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
for _ in range(10):
    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.tensor([0.1])).square().mean()
    loss.backward()
    optimizer.step()
with torch.no_grad():
    prediction = model(torch.tensor([0.3]))
assert torch.isfinite(prediction).all()
```

This code runs today using the PyTorch default. Automatic route selection and
batched feature execution are future outcomes, not properties of this example.
The existing explicit JAX policy remains supported. Installed native-adjoint
availability does not imply that Module uses that derivative implementation.

## Reuse existing responsibilities

Do not introduce another public Module, IR, registry or planner. Module owns
parameter state and the Torch interface; FlagQuantum IR owns circuit meaning;
Compiler owns transformations; Simulation owns numerical execution and
derivatives; Runtime coordinates execution, resources and snapshots. Extend
existing compiled builder and executor boundaries rather than creating a
second circuit language or a benchmark-only execution path.

The desired reusable execution consists of static topology and observables,
dynamic parameter/input bindings, and a supported derivative implementation.
Runtime must key reuse by topology, binding shape/dtype/device, observable,
precision and execution policy. Cache invalidation must cover Module.to(), policy
changes, changed input signatures and checkpoint reload. Cache growth must be
bounded; no cache keyed by every parameter value or training step is acceptable.

## Ordered vertical slices

| Slice | Primary owner | Replacement or extension | Exit criteria |
| --- | --- | --- | --- |
| A: derivative demand and matrix | Simulation; shared benchmark tests | Replace unconditional JAX value-and-gradient execution at the existing kernel boundary | All four parameter/input gradient masks, no_grad/inference, changing values, scalar and existing parameter-batch shapes preserve values/VJPs. Public Module inherits the behavior. |
| B: one reusable IR-derived program | Compiler, followed by Runtime/Simulation | Replace duplicated Torch FX capture plus re-invocation of a Torch-shaped Python builder inside JAX tracing | Same public consumer runs with old and replacement implementation. General static gate recipes work without a JAX-specific builder rewrite. Cold trace/build counts, hot binding cost, checkpoint hashes and invalidation are measured. |
| C: input batches and observable VJPs | Existing frozen Proposal 005; Simulation/Runtime conformance | Implement existing explicit Circuit.bsz semantics and executor VJP, rather than exposing a new batch class | Shared-parameter input batch matches a serial oracle in values, parameter/input gradients, Adam updates and checkpoint recovery. Nonuniform output cotangents are tested. |
| D: effective route selection | API-owner contract first; then Runtime | Extend existing policy/planning selection | Explicit requests stay authoritative. Automatic selection only considers supported implementations and explains choice, derivative, precision, device and exclusions. Missing optional dependencies and unsupported operations have tested outcomes. |
| E: adoption and performance acceptance | Runtime; shared benchmark/docs | Existing quick starts, examples and performance catalog | Public entry points pass the reviewed hardware/workload matrix; every remaining loss has an owner. No claim is based solely on an internal kernel. |

Slice A is implemented here. B through E are open. Do not mark the parent done
because A or an internal kernel is fast. B requires a separately reviewed
cross-domain lowering contract and replacement test before implementations.
Implementation issues: [B: reusable program #928](https://github.com/flagos-ai/FlagQuantum/issues/928),
[C: input batches #929](https://github.com/flagos-ai/FlagQuantum/issues/929),
[D/E: execution selection #930](https://github.com/flagos-ai/FlagQuantum/issues/930).

## Derivative demand

Derive demand from current Torch grad mode and tensor requires_grad flags on
each call. Cache JIT variants by the supported derivative demand, not by values.

| Grad enabled | Parameter gradient | Input gradient | Work |
| --- | --- | --- | --- |
| no | any | any | value only |
| yes | no | no | value only |
| yes | yes | no | value and parameter derivative |
| yes | no | yes | value and input derivative |
| yes | yes | yes | value and both derivatives |

Variants compile lazily. A validation phase can incur an additional first JIT;
report that separately and include it in time to a trained result. Module.eval()
alone does not disable gradients. Do not infer grad demand from training mode.
Existing first-order differentiation limits remain; no higher-order claim is made.

## Decisions that need API-owner approval

Batch contract: specify feature axes, output axes, parameter sharing, gradient
reduction, empty batches and noncontiguous tensors. Distinguish parameter batches
from input batches; do not silently form a Cartesian product. The current
scalar-output bridge multiplies saved gradients by an upstream scalar/batch
scale. Shared parameters with vector outputs require a true cotangent contraction
and reduction, not that broadcast rule. Arbitrary leading dimensions cannot be
interpreted as a batch without an approved contract.

Selection contract: retain current explicit backend precedence and preserve
serialized policies/checkpoints. Review an opt-in automatic policy before any
default change. Name old/proposed defaults, signatures and serialized fields in
the API proposal; include old-checkpoint reading, opt-out, migration and release
version. Selection must account for gradient support, precision, device, memory,
batch shape and expected reuse. Kernel timing alone cannot select a backend for
a one-call job; compilation cost matters. Prohibit hidden device transfers,
precision reduction, representation changes or unsupported fallback.

The proposals must include alternatives (keep explicit selection, improve only
the Torch route, or opt-in selection), concrete journey, compatibility analysis,
error cases, naming review, approval owner and executable conformance fixtures.
See [public API protection](PUBLIC_API_PROTECTION.md). No new stable export is
needed just to make this work discoverable.

## Executable development matrix

```bash
python -m benchmarks.module_execution_matrix --output /tmp/fq-module-matrix
# Opt into failure on proposed development budgets:
python -m benchmarks.module_execution_matrix --output /tmp/fq-module-budget --enforce
```

The current CPU matrix uses widths 2/4/6, depths 1/3 and chain/ring connections,
omitting the duplicate two-wire ring. Ten shape processes each measure five
derivative contexts and three routes: default Module, explicit JAX Module and
the same JAX executor as an internal diagnostic. Five timing blocks rotate
route order, with 20 completed calls per block. First calls in each context are
separate; they reuse earlier contexts' caches and are not cold-process rankings.
Every route checks values and requested gradients against Torch execution.
Analytic derivative-demand tests independently check numerical correctness.

Proposed targets are `public <= 1.15 * kernel + 5 us` for wrapper overhead and
`default <= 1.25 * best_measured_public_route + 5 us` for accessibility. These are
engineering targets, not approved release thresholds. Keep failed cases and
full samples/MAD. Do not loosen targets after seeing a candidate. The script
does not run in CI automatically and does not compare external frameworks.
Timeouts, worker failures and numerical mismatches fail even without --enforce.

Extend the existing performance catalog and matched training protocol for final
acceptance: repeated fresh processes, complete cold imports/build/JIT, batch sizes
1/8/64, widths/depths limited by declared memory budgets, changing features,
multiple observables, native-adjoint and applicable compiled routes, CPU and
available GPU devices, time-to-target, memory, and optimizer/checkpoint parity.
Compare the strongest supported external routes with matching precision,
derivatives, batching and quality. Do not assume a serial-example result proves
best batched throughput. Reserve circuit seeds/topologies before implementation
and use an untouched evaluation set to detect workload-specific optimizations.

## Completion criteria

No universal fastest-framework promise is certified. Completion requires
ordinary public workflows to meet reviewed budgets across the supported matrix,
independent correctness and recovery checks, explicit unsupported cases and
documented reproducible losses. GPU/distributed acceptance stays unverified until
run on actual hardware. All local evidence remains single_device_fast_path and
does not authorize a scalability claim. Track finite-angle binding refusal in
[#925](https://github.com/flagos-ai/FlagQuantum/issues/925); this work does not
weaken or certify that existing validation boundary.

## Declared-batch implementation

The optional JAX Module route now executes explicitly declared Circuit.bsz batches
with scalar, length-one or length-bsz gate angles in statevector, MPS and TN.
It uses cotangent-weighted VJPs rather than full Jacobians. Tests cover all four
gradient-demand masks, nonuniform cotangents, changing and noncontiguous inputs,
Adam recovery and retained consumed values. Forward and backward compile
separately; backward recomputes numerical work. Automatic input-axis inference,
default backend selection and whole-training compilation remain unimplemented.
See API_CHANGE_PROPOSAL_MODULE_JAX_DECLARED_BATCH.md for the compatibility
assessment under Proposal 005; it introduces no new public contract.
