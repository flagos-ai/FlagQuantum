# Opt-in Module backend selection

Status: proposed for API-owner review. Tracking issue: #930. Proposed first
release: 0.5.0, conditional on approval and the evidence below. This PR records
the proposal and pins existing behavior in tests.

## User journey and alternatives

A user training an ordinary Module should be able to request a supported route
without predicting the relative cost of eager execution and JIT compilation.
Keep explicit PyTorch and JAX selection as the baseline. Improving only the
PyTorch path benefits existing callers and remains independently valuable, but
cannot make a measured JAX route accessible to an ordinary caller. Propose an
opt-in selector so users can request a decision and inspect its evidence.

```python
policy = fq.RuntimePolicy(
    execution_options=fq.ExecutionOptions(backend="auto"),
)
model = fq.Module(build, n_parameters=2, policy=policy)
result = model.execute(features)
```

## Protected surface and compatibility

- Retain the existing signatures of `ExecutionOptions`, `RuntimePolicy`,
  `Module`, `Module.execute`, `Module.forward`, and `fq.train`. Extend the
  accepted value of `ExecutionOptions.backend` for Module consumers with
  `"auto"`. Its current accepted Module values are `"pytorch"`, `"jax"`, and
  the omitted or `None` default resolved to PyTorch. Both old and proposed
  defaults select PyTorch.
- Keep `flagquantum.execution_options` v1.0 and `flagquantum.runtime_policy`
  v2.0 fields and serialization versions. Serialize the requested backend in
  the existing `execution_options.backend` field. Keep both existing checkpoint
  readers and the checkpoint schema. Old policies and checkpoints keep their
  explicit or default backend. A checkpoint with an auto policy restores the
  request and reevaluates eligibility on the destination. It does not persist a
  machine-specific selected implementation. A pre-0.5.0 runtime cannot execute
  an auto policy. To downgrade, set an explicit policy and save a checkpoint on
  0.5.0 before restoring on the older release.
- Explicit `"pytorch"` or `"jax"` bypasses selection. To opt out, set one of these
  values using the existing policy setter. Keep explicit mode, precision, device,
  derivative constraints, and authorized fallback policy authoritative. Existing
  unsupported explicit choices retain their refusal behavior.
- For auto, use the existing `ExecutionResult.compatibility` mapping for
  `requested_backend`, `selected_backend` and `fallback_used`. Add
  `runtime["execution_selection"]` with `reason`, `excluded`,
  `expected_calls` and `evidence_id`. Exclusions map backend names to stable
  reason strings. Use the existing executor and derivative diagnostics to report
  the actual implementation and its derivative. The initial selection within
  auto is a normal choice. An authorized change after the selected backend fails
  uses the existing fallback fields. A missing or inapplicable evidence row
  reports its reason and a null evidence identifier.

## Decision and acceptance

Runtime owns selection through the existing policy, planner and capability
checks. Executors own numerical implementation and eligibility facts. Start
with single-device CPU and existing PyTorch/JAX implementations. Check
dependency availability, operations, observables, requested derivative domain,
dtype, device, declared batching, and memory before comparing measured costs.
A loaded native extension alone cannot qualify a native derivative.

Treat standalone calls as one use and use the planned step count for a single
`fq.train` invocation. Select only when reviewed evidence for the workload
bounds the candidate's cold cost and total cost at the expected reuse. A
standalone call or training signature without evidence uses eligible PyTorch.
Keep the choice for that signature within the training call. Recheck eligibility
on a topology, policy, precision, device, batch shape or derivative-demand
change. Reuse evidence lookup without measuring inside optimizer updates.
Refuse execution when no eligible route satisfies the request.

Before implementation, obtain API-owner review of the value name, precedence,
serialization, migration and diagnostics. Then exercise the following fixtures
through the public workflow.

- Optional JAX unavailable, unavailable native derivative, unsupported operation
  and observable, and explicit fallback both enabled and disabled.
- A cold one-off call and repeated training where startup is amortized, with
  deterministic decisions and reasons tied to the recorded evidence.
- An existing checkpoint read, an auto checkpoint round trip, and explicit
  opt-out with matched optimizer state and continuation.
- The generated matrix and matched training budget in the
  [execution plan](MODULE_TRAINING_EXECUTION_PLAN.md), with held-out circuits,
  cold time-to-result, input gradients, declared batches, and recovery.

The existing [CPU matrix](../../artifacts/development/module_execution_demand_cpu_20261026/README.md)
records 19/50 proposed public-wrapper budgets and 0/50 default-accessibility
budgets met at its producer revision. The
[declared-batch comparison](../../artifacts/development/declared_batch_cpu_20261026/README.md)
records repeated CPU timings and separate whole-JAX results. These are inputs
to the review, with the existing performance gaps owned by #918 and #936.
The proposed selector still needs reviewed budgets and fresh matched evidence
before promotion. GPU and distributed certification require separate runs.
