# Resource-constrained fault-tolerant execution estimate

This is the internal Algorithms preview for [issue #1131](https://github.com/flagos-ai/FlagQuantum/issues/1131).
It estimates one explicitly decomposed static unitary Clifford+T program against
one caller-declared machine. It performs no QPU execution. The existing
`estimate_logical_resources` function and all of its fields retain their meaning.
The preview has no stable export; see the [interface proposal](../api-changes/FQ-FT-EXECUTION-1131.md).

Run from a source checkout with FlagQuantum's normal CPU dependencies:

```bash
PYTHONPATH=. python examples/algorithms/fault_tolerant_execution.py > /tmp/ft-estimate.json
PYTHONPATH=. python examples/algorithms/fault_tolerant_execution.py \
  --replay /tmp/ft-estimate.json > /tmp/ft-replay.json
cmp /tmp/ft-estimate.json /tmp/ft-replay.json
```

The example reports completion at tick **15**, a dependency lower bound of **8**,
a supply lower bound of **14**, and **151** provisioned physical qubits. These
are illustrative arithmetic assumptions, not hardware calibration. All inputs,
including the authoritative IR and machine assumptions, travel in the JSON;
replay reconstructs them and reruns the estimate. No random seed, network or
device is involved. Replay does not trust the old timings.

## Ownership and existing records

`team-ownership.toml` assigns this composition to Algorithms. The compiler's
`schedule_circuit_dependencies` supplies `CircuitSchedule` and its instruction
indices, predecessor edges, and identity. The original `CircuitIR` remains the
only program representation. The logical report embeds the compiler's existing
`ResourceEstimate`; rotated-surface-code patch arithmetic comes from
`logical_resources.py`. No compiler contract, scheduler IR, operator/code
registry, runtime path, stable API or dependency is added. This is local static
analysis, classified with the `single_device_fast_path` checks, and makes no
runtime or scalability claim.

## Required assumptions and conventions

Every machine field is explicit; there are no device constants or inferred
durations. Times use positive integer ticks of the caller's named `time_unit`.
Durations must be supplied for every opcode used. Startup may be zero; period
must be positive. Each of the homogeneous factories emits one state first at
`factory_startup + factory_period`, then once each period. All factories start
at time zero and emit simultaneously. Factory production precedes gate starts
at equal timestamps. An initial inventory is available at time zero.

Each T or T-dagger consumes one state at its start; its supplied duration must
include whatever injection work the caller means to charge. Stored states leave
storage at consumption, so storage slots are reusable immediately. Factories
run independently and continuously. Overflow is discarded, never borrowed from
the future. Events when full may be grouped as discarded production at the
next operation event; no inventory transition is lost. There is no aging model.

The named code convention is `rotated_surface_code_2d`, with odd distance at
least three and `2*d*d - 1` physical qubits per logical data patch. The machine
declares a logical data capacity, factory count and per-factory physical
footprint, inventory capacity, and per-slot storage physical footprint. Storage
footprints must include the caller's intended code protection. The physical
budget is optional only in the sense that the caller explicitly chooses `null`
to decline a budget check. Provisioned footprint is:

```text
logical_data_capacity * (2*d*d - 1)
  + factory_count * factory_physical_qubits
  + inventory_capacity * storage_physical_qubits_per_state
```

This includes unused provisioned capacity. It is a patch-and-factory sum, not a
layout. Every result, including an infeasible one, declares the all-to-all
assumption. There is no placement, routing, communication, measurement,
observable evaluation, classical feed-forward or factory lane-sharing model.
Terminal measurement records, dynamic/conditional instructions, noise channels,
custom matrices, undecomposed gates and parametric rotations are refused.

## How to interpret the result

- `dependency_time` is the duration-weighted critical path over the compiler's
  predecessor edges with unlimited magic supply.
- `supply_time` is a separate optimistic bound: the earliest production tick
  capable of supplying the total shortage, plus the shortest required T-family
  duration. It ignores dependencies and discards. Zero means initial supply
  suffices; `null` means a shortage has no factory to supply it. Neither bound
  is presented as a feasible schedule.
- `operations` records source instruction index, start/end ticks, predecessor
  facts from the compiler, and the operation's wait after becoming ready.
  Ready operations are considered in source-index order. An empty inventory
  stalls a T gate while independent ready Clifford gates can still start.
  A wire is exclusive until its operation completes.
- `inventory` records initial, production and consumption transitions, plus
  discarded production. `peak_inventory` measures occupied slots, whereas the
  physical budget pays for provisioned slots.
- `stalls` records intervals with ready T gates blocked by missing supply.
  `factory_stall_time` is the union length of these intervals; other work may
  overlap them. It is not generally the difference from the dependency bound.
- `capacity_infeasible` means the register, storage, total supply without a
  factory, or physical budget cannot support the request. It carries reasons,
  no completion time or peak inventory, and no partial schedule.
- `failure_budget.status` is always `unavailable`. No calibrated error model or
  provenance is implemented. Code distance alone never produces a success
  probability; this preview makes no failure-budget claim.

This deterministic greedy schedule is feasible under the declared abstraction;
it is not an optimality guarantee. The hand-solvable cases and seeded source-wire
and inventory conservation checks are in
`tests/unit/test_fault_tolerant_execution.py`.
