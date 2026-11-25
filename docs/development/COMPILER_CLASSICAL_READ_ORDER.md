# Preserve classical readers during SABRE routing

## Correctness requirement

A measurement may reuse a classical bit after several conditional instructions
have read its previous value. Those readers can act on independent qubits and
therefore have no quantum dependency ordering them. Every reader must execute
before the next write to the bit. Tracking only the most recent reader is
insufficient.

On baseline `cf5fc36c29c3f9162b6b58054f8163c98944e33d`, a six-qubit line
exposes the failure with this source order:

1. Measure qubit 0 into classical bit 0.
2. Apply CX(1, 5) conditioned on bit 0 being one.
3. Apply X(2) conditioned on the same value.
4. Measure qubit 3 into classical bit 0.

The baseline deterministic trial emits source indices `(0, 2, 3, 1)`:
the second measurement can overwrite the value before the conditional CX reads
it. Preparing qubits 0 and 1 as one and qubit 3 as zero then gives the wrong
final value on qubit 5. This is an execution error, not just a different valid
schedule or an excessive SWAP count.

## Implementation boundary

The private SABRE dependency builder now retains all readers since the last
write for each bit. A subsequent write depends on all those readers and the
previous writer, then clears that bit's reader list. Read-after-write and
write-after-write edges remain in place. Reads on different qubits remain
independent; unrelated classical bits do not share a dependency chain.

Existing condition parsing deduplicates bit accesses within one instruction.
Graph edges still point forward in source order, including when an instruction
both reads and writes a bit: reads are appended after processing its write, so
no self-edge is introduced. This does not add support for conditional
measurement to the dynamic execution API.

The additional retained indices and anti-dependency edges are linear in the
number of classical read incidences. This is a complexity bound, not a measured
peak-memory result. Programs without classical accesses retain the same graph.
No public signature, result field, layout contract, strategy selection rule,
dependency, or SDKBench threshold changes. The path is
`single_device_fast_path`; there is no distributed or performance claim.

## Runnable user workflow

This public compiler and experimental dynamic workflow runs locally on CPU:

```python
import torch
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit

program = CircuitIR(6, (
    Instruction("h", (0,)),
    Instruction("x", (1,)),
    Instruction("measure", (0,), metadata={"classical_bit": 0, "is_dynamic": True}),
    Instruction("cx", (1, 5), metadata={"conditions": ((0, 1),), "is_dynamic": True}),
    Instruction("x", (2,), metadata={"conditions": ((0, 1),), "is_dynamic": True}),
    Instruction("measure", (3,), metadata={"classical_bit": 0, "is_dynamic": True}),
))
routed = compile(program, coupling_map=CouplingMap.line(6),
                 routing_strategy="sabre", optimize=False)
result = fq.experimental.dynamic.run_dynamic(
    DynamicCircuit.from_ir(routed), shots=32, seed=3,
)
assert torch.all(result.final_samples[:, 5] == result.final_samples[:, 0])
assert torch.all(result.classical_bits[:, 0] == 0)
```

Qubit 5 preserves the feedback from the first measurement even though the final
classical bit contains the second measurement. The test suite also compares the
entire result against execution of the original program, using X and H
preparation with both `auto` and `sabre`.

## Validation and limits

On macOS, Python 3.12.14 and CPU Torch 2.14.0, with numerical-library thread
counts set to one:

- The new regression module passes all 19 cases. Substituting only the baseline
  private trial implementation produces 16 failures and three passes. In
  particular, explicit SABRE fails actual dynamic execution with both X and H
  preparation; the auto cases pass on this baseline and are retained as controls.
- Each of the five trial variants is checked with both condition spellings and
  an independent basis-state interpreter over all 64 input states. A separate
  two-bit disjunction checks all 256 input states. The interpreter executes
  measurement, conditional X/CX and SWAP without consulting the planner graph.
- Repeated overwrites and an unrelated-bit negative control verify the scope of
  the dependency. Four trajectory tests use 32 shots and seed 3, including
  superposition, through the existing dynamic executor.
- The expanded routing and dynamic selection passes **191 tests, with four
  skips**. The skips retain their existing conditions; they are not execution
  evidence for their unsupported configurations.
- [The nonclassical plan evidence](evidence/classical-reader-order-plans.json)
  records 80 complete plan comparisons against the baseline: widths 6, 8, 12
  and 16, line and two-row grids, seeds 0 through 9, and four times width CX
  gates. For each seed, Python `random.Random(seed).sample(range(width), 2)`
  generates each pair. The complete dataclass dictionaries match, including
  schedule, intermediate placements, SWAPs, anchors and final layout. The file
  binds the corrected source by SHA-256. These are inspected development inputs,
  not a held-out campaign or an SDK ranking.

Reproduce the focused checks with:

```bash
python -m pytest tests/unit/test_compilation_sabre_classical_readers.py -q
python -m pytest tests/unit/test_compilation_sabre*.py \
  tests/unit/test_compilation_layer_lookahead.py \
  tests/unit/test_compilation_default_routing.py \
  tests/test_dynamic_circuit.py tests/test_dynamic_conformance.py -q
```

During test development an unsupported `route_to_topology(strategy="auto")`
call was corrected to the existing `compiler.compile` API. An initial sparse
checkout omitted contracts and a local example; restoring those tracked paths
resolved four setup failures without modifying their assertions. Neither run
is counted as a passing verification.

The complete default and runtime tiers have not been run locally for this
change: the host has approximately 189 MiB free disk space, and a previous
full-tier attempt for a separate change exhausted it. Complete-tier CI remains
required before merge. Formal SDKBench hardware qualification remains deferred;
no host reboot or qualification bypass is part of this change.
