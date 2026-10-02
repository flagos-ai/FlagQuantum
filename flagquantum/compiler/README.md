# Compiler

Transform quantum programs while preserving their numerical meaning,
measurement behavior, and required gradients. The design direction is one
verified path from quantum-classical source to target-legal output.

Compiler owns program transformations over Core semantics. Runtime chooses
execution; Simulation supplies numerical kernels; Compute and Remote adapt
resources. Compiler does not run the program.

## Start here

From the repository root:

```bash
python -m examples.compiler_optimize
python -m examples.target_aware_compilation
```

The first example checks optimization against the original circuit. The second
checks routing legality and numerical equivalence on a concrete topology.
Use `optimize(program)` for target-independent optimization and
`compile(program, coupling_map=...)` for target-aware compilation.

## Change the owning stage

| Change | Entry point |
| --- | --- |
| Canonical optimization | [pipeline.py](pipeline.py) |
| Connectivity and routing | [routing.py](routing.py), [sabre.py](sabre.py), [topology_legalization.py](topology_legalization.py) |
| Wire layouts and the layout restore | [layout.py](layout.py) |
| Initial placement on a device | [layout_planning.py](layout_planning.py) |
| Native-gate and target requirements | [native_gate_legalization.py](native_gate_legalization.py), [target_legalization.py](target_legalization.py) |
| Dependency scheduling | [schedule_legalization.py](schedule_legalization.py) |
| Emission and round-trip checks | [target_emission.py](target_emission.py), [target_conformance.py](target_conformance.py) |
| OpenQASM interchange | [openqasm.py](openqasm.py), [openqasm_gates.py](openqasm_gates.py), [openqasm_import.py](openqasm_import.py) |
| Structured hybrid programs | [_hybrid/](_hybrid/README.md) |

## Save and load OpenQASM

`emit_openqasm` writes one program as OpenQASM 2 or 3 text, and the stable
`fq.from_openqasm` reads that text back. It accepts only the subset the emitter
writes and refuses everything else with an `OpenQASMImportError` that names the
reason in `issue_code`:

```python
import flagquantum as fq
from flagquantum.compiler.openqasm import emit_openqasm

text = emit_openqasm(fq.Circuit(2).h(0).cx(0, 1))
program = fq.from_openqasm(text)

program.instructions            # the imported gates, in source order
program.measurement_qubits      # the qubit behind each classical bit
fq.run(program.to_circuit(), outputs=fq.counts(), shots=256).counts[0]
```

Import returns the program rather than a `Circuit` because a `Circuit` cannot
carry a terminal measurement; call `to_circuit()` for the sampling path and
`to_ir()` when the measurement mapping must be preserved. It never executes
anything, and it does not read arbitrary third-party OpenQASM.

`openqasm_gates.py` holds the gate spellings both directions read. `openqasm.py`
emits, `openqasm_import.py` parses, and `target_conformance.py` checks an
emission against an expected program; the three share one table, so a spelling
one direction accepts and the other writes cannot appear.

A transformation is acceptable when it preserves the relevant state,
measurement, and gradient references, produces legal output, and has bounded
code growth. An optimization must not remove a trainable gate solely because
its present angle is zero.

Routing strategies are one boundary with several implementations, so a new or
replaced strategy is accepted only when the shared conformance suite in
[test_routing_conformance.py](../../tests/team/compiler/test_routing_conformance.py)
passes over it. That suite also runs an independent router and a replacement
router through the compiler, the topology legalizer, and the dynamic-circuit
runtime, which is how the boundary is shown to be replaceable rather than
merely defined.

The reference lookahead SWAP search Qiskit ships was measured against these
strategies before being rejected as a fifth one. It passes the boundary above:
at search depth and width one it reproduces the shipped planner's plan, and every
program it compiles is legal on the device and numerically equal to its source.
It is rejected on cost. No objective of it beats `sabre_layout`, and the objective
Qiskit documents fails closed on programs where one operation stays stranded
behind the front layer, so it cannot be routed at all on part of the workload.
[benchmarks/compiler_lookahead_swap.py](../../benchmarks/compiler_lookahead_swap.py)
holds the measurement and the commands that reproduce it.

The randomized layer-permutation search Qiskit shipped as `StochasticSwap` was
measured the same way, and rejected on cost as well. Its plan is sound: the
placements it records are the replay of its own SWAPs, and every two-wire
operation it places sits on a device edge. It still retains 1.617 times the SWAPs
`sabre_layout` retains and beats that strategy on none of the 140 measured
programs, and Qiskit's own compiled implementation of the same algorithm retains
0.5% fewer SWAPs on the same basis -- effectively the same count -- so the
shortfall is the algorithm rather than the port. [benchmarks/compiler_stochastic_swap.py](../../benchmarks/compiler_stochastic_swap.py)
holds the measurement and the commands that reproduce it. The checked-in port is
also the only runnable form of the algorithm left to this repository, because
Qiskit 2.0 removed the pass and this repository certifies Qiskit 2.x.

Routing moves two-wire operations onto device edges by inserting SWAPs. No
strategy here synthesizes an operation that touches three or more wires, so such
an operation is carried through unchanged, and only when the device already
carries the couplings its operands interact over; it is refused otherwise.
Decomposing it is a caller or native-gate step.
[test_multi_wire_routing_locality.py](../../tests/team/compiler/test_multi_wire_routing_locality.py)
holds that boundary, including the case where a chosen layout would move a
legal multi-wire operation onto non-adjacent physical wires.

A placement is a tuple of physical wires, one per logical wire: the argument
`route_to_directed_topology` takes as `initial_layout`. [layout.py](layout.py)
`Layout` carries the same assignment as a value, and since a routed program on a
device wider than the program leaves slots idle, `Layout` takes a
`physical_slot_count` and reports `None` for an idle slot. Routing on a plain
`CouplingMap` may only use wires the program owns and so refuses `initial_layout`
outright; a non-identity placement therefore requires a `DirectedCouplingMap`,
where the idle slots are a workspace that the inverse routing SWAPs clean.

[Implementation details](IMPLEMENTATION.md) document stage ownership and
migration constraints. [Testing policy](../../docs/development/TESTING.md)
defines the additional checks for each change.
