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
| Cancellation of a declared inverse pair | [inverse_cancellation.py](inverse_cancellation.py) |
| Connectivity and routing | [routing.py](routing.py), [sabre.py](sabre.py), [topology_legalization.py](topology_legalization.py) |
| Wire layouts and the layout restore | [layout.py](layout.py) |
| Initial placement on a device | [layout_planning.py](layout_planning.py) |
| Native-gate and target requirements | [native_gate_legalization.py](native_gate_legalization.py), [target_legalization.py](target_legalization.py) |
| Named-gate identities and the basis search | [basis_translation.py](basis_translation.py) |
| One-qubit Euler angles | [one_qubit_synthesis.py](one_qubit_synthesis.py) |
| Two-qubit KAK angles and entangler cost | [two_qubit_synthesis.py](two_qubit_synthesis.py) |
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
one direction accepts and the other writes cannot appear. `tools/check_openqasm_import_contract.py`
reads [openqasm-import-v1-candidate.json](../../contracts/openqasm-import-v1-candidate.json)
against the shipped importer: the refusal vocabulary, the version lanes, the
exposed signature, the root additions, and a reader for every declared rule.

A transformation is acceptable when it preserves the relevant state,
measurement, and gradient references, produces legal output, and has bounded
code growth. An optimization must not remove a trainable gate solely because
its present angle is zero.

A gate that is its own inverse is one thing; a gate whose inverse is a *different*
opcode is another, and `merge_self_inverse` only saw the first kind, so it left
`s(0) sdg(0)` standing as two instructions although the pair is the identity.
[inverse_cancellation.py](inverse_cancellation.py) closes that with
`merge_inverse_pairs`, and it holds no opcode table of its own: it reads the
`OperatorSchema.adjoint` declaration through `operator_schema.inverse_operator`,
the same declaration `Circuit.adjoint` already consumes. That keeps the inverse
relation to one source of truth, and
`tests/unit/test_compilation_inverse_cancellation.py` enforces it structurally --
the module contains no opcode spelled as a code string and no module-level table.

The declared rule is a rule of proof rather than of record: every declared unitary
was multiplied by its declared inverse on the runtime's own matrices over seeded
draws, and the worst residual is 2.22e-16 against a smallest wrong-partner
residual of 1.22 -- a control, because a residual of zero would mean nothing if
the same comparison could not produce a nonzero one.

The pass removes a pair only when the gap between the two members is empty on
their own wire, and that boundary is measured rather than asserted:
[benchmarks/compiler_inverse_cancellation.py](../../benchmarks/compiler_inverse_cancellation.py)
drives every pair through ten gaps and reports, per row, whether the gap's
operator commutes with the pair and whether the pair was in fact removable --
measured against whole-register operators, since several of these gates act
trivially on `|0...0>` and a state comparison would call a row redundant for a
reason unrelated to the shape. Of the 60 rows, 6 gaps are empty and 12
instructions are removed there, 32 gaps commute and 24 instructions are removed
across the sub-case where the gap misses the pair's wire entirely, 22 gaps do not
commute and are never crossed, and 20 rows are declined although the pair was
removable: reach left on the table, reported as a number rather than as a promise.
A refusal is therefore either correctness or deferred reach, never a single total.
The module also drives Qiskit's `InverseCancellation` over the identical circuits
and finds the two agreeing on every row.

A *named* gate has no matrix on this layer, so it can only leave a program for
a basis that does not carry it through a closed identity.
[basis_translation.py](basis_translation.py) is that table, plus the
deterministic search that composes it, and it is the Compiler-layer counterpart
of the equivalence library and basis search Qiskit ships. Each rule is an exact
closed form that forwards the source instruction's own parameter objects and
metadata, so a trainable angle stays in the autograd graph; no rule introduces an
angle the caller did not write; and the search branches on opcode names only,
never on a value. Seventeen of the eighteen entries reproduce their source
exactly and `cphase` is equal to it up to one global phase, which FlagQuantum IR
has no field to record. The table stores the shortest statement of each identity
rather than its closure, so the search expands each rule's leaves through the
table again: a `cz` basis plus a z-rotation reaches all eleven declared two-qubit
opcodes by name, where the four hand-written rules this replaced reached two of
them. The two three-wire entries cover the whole of this IR's multi-controlled
surface -- `ccx` as its fifteen-gate standard form and `cswap` as a `ccx` around
two `cx` -- and they reach the three bases that publish a `cx` or a `cz` sink.
Qiskit splits the same operation into a Gray-code, a recursive and a V-chain
construction, but at two controls the first of those is `CCXGate` itself and the
other two exist to trade ancillas for fewer entanglers at five controls and up,
which this IR cannot express. The Gray-code *statement* is still measured against
the taken one, and declined: seven declared leaves against fifteen, but eight
two-qubit gates against six.
`ion-trap-rz-rx-rzz` is the recorded gap -- every entangling rule routes down to
`cx` or `cz`, and that basis publishes neither.
[benchmarks/compiler_basis_translation.py](../../benchmarks/compiler_basis_translation.py)
holds the per-rule fidelity, the per-basis reach, and the difference from Qiskit's
`BasisTranslator` on the same target bases.

Native-gate legalization rewrites one-qubit instructions by Euler synthesis
whenever the target publishes a z-rotation and a pi/2 x-rotation: every declared
single-qubit unitary becomes z-rotations plus `sx`, or plus `rx` at a fixed
`pi/2` when the basis publishes `rx` instead. An exact identity is tried first, so
a basis that can carry a gate exactly keeps its exact form. Such a basis cannot
carry the global phase of `h`, `s`, or `t`, and FlagQuantum IR has no field for
it, so those rewrites are equal up to one global phase -- the same contract every
`U3`-based hardware basis publishes. A consumer that compares raw statevectors
instead of measurement statistics has to know that; flag records of it belong in
the capability registry, which `capability-maturity.toml` owns.

Two-qubit KAK synthesis extends that to a matrix-carrying instruction on two
wires, over any supercontrolled entangler the basis publishes. Six declared
arity-2 opcodes reach a supercontrolled Weyl point: `cx`, `cz` and `cy` as they
stand, and `rzz`, `ryy` and `rxx` at an angle of `pi/2`. `cphase` is excluded on
purpose -- its only supercontrolled angle is `pi`, where it is `cz`, so it would
add a spelling rather than reach. Carrying the interaction rotations is what
opens a trapped-ion or flux-tunable-coupler basis, whose only two-qubit gate is a
rotation and which publishes no `cx` at all; such a basis now reaches all eleven
declared two-qubit unitaries where it previously reached one. The cost is one
entangler for `cx`, `cz`, and `cy`, two for the controllized rotations, and three
for `swap`, which is the far corner of the Weyl chamber; all eleven declared
two-qubit unitaries reproduce their source to within `1.1e-15` over any of the six
entanglers, up to one global phase.
[benchmarks/compiler_two_qubit_synthesis.py](../../benchmarks/compiler_two_qubit_synthesis.py)
holds the reach, cost, and phase measurement, and cross-checks the entangler
count against Qiskit's `TwoQubitBasisDecomposer` over the whole table.

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
