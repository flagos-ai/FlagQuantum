# Compiler

This package transforms a Core-owned `CircuitIR` without executing it.
`pipeline.py` owns the stable optimization, layer scheduling, and topology-aware
compilation entry points. `routing.py` owns coupling maps and SWAP
routing, including the bounded all-pairs hop-count index that
`CouplingMap.distance` and `CouplingMap.distance_matrix` expose.
`sabre.py` owns the lookahead SWAP planner that `routing.py` exposes as the
`sabre` and `sabre_layout` routing strategies: it chooses one persistent layout
for the whole program, and `routing.py` materializes that plan into Core
`CircuitIR`. `sabre_layout` additionally searches for the initial layout, using
the SABRE layout pass of routing the reversed program, and therefore restores the
output layout with an explicit SWAP sequence instead of replaying the forward
SWAPs in reverse.
The planner swaps only between wires the program owns, so it plans on the
coupling subgraph those wires induce rather than on the whole device. A device
wider than the program is therefore supported exactly as far as the program's own
wires connect it, and a program that the device connects only through a padding
wire is refused rather than costed against a route the plan cannot emit. Routing
through a physical ancilla remains the unsupported case it is for the
shortest-path strategies.
The beam search Qiskit ships as `LookaheadSwap` was ported and measured against
these strategies before being rejected as a fifth one. It is a faithful
replacement at search depth and width one, but no objective of it beats
`sabre_layout`, and the objective Qiskit documents fails closed on programs where
one operation stays stranded. `benchmarks/compiler_lookahead_swap.py` holds that
measurement, so the search does not need to be rebuilt to re-test the decision.
The randomized layer-permutation search Qiskit shipped as `StochasticSwap` and
removed in 2.0 was ported and measured the same way, and rejected on cost too. It
is a correct planner -- the placements it records are the replay of its own SWAPs,
and every two-wire operation it places sits on a device edge -- but it retains
1.617 times the SWAPs `sabre_layout` retains and beats that strategy on none of the
140 measured programs. Raising the trial count does not close the gap, and
Qiskit's own compiled implementation of the same algorithm retains 0.5% fewer
SWAPs on the same basis -- effectively the same count -- so the shortfall belongs
to the algorithm rather than to the port. `benchmarks/compiler_stochastic_swap.py` holds that measurement, including
the independent replay of every plan it produces.
`layout.py` owns the logical-to-physical `Layout` value and the two
transformations over it: applying a layout to a program by relabelling its wires,
and removing the trailing restore SWAPs a routed program ends with. A routing
strategy always returns a program that ends on the identity layout, because
target legalization and the deployment routing evidence both require that
postcondition; removing the restore is therefore an explicit opt-in
transformation over an already routed program, not a routing strategy and not a
routing result to hand back to legalization.
`openqasm.py` and `qcis.py` own their target-format emission.
`noise.py` owns the deterministic `CircuitIR + NoiseModel` to channel-bearing
`CircuitIR` transformation. `operator_lowering.py` owns the
internal backend/operator capability registry used before lowering or
serialization. `native_gate_legalization.py` validates evidenced native-gate
descriptors and applies the bounded, verified CircuitIR decompositions.
`one_qubit_synthesis.py` owns the one-qubit Euler angles behind those
decompositions: it turns any declared single-qubit unitary into z-rotations plus
a pi/2 x-rotation, `sx` or `rx`, and it is a private helper rather than an
expert-facing entry point.
`two_qubit_synthesis.py` owns the two-qubit KAK angles, the entangler table, and
the entangler cost behind the same decompositions: it turns a 4x4 unitary into a
supercontrolled entangler repeated one to three times, with one one-qubit factor
between each, and it too is a private helper. The table holds every declared
arity-2 opcode that reaches a supercontrolled Weyl point together with the angle
it has to be applied at, so a target whose only two-qubit gate is an interaction
rotation is synthesizable. It takes a matrix rather than an instruction, because
the matrix of a named two-qubit gate belongs to `flagquantum.simulation`, which
this layer must not import. The three parameter-free entangler matrices and the
closed form of the rotation family are the only gate matrices it holds, and
`tests/unit/test_compilation_two_qubit_synthesis.py` pins them entry-by-entry
against that table.
`topology_legalization.py` applies the existing router to one explicit coupling
map and verifies edge legality, restored output layout, bounded growth, and
deterministic evidence.
`schedule_legalization.py` constructs deterministic logical ASAP layers with
explicit wire and classical-data dependencies. Dynamic operations and channels
remain conservative barriers; this is not target timing or pulse scheduling.
`target_emission.py` gates the existing OpenQASM and QCIS text emitters behind
completed target legalization and binds deterministic emission audit facts. Its
result is not a new executable-artifact envelope.
`target_conformance.py` independently parses that closed emitted subset back
into Core `CircuitIR` for structural and test-time numerical conformance; it
does not execute programs or certify hardware behavior.
`target_legalization.py` derives mandatory circuit requirements, checks one
explicit backend lowering, and matches one Core-owned target capability
snapshot without introducing a target IR or selecting a target.
`__init__.py` is the stable
expert-facing compiler interface.

`_hybrid/` owns the private structured-program semantic slice used to migrate
program-level control flow and ordered quantum effects into vNext. It is not a
second circuit compiler: its future quantum-region output is the existing
Core-owned `CircuitIR`, and it is intentionally absent from public exports.

Use `optimize(program)` for target-independent canonical optimization and
`compile(program, coupling_map=...)` for target-aware lowering. The pre-release name
`simple_compile` has been removed; it did not describe a distinct compilation
stage.

Run the user-facing optimization path from the repository root:

```bash
python -m examples.compiler_optimize
python -m examples.target_aware_compilation
```

Individual canonicalization functions are pipeline implementation details, not
expert-facing entry points. Change or compose them through `optimize`.

## Ten-minute change path

- Change local canonical optimization in `pipeline.py`.
- Change instruction layer scheduling in `pipeline.py`.
- Change coupling maps or SWAP routing in `routing.py`.
- Change lookahead SWAP planning or the SABRE layout search in `sabre.py`.
- Change noise-model lowering in `noise.py`.
- Change OpenQASM 2/3 target emission in `openqasm.py`.
- Change QCIS target emission in `qcis.py`.
- Change operator/backend lowering capabilities in `operator_lowering.py`.
- Change native gate matching and verified decompositions in
  `native_gate_legalization.py`.
- Change one-qubit Euler angles or the z-rotation plus pi/2 pulse leaf form in
  `one_qubit_synthesis.py`.
- Change two-qubit KAK angles, the Weyl-chamber fold, or the entangler cost in
  `two_qubit_synthesis.py`.
- Change topology postconditions and routing audit in
  `topology_legalization.py`.
- Change dependency-preserving logical scheduling and its audit in
  `schedule_legalization.py`.
- Change verified static text emission and its identity binding in
  `target_emission.py`.
- Change strict emitted-text parsing and conformance evidence in
  `target_conformance.py`.
- Change capability-driven legality checks in `target_legalization.py`.
- Change private structured program semantics through `_hybrid/README.md` and
  its focused golden scenario; do not restore the historical `_compiler` tree.
- Run the compiler fixed-point, trainable-parameter, scheduler, routing, public
  namespace, noise, and CPU vertical-slice tests.

Runtime planning, backend selection, resource estimation, execution-plan
assembly, noise execution policy, provider lifecycle, and simulation numerics
do not belong here.
