# Runtime statevector numerical ownership

This implemented batch is based on CI-qualified PR1333 head
`ac2eee3c4ae85d3014bd98f76a14632f0df1b3ae`. It relocates three private
numerical functions and shares the forward/reverse tensor-product fold.

| Previous Runtime definition | Simulation owner |
| --- | --- |
| `dynamic/execution.py::_measure_wire` | `statevector/measurement_ops.py` |
| `dynamic/execution.py::_measure_wires_batched` | `statevector/measurement_ops.py` |
| `dynamic/execution.py::_apply_instruction` | `statevector/instruction.py` |

Simulation owns collapse probabilities, draws through the supplied generator,
normalization, tensor application and matrix dispatch. Runtime calls those
module owners directly and keeps generator creation, seeds, validation,
feedback, masks, shot records, rank ownership and evidence assembly. The old
three private attributes and definitions are removed without forwarding.

Instruction application uses existing operations primitives. Its separate
internal module keeps the existing operations module within its enforced
1250-line ceiling and introduces no package, public concept or contract type.
Forward/reverse local block composition shares a private numerical fold in the
existing `simulation/gate_matrix.py`. Grouping, admissibility, memory bounds
and scratch accounting remain with the Runtime sweeps. Operand order,
allocation behavior, precision, autograd and random policy are unchanged.

The three moved functions retain their ASTs after gate-matrix import-binding
normalization. All 17 remaining Runtime definitions retain ASTs and docstrings
after direct-owner call normalization; all pre-existing target definitions
remain unchanged. The matrix-bearing/plain-Z dispatch test patches the real
Simulation instruction owner and preserves its existing assertions.

The incoming source file changes in PR1084, but these three function ASTs are
unchanged across its own base-to-head diff and equal to the parent definitions.
The activation audit reaches `e7259749b2686482ca33ba2d0d6049e913cb024e`:
3221 changed incoming Python trees have no imports of these three retired
private members, and the targets have no definition collisions. DynamicCircuit
construction, classical IR and evolving component execution contracts remain
within the separate gated joint migration.

Use the existing `tests/test_dynamic_circuit.py` and
`tests/test_dynamic_conformance.py` workflows to change or test trajectory
behavior. `tests/unit/test_statevector_reverse.py` and
`tests/integration/test_ising_fusion_measurement.py` cover the paired fold.
The [acceptance record](runtime-state-kernel-acceptance.md) records formal
regressions, installed artifacts and the remaining pushed-head CI gate.
