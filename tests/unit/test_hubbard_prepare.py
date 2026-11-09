"""Validate Hubbard PREPARE against the unmodified source decomposition."""

from dataclasses import asdict
from itertools import product

import numpy as np
import pytest

from flagquantum.algorithms.primitives.hubbard_prepare import PrepareHubbard
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.registers.arithmetic.bounded_increment import (
    bounded_increment_ir,
    bounded_increment_requirements,
)
from flagquantum.compiler.registers.control.multi_control import multi_and_ir
from flagquantum.compiler.registers.layout.partition import partition_register
from flagquantum.compiler.registers.preparation.uniform_preparation import (
    uniform_preparation_ir,
    uniform_preparation_requirements,
)
from flagquantum.compiler.registers.primitives.controlled_swap import controlled_swap_ir
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("dim", [2, 3, 4, 5, 8, 9])
def test_locked_hubbard_prepare_signature_and_resources(dim):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.prepare_hubbard import (
        PrepareHubbard as Source,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = Source(dim, dim, t=2.0, u=7.25)
    prepare = PrepareHubbard(dim, t=2.0, u=7.25)
    assert prepare.layout == tuple(
        (reg.name, reg.total_bits()) for reg in source.signature
    )
    assert prepare.alpha == source.l1_norm_of_coeffs
    result = estimate_component_resources(prepare.source_resource_model())
    assert {key: value for key, value in asdict(result.gates).items() if value} == (
        get_cost_value(source, QECGatesCost()).asdict()
    )
    assert result.peak_logical_qubits == get_cost_value(source, QubitCount())


@pytest.mark.parametrize("dim", [2, 3])
def test_hubbard_prepare_complex_amplitudes_and_roundtrip(dim):
    cirq = pytest.importorskip("cirq")
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import CSwap
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.prepare_hubbard import (
        PrepareHubbard as Source,
    )
    from qualtran.bloqs.mcmt.and_bloq import MultiAnd
    from qualtran.bloqs.mod_arithmetic import ModAddK
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )

    prepare = PrepareHubbard(dim, t=2, u=8)
    source = Source(dim, dim, t=2, u=8)
    logical = sum(size for _, size in prepare.layout)
    wires = cirq.LineQubit.range(logical)
    context = cirq.DecompositionContext(cirq.GreedyQubitManager(prefix="reference"))
    source_ops = cirq.decompose_once(source.on(*wires), context=context)

    def semantic_op(op):
        gate = op.gate
        if not isinstance(gate, ModAddK):
            return op
        # The only undecomposable source leaf exposes an exact permutation.
        words = 1 << gate.bitsize
        matrix = np.zeros((4 * words, 4 * words))
        for ctrl, x in product(range(4), range(words)):
            out = gate.on_classical_vals(ctrl=ctrl, x=x)
            matrix[ctrl * words + out["x"], ctrl * words + x] = 1
        return cirq.MatrixGate(matrix).on(*op.qubits)

    source_circuit = cirq.Circuit(
        cirq.decompose((semantic_op(op) for op in source_ops), context=context)
    )
    source_work = sorted(source_circuit.all_qubits() - set(wires))
    result = (
        cirq.Simulator(dtype=np.complex128, seed=17)
        .simulate(source_circuit, qubit_order=(*wires, *source_work))
        .final_state_vector
    )
    source_rows = result.reshape(1 << logical, -1)
    np.testing.assert_allclose(source_rows[:, 1:], 0, atol=3e-12, rtol=0)

    work, _ = prepare.requirements()
    width = logical + work
    ports = partition_register(tuple(range(logical)), prepare.layout)
    workspace = tuple(range(logical, width))
    forward = prepare.make_on(ports, workspace=workspace)
    backward = prepare.make_on(ports, workspace=workspace, adjoint=True)
    ir = DynamicCircuit(width).append(forward).to_ir()
    inverse = DynamicCircuit(width).append(backward).to_ir()
    counts = count_emitted_instructions(ir)
    events = counts.unconditional["measure"]
    assert (counts.t_count, events, work, counts.declared_qubits) == {
        2: (47, 3, 2, 12),
        3: (155, 23, 6, 20),
    }[dim]
    # Bind the original top-level operands to the same reusable native work.
    source_map = dict(zip(wires, range(logical), strict=True))
    source_map.update(zip(source_ops[-3].qubits[-2:], workspace[:2], strict=True))
    bits = tuple(range(prepare.requirements()[1]))
    expected_stream = []
    for op in source_ops:
        gate = op.gate
        operands = tuple(source_map[wire] for wire in op.qubits)
        if isinstance(gate, PrepareUniformSuperposition):
            _, nwork, nbits = uniform_preparation_requirements(dim)
            expected_stream.extend(
                uniform_preparation_ir(dim, operands, workspace[:nwork], bits[:nbits])
            )
        elif isinstance(gate, ModAddK):
            nwork, nbits = bounded_increment_requirements(
                len(operands) - 2, gate.mod, 2
            )
            expected_stream.extend(
                bounded_increment_ir(
                    operands[2:],
                    gate.mod,
                    workspace[:nwork],
                    bits[:nbits],
                    controls=operands[:2],
                    control_values=gate.cvs,
                )
            )
        elif isinstance(gate, CSwap):
            split = (len(operands) + 1) // 2
            expected_stream.extend(
                controlled_swap_ir(operands[0], operands[1:split], operands[split:])
            )
        elif isinstance(and_gate := getattr(gate, "bloq", None), MultiAnd):
            expected_stream.extend(
                multi_and_ir(operands[:3], and_gate.cvs, operands[3:])
            )
        elif isinstance(and_gate := getattr(gate, "subbloq", None), MultiAnd):
            expected_stream.extend(
                multi_and_ir(
                    operands[:3],
                    and_gate.cvs,
                    operands[3:],
                    classical_bits=bits[:2],
                    uncompute=True,
                )
            )
        elif isinstance(gate, cirq.Ry):
            expected_stream.append(
                Instruction("ry", operands, params={"theta": gate.exponent * np.pi})
            )
        elif isinstance(gate, cirq.ControlledGate) and isinstance(
            gate.sub_gate, cirq.Ry
        ):
            expected_stream.append(
                Instruction(
                    "cry", operands, params={"theta": gate.sub_gate.exponent * np.pi}
                )
            )
        else:
            name = {
                cirq.H: "h",
                cirq.X: "x",
                cirq.CNOT: "cx",
                cirq.H.controlled(): "ch",
            }[gate]
            expected_stream.append(Instruction(name, operands))
    assert len(ir.instructions) == len(expected_stream)
    for emitted, reference in zip(ir.instructions, expected_stream, strict=True):
        assert (emitted.name, emitted.wires, emitted.metadata) == (
            reference.name,
            reference.wires,
            reference.metadata,
        )
        assert emitted.params == pytest.approx(reference.params)
    expected = {
        index << work: value
        for index, value in enumerate(source_rows[:, 0])
        if abs(value) > 1e-12
    }
    histories = (
        product((0, 1), repeat=events)
        if dim == 2
        else ((pattern * events)[:events] for pattern in ((0,), (1,), (0, 1), (1, 0)))
    )
    for history in histories:
        result = _branch(ir.instructions, width, history, normalize_measurements=True)
        assert result == pytest.approx(expected, abs=6e-12)
        opposite = tuple(1 - bit for bit in history)
        assert _branch(
            inverse.instructions,
            width,
            opposite,
            result,
            normalize_measurements=True,
        ) == pytest.approx({0: 1 + 0j}, abs=6e-12)
