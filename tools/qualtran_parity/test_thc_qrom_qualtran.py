"""Compare the THC register decompositions against Qualtran 0.7.0."""

from importlib.metadata import version

import pytest

from flagquantum.compiler.registers.lookup.qrom import controlled_qrom_ir
from flagquantum.core.ir import CircuitIR
from tests.unit.algorithms.primitives.multiplexers.test_thc_qrom import _execute_sparse

np = pytest.importorskip("numpy")


def test_qualtran_v070_component_decompositions() -> None:
    from itertools import product

    assert version("qualtran") == "0.7.0"
    cirq = pytest.importorskip("cirq")
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.controlled_add_or_subtract import (
        ControlledAddOrSubtract,
    )
    from qualtran.bloqs.data_loading import QROM, QROAMClean
    from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState

    from flagquantum.compiler.registers.arithmetic.addition import (
        controlled_add_or_subtract_ir,
    )
    from flagquantum.compiler.registers.phase.phase_gradient import (
        phase_gradient_state_ir,
    )

    words = (6, 1, 5, 0)
    for blocked in (False, True):
        options = {"log_block_sizes": (1,)} if blocked else {}
        upstream = (
            (QROAMClean if blocked else QROM)
            .build_from_data(
                np.asarray(words), num_controls=1, target_bitsizes=(3,), **options
            )
            .decompose_bloq()
        )
        ir = controlled_qrom_ir(2, 3, words, blocked=blocked)
        for enable, address in product(range(2), range(4)):
            kwargs = {"control": enable, "selection": address}
            if not blocked:
                kwargs["target0_"] = 0
            reference = upstream.on_classical_vals(**kwargs)
            result = _execute_sparse(
                ir, {(enable * 4 + address) << (ir.n_wires - 3): 1}, seed=0
            )
            word = int(reference["target0_"])
            if blocked:
                word = (word << 3) | int(reference["junk_target0_"][0])
            assert result == {
                ((enable * 4 + address) << (ir.n_wires - 3)) | (word << 2): 1
            }

    add = ControlledAddOrSubtract(QUInt(3), QUInt(3), add_when_ctrl_is_on=False)
    upstream_add = add.decompose_bloq()
    ir = CircuitIR(8, controlled_add_or_subtract_ir(0, (1, 2, 3), (4, 5, 6), 7))
    for control, a, b in product(range(2), range(8), range(8)):
        reference = upstream_add.on_classical_vals(ctrl=control, a=a, b=b)
        result = _execute_sparse(ir, {(control << 7) | (a << 4) | (b << 1): 1}, seed=0)
        assert result == {(control << 7) | (a << 4) | (int(reference["b"]) << 1): 1}

    wires = tuple(cirq.LineQubit.range(8))
    phase_state = PhaseGradientState(8, exponent=1).on_registers(phase_grad=wires)
    expected = (
        cirq.Simulator(dtype=np.complex128)
        .simulate(cirq.Circuit(cirq.decompose(phase_state)), qubit_order=wires)
        .final_state_vector
    )
    ir = CircuitIR(8, phase_gradient_state_ir(tuple(range(8))))
    result = _execute_sparse(ir, {0: 1}, seed=0)
    np.testing.assert_allclose(
        [result[i] for i in range(256)], expected, atol=1e-12, rtol=0
    )


def test_qualtran_add_gate_sequence_and_complex_amplitudes() -> None:
    assert version("qualtran") == "0.7.0"
    cirq = pytest.importorskip("cirq")
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.mcmt.and_bloq import And

    from flagquantum.compiler.registers.arithmetic.addition import add_ir

    for width in (1, 3, 8):
        wires = cirq.LineQubit.range(2 * width)
        context = cirq.DecompositionContext(cirq.SimpleQubitManager())
        upstream = Add(QUInt(width)).decompose_from_registers(
            context=context, a=np.asarray(wires[:width]), b=np.asarray(wires[width:])
        )
        decomposed = []
        for gate in cirq.flatten_op_tree(upstream):
            if isinstance(gate.gate, And):
                decomposed.extend(
                    cirq.flatten_op_tree(
                        gate.gate.decompose_from_registers(
                            context=context,
                            ctrl=np.asarray(gate.qubits[:2]).reshape(2, 1),
                            target=np.asarray(gate.qubits[2:]),
                        )
                    )
                )
            else:
                decomposed.append(cirq.CNOT.on(*gate.qubits))

        labels = {wire: i for i, wire in enumerate(wires)}
        for gate in decomposed:
            for wire in gate.qubits:
                labels.setdefault(wire, len(labels))
        gate_names = {
            cirq.H: "h",
            cirq.T: "t",
            cirq.T**-1: "tdg",
            cirq.S: "s",
            cirq.CNOT: "cx",
            cirq.CZ: "cz",
        }
        measured = {}
        expected = []
        for gate in decomposed:
            metadata = {}
            if isinstance(gate, cirq.ClassicallyControlledOperation):
                (key,) = cirq.control_keys(gate)
                metadata["conditions"] = ((measured[key], 1),)
                gate = gate.without_classical_controls()
            operands = tuple(labels[wire] for wire in gate.qubits)
            if cirq.is_measurement(gate):
                key = cirq.measurement_key_obj(gate)
                measured[key] = operands[0]
                metadata = {"is_dynamic": True, "classical_bit": operands[0]}
                name = "measure"
            elif isinstance(gate.gate, cirq.ResetChannel):
                name = "reset"
                metadata = {"is_dynamic": True}
            else:
                name = gate_names[gate.gate]
            expected.append((name, operands, metadata))

        carries = tuple(range(2 * width, 3 * width - 1))
        actual = add_ir(
            tuple(range(width)), tuple(range(width, 2 * width)), carries, carries
        )
        assert [
            (gate.name, gate.wires, dict(gate.metadata)) for gate in actual
        ] == expected
        assert sum(gate.name in ("t", "tdg") for gate in actual) == 4 * (width - 1)
