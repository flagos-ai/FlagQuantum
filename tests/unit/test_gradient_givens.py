"""Check the source Givens ladder, fractional bindings, and clean histories."""

from cmath import exp
from collections import Counter
from dataclasses import asdict
from itertools import product
from math import tau

import pytest
import torch

from flagquantum.algorithms.primitives.gradient_givens import (
    gradient_givens_cost,
    gradient_givens_on,
    gradient_givens_signature,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import walk_source_graph
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def source_rotation(width, complex_rotation, adjoint):
    from qualtran.bloqs.chemistry.quad_fermion.givens_bloq import (
        ComplexGivensRotationByPhaseGradient,
        RealGivensRotationByPhaseGradient,
    )

    factory = (
        ComplexGivensRotationByPhaseGradient
        if complex_rotation
        else RealGivensRotationByPhaseGradient
    )
    source = factory(width)
    return source.adjoint() if adjoint else source


def bindings(width, complex_rotation, *, permuted=False):
    count = 2 + (4 if complex_rotation else 3) * width - 1
    wires = iter(reversed(range(count)) if permuted else range(count))

    def take(size):
        return tuple(next(wires) for _ in range(size))

    ports = {"target_i": take(1), "target_j": take(1)}
    ports["real_rom_data" if complex_rotation else "rom_data"] = take(width)
    if complex_rotation:
        ports["cplx_rom_data"] = take(width)
    ports["phase_gradient"] = take(width)
    return ports, take(width - 1), tuple(range(40, 40 + width - 1)), count


def bound(ports, carries, slots, adjoint):
    return gradient_givens_on(
        *ports["target_i"],
        *ports["target_j"],
        ports["real_rom_data"] if "real_rom_data" in ports else ports["rom_data"],
        ports["phase_gradient"],
        workspace=carries,
        classical_bits=slots,
        complex_angle=ports.get("cplx_rom_data"),
        adjoint=adjoint,
    )


def clifford_instruction(bloq, inputs):
    from qualtran.bloqs.basic_gates import CNOT, Hadamard, SGate, XGate

    if isinstance(bloq, CNOT):
        return Instruction("cx", (*inputs["ctrl"], *inputs["target"]))
    if isinstance(bloq, Hadamard):
        return Instruction("h", inputs["q"])
    if isinstance(bloq, SGate):
        return Instruction("sdg" if bloq.is_adjoint else "s", inputs["q"])
    if isinstance(bloq, XGate):
        return Instruction("x", inputs["q"])
    return None


@pytest.mark.parametrize("width", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "complex_rotation,adjoint", list(product((False, True), repeat=2))
)
def test_source_givens_ports_gates_and_counts(width, complex_rotation, adjoint):
    pytest.importorskip("qualtran")
    from qualtran import QBit, QFxp
    from qualtran.bloqs.arithmetic.controlled_add_or_subtract import (
        ControlledAddOrSubtract,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = source_rotation(width, complex_rotation, adjoint)
    ports, carries, slots, count = bindings(
        width, complex_rotation, permuted=width >= 4
    )
    signature = gradient_givens_signature(width, complex_rotation=complex_rotation)
    assert [(p.name, p.shape, p.side.value) for p in signature] == [
        (p.name, p.shape, p.side.value) for p in source.signature
    ]
    for native, reference in zip(signature, source.signature, strict=True):
        if isinstance(reference.dtype, QBit):
            assert native.dtype.encoding == "bit"
        else:
            assert reference.dtype == QFxp(width, width, signed=False)
            assert native.dtype.encoding == "fixed_point"
            assert native.bitsize == width
            assert dict(native.dtype.parameters) == {
                "fractional_width": width,
                "signed": False,
            }

    def leaf(bloq, inputs, desired):
        instruction = clifford_instruction(bloq, inputs)
        if instruction is not None:
            return (instruction,), inputs
        if isinstance(bloq, ControlledAddOrSubtract):
            assert bloq.add_when_ctrl_is_on is not adjoint
            return (
                reference_arithmetic_gates(
                    bloq,
                    inputs,
                    carries,
                    dict(zip(carries, slots, strict=True)),
                    preserve_components=True,
                ),
                inputs,
            )
        return None

    expected, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports
    actual = DynamicCircuit(count).append(bound(ports, carries, slots, adjoint)).to_ir()
    assert_source(actual.instructions, expected)
    counts = Counter(g.name for g in actual.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count()
    assert counts["measure"] == counts["reset"] == model.measurement
    native_model = estimate_component_resources(
        gradient_givens_cost(width, complex_rotation=complex_rotation, adjoint=adjoint)
    )
    assert {
        name: value for name, value in asdict(native_model.gates).items() if value
    } == model.asdict()
    assert (
        native_model.peak_logical_qubits
        == get_cost_value(source, QubitCount())
        == count
    )


@pytest.mark.parametrize(
    "complex_rotation,adjoint", list(product((False, True), repeat=2))
)
def test_all_small_measurement_histories_and_complex_amplitudes(
    complex_rotation, adjoint
):
    pytest.importorskip("qualtran")
    import cirq
    from qualtran import Adjoint
    from qualtran.bloqs.rotations.rz_via_phase_gradient import RzViaPhaseGradient

    width = 2
    source = source_rotation(width, complex_rotation, adjoint)
    ports, carries, slots, count = bindings(width, complex_rotation)
    op = bound(ports, carries, slots, adjoint)
    gates = DynamicCircuit(count).append(op).to_ir().instructions
    initial = torch.zeros(1 << count, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    i, j = cirq.LineQubit.range(2)
    real_port = "real_rom_data" if complex_rotation else "rom_data"
    dim = 1 << width

    for real, extra in product(range(dim), range(dim) if complex_rotation else (0,)):
        angle_values = {ports[real_port]: real}
        if complex_rotation:
            angle_values[ports["cplx_rom_data"]] = extra
        ideal = []

        def leaf(bloq, inputs, desired, *, ideal=ideal, angle_values=angle_values):
            instruction = clifford_instruction(bloq, inputs)
            if instruction is not None:
                q = cirq.LineQubit(inputs.get("q", (0,))[0])
                if instruction.name == "cx":
                    ideal.append(cirq.CNOT(*map(cirq.LineQubit, instruction.wires)))
                else:
                    gate = {"h": cirq.H, "s": cirq.S, "sdg": cirq.S**-1, "x": cirq.X}
                    ideal.append(gate[instruction.name](q))
            else:
                base = bloq.subbloq if isinstance(bloq, Adjoint) else bloq
                if not isinstance(base, RzViaPhaseGradient):
                    return None
                sign = -1 if isinstance(bloq, Adjoint) else 1
                radians = sign * 2 * tau * angle_values[inputs["angle"]] / dim
                ideal.append(cirq.rz(radians)(cirq.LineQubit(inputs["q"][0])))
            return (), inputs

        walk_source_graph(source, ports, ports, leaf)
        matrix = torch.tensor(cirq.Circuit(ideal).unitary(qubit_order=[i, j]))
        data = torch.tensor([1 + 2j, 2 - 1j, 3 + 1j, -2 + 3j], dtype=torch.complex128)
        kicked = matrix @ data
        for target, grad in product(range(4), range(dim)):
            prefix = (target << width) | real
            if complex_rotation:
                prefix = (prefix << width) | extra
            index = ((prefix << width) | grad) << len(carries)
            phase = exp(1j * (real - extra - tau * grad / dim))
            initial[index] = data[target] * phase
            expected[index] = kicked[target] * phase

    norm = initial.norm()
    initial /= norm
    expected /= norm
    measurements = sum(g.name == "measure" for g in gates)
    for history in product((0, 1), repeat=measurements):
        torch.testing.assert_close(
            forced_branch(initial, gates, history),
            expected * 2 ** (-measurements / 2),
            atol=2e-12,
            rtol=0,
        )


def test_givens_rejects_aliases_and_incomplete_work():
    ports, carries, slots, _ = bindings(2, True)
    with pytest.raises(ValueError, match="n-1 work slots"):
        bound(ports, carries, (), False)
    ports["cplx_rom_data"] = ports["real_rom_data"]
    with pytest.raises(ValueError, match="distinct nonnegative wires"):
        bound(ports, carries, slots, False)
