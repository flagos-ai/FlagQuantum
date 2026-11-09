"""Bounded Pauli SELECT source gates, phases, resource counts and padding."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.algorithms.primitives.multiplexers.select_pauli_lcu import (
    SelectPauliLCU,
)
from flagquantum.algorithms.primitives.select_block_encoding import SelectBlockEncoding
from flagquantum.compiler.registers.lookup.unary_iteration import (
    unary_iteration_workspace,
)
from flagquantum.compiler.select_lowering import unary_select_ir
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import (
    assert_controlled_exchange_branches,
    assert_source_dependencies,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("words", "width", "value", "reversed_wires"),
    [
        (("X", "Z", "X"), 2, None, False),
        (("-YX", "ZI", "II"), 2, 0, False),
        (("X", "-I", "Y", "Z"), 2, 1, False),
        (("-Y",), 1, 1, False),
        (("X", "Y", "Z"), 4, None, True),
        (("-X", "I", "Y", "X", "Z"), 4, 0, True),
    ],
)
@pytest.mark.ftqc_reference
def test_bounded_pauli_source_gates_and_cost(words, width, value, reversed_wires):
    pytest.importorskip("qualtran")
    import cirq
    from qualtran.bloqs.multiplexers.select_pauli_lcu import (
        SelectPauliLCU as SourceSelect,
    )

    target_width = len(words[0].removeprefix("-"))
    controls = int(value is not None)
    data = controls + width + target_width
    work = unary_iteration_workspace(width, value)
    total = data + work
    wires = tuple(reversed(range(total)) if reversed_wires else range(total))
    ctrl, address, target = (
        wires[:controls],
        wires[controls : controls + width],
        wires[controls + width : data],
    )
    workspace = wires[data:]
    slots = tuple(range(20, 20 + work))
    op = SelectPauliLCU(width, target_width, words, value).make_bounded_on(
        selection=address,
        target=target,
        control=ctrl,
        workspace=workspace,
        classical_bits=slots,
    )
    circuit = DynamicCircuit(total).append(op)
    actual = circuit.to_ir().instructions

    source = SourceSelect(
        width,
        target_width,
        tuple(
            cirq.DensePauliString(
                word.removeprefix("-"), coefficient=-1 if word.startswith("-") else 1
            )
            for word in words
        ),
        value,
    )
    ports = {"selection": address, "target": target}
    if ctrl:
        ports["control"] = ctrl
    expected = reference_arithmetic_gates(
        source,
        ports,
        workspace,
        dict(zip(workspace, slots, strict=True)),
        cirq_style=True,
        keep_pauli_leaves=True,
    )
    assert_source_dependencies(actual, expected)
    assert_source_model(op.resource_model(), source)
    planned = estimate_ftqc_resources(op)
    report = estimate_ftqc_resources(op, emitted_ir=circuit.to_ir())
    assert planned.emitted is None
    assert report.logical == planned.logical
    assert report.emitted is not None
    assert report.emitted.t_count == report.logical.gates.total_t_count()
    assert (
        Counter(gate.name for gate in actual)["measure"]
        == report.logical.gates.measurement
    )
    assert report.emitted.declared_qubits == total
    if not reversed_wires:
        assert_controlled_exchange_branches(actual, expected, actual, data, workspace)


def test_explicit_bounded_padding_and_default_identity():
    select = SelectPauliLCU(2, 1, ("X", "Z", "X"))
    state = torch.zeros(16, dtype=torch.complex128)
    state[12] = 1j  # address 11, target zero and clean work
    bounded = DynamicCircuit(4, inputs=state, dtype=torch.complex128).append(
        select.make_bounded_on(
            selection=(0, 1), target=(2,), workspace=(3,), classical_bits=(0,)
        )
    )
    expected = torch.zeros_like(state)
    expected[14] = 1j
    result = run_dynamic(bounded, shots=4, seed=7, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states, expected.expand(4, -1), atol=2e-12, rtol=0
    )

    default = DynamicCircuit(4, inputs=state, dtype=torch.complex128).append(
        select.make_on(selection=(0, 1), target=(2,))
    )
    (logical,) = default.to_ir().instructions
    gates = unary_select_ir(logical).instructions
    for history in product((0, 1), repeat=2):
        torch.testing.assert_close(
            forced_branch(state, gates, history), state / 2, atol=2e-12, rtol=0
        )


def test_bounded_pauli_validates_workspace_and_source_boundaries():
    select = SelectPauliLCU(2, 1, ("X", "Z", "X"))
    with pytest.raises(ValueError, match="clean work"):
        select.make_bounded_on(
            selection=(0, 1), target=(2,), workspace=(), classical_bits=()
        ).emit()
    with pytest.raises(ValueError, match="classical slots"):
        select.make_bounded_on(
            selection=(0, 1), target=(2,), workspace=(3,), classical_bits=()
        ).emit()
    with pytest.raises(ValueError, match="two rows"):
        SelectPauliLCU(1, 1, ("X",)).make_bounded_on(
            selection=(0,), target=(1,), workspace=(), classical_bits=()
        ).emit()


def test_select_block_can_consume_bounded_pauli():
    select = SelectPauliLCU(2, 1, ("X", "Z", "X"))
    address, target, work, bits = (0, 1), (2,), (3,), (0,)

    def emit(ports, workspace, classical_bits, _adjoint):
        return select.make_bounded_on(
            selection=ports["address"],
            target=ports["target"],
            workspace=workspace,
            classical_bits=classical_bits,
        ).emit()

    def prepare(*_):
        pytest.fail("SELECT block must leave signal preparation separate")

    signal = BlackBoxPrepare((("address", 2),), (), prepare, normalization=3.0)
    encoding = SelectBlockEncoding((("address", 2),), (("target", 1),), emit, signal)
    circuit = DynamicCircuit(4).append(
        encoding.make_on(
            selection=address, target=target, workspace=work, classical_bits=bits
        )
    )
    op = select.make_bounded_on(
        selection=address, target=target, workspace=work, classical_bits=bits
    )
    assert circuit.to_ir().instructions == op.emit()
    report = estimate_ftqc_resources(
        encoding.resource_model(op.resource_model()), emitted_ir=circuit.to_ir()
    )
    assert report.emitted is not None and report.emitted.t_count == 4
