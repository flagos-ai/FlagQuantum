"""Exact source composition and complex measured rotation preparation."""

from cmath import exp
from collections import Counter
from itertools import product
from math import tau
from typing import Any

import pytest
import torch

from flagquantum.algorithms.primitives.rotation_preparation import RotationPreparation
from flagquantum.core.ir import CircuitIR, Instruction
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_phase_gradient_lookup import source_lookup_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "coefficients,width,control_count",
    [
        ((0.5 - 0.5j, 0.5 + 0.5j), 2, 0),
        ((0.5 - 0.5j, 0.5 + 0.5j), 2, 1),
        ((0.5, 0.5j, -0.5, -0.5j), 2, 2),
        ((0.5, 0.5j, -0.5, -0.5j), 8, 0),
    ],
)
@pytest.mark.parametrize("uncompute", [False, True])
def test_rotation_preparation_graph_and_state(
    coefficients: tuple[complex, ...], width: int, control_count: int, uncompute: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import Rx, XGate
    from qualtran.bloqs.bookkeeping import Allocate, Free
    from qualtran.bloqs.state_preparation.state_preparation_via_rotation import (
        PRGAViaPhaseGradient,
        StatePreparationViaRotations,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = StatePreparationViaRotations(
        coefficients, width, control_bitsize=control_count, uncompute=uncompute
    )
    preparation = RotationPreparation(coefficients, width)
    target_size = (len(coefficients) - 1).bit_length()
    controls = tuple(range(control_count))
    target = tuple(range(control_count, control_count + target_size))
    gradient = tuple(
        range(control_count + target_size, control_count + target_size + width)
    )
    data_size = control_count + target_size + width
    q, c = preparation.requirements(control_count, uncompute=uncompute)
    work = tuple(range(data_size, data_size + q))
    bits = tuple(range(c))
    gates = preparation.emit(
        controls, target, gradient, work, bits, uncompute=uncompute
    )
    ports = {"target_state": target, "phase_gradient": gradient}
    if controls:
        ports["prepare_control"] = controls

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        if isinstance(bloq, Rx):
            return (
                Instruction("rx", inputs["q"], {"theta": float(bloq.angle)}),
            ), inputs
        if isinstance(bloq, XGate):
            return (Instruction("x", inputs["q"]),), inputs
        if isinstance(bloq, Allocate):
            assert not bloq.dirty and bloq.signature.get_right("reg").total_bits() == 1
            return (), {"reg": work[-1:]}
        if isinstance(bloq, Free):
            assert inputs == {"reg": work[-1:]} and not bloq.dirty
            return (), {}
        if isinstance(bloq, PRGAViaPhaseGradient):
            return source_lookup_gates(bloq, inputs, work, bits), inputs
        return None

    expected, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports
    assert_source(gates, expected)
    assert [gate for gate in gates if gate.name == "rx"] == [
        gate for gate in expected if gate.name == "rx"
    ]
    counts = Counter(gate.name for gate in gates)
    model = get_cost_value(source, QECGatesCost())
    assert counts["rx"] == 2 * target_size
    assert counts["measure"] == model.measurement + (target_size + 1) * (width - 1)
    assert counts["t"] + counts["tdg"] == 4 * counts["measure"]
    assert model.toffoli == (target_size + 1) * (width - 2)
    assert get_cost_value(source, QubitCount()) >= data_size + width + 1

    # Contract the actual source on a coherent control, target and reusable gradient.
    matrix_source = source
    if width == 8:
        # This dyadic example has the same kickback phases at either precision.
        matrix_source = StatePreparationViaRotations(
            coefficients, 2, control_bitsize=control_count, uncompute=uncompute
        )
        full_rows, full_phase = source.rotation_tree.get_rom_vals()
        short_rows, short_phase = matrix_source.rotation_tree.get_rom_vals()
        assert full_rows == [[word << 6 for word in row] for row in short_rows]
        assert full_phase == [word << 6 for word in short_phase]
    matrix = torch.tensor(matrix_source.tensor_contract(), dtype=torch.complex128)
    data = torch.zeros(1 << data_size, dtype=torch.complex128)
    phases = torch.exp(
        -1j * tau * torch.arange(1 << width, dtype=torch.float64) / (1 << width)
    )
    for address in range(1 << (data_size - width)):
        data[address << width : (address + 1) << width] = (
            complex(address + 1, -address) * phases
        )
    data /= data.norm()
    if width == 8:
        # Extract the reduced source action on its two-bit gradient eigenstate.
        short_gradient = phases[::64] / 2
        short_data = (data.reshape(-1, 256)[:, :1] * 16) * short_gradient
        short_output = (matrix @ short_data.flatten()).reshape(-1, 4)
        result = (short_output @ short_gradient.conj())[:, None] * phases / 16
        expected_sparse = {
            i << q: complex(value) for i, value in enumerate(result.flatten())
        }
        input_sparse = {i << q: complex(value) for i, value in enumerate(data)}
        # Rx(theta) = exp(-i*theta/2) H Phase(theta) H. The scalar
        # factors multiply to one across the literal paired rotations.
        rotations = []
        for gate in gates:
            if gate.name == "rx":
                rotations.extend(
                    (
                        Instruction("h", gate.wires),
                        Instruction("phase", gate.wires, gate.params),
                        Instruction("h", gate.wires),
                    )
                )
            else:
                rotations.append(gate)
        assert exp(-0.5j * sum(g.params["theta"] for g in gates if g.name == "rx")) == 1
        ir = CircuitIR(data_size + q, tuple(rotations))
        for seed in (0, 1, 27):
            actual = execute_sparse_register(ir, input_sparse, seed=seed)
            assert (
                max(
                    abs(actual.get(i, 0) - expected_sparse.get(i, 0))
                    for i in actual.keys() | expected_sparse.keys()
                )
                < 2e-12
            )
        return

    state = torch.zeros(1 << (data_size + q), dtype=torch.complex128)
    state[:: 1 << q] = data
    reference = torch.zeros_like(state)
    reference[:: 1 << q] = matrix @ data
    events = counts["measure"]
    histories = (
        product((0, 1), repeat=events)
        if events <= 4
        else ((0,) * events, (1,) * events, tuple(i % 2 for i in range(events)))
    )
    for history in histories:
        torch.testing.assert_close(
            forced_branch(state, gates, history),
            reference * 2 ** (-events / 2),
            rtol=0,
            atol=2e-12,
        )


def test_single_coefficient_has_no_source_target_port() -> None:
    with pytest.raises(ValueError, match="nonempty target"):
        RotationPreparation((1j,), 2).requirements(0)
