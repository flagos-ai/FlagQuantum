"""Locked resource-state preparation and its source costing boundary."""

from collections import Counter
from math import pi, sqrt
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.programmed_phase import programmed_phase_ir
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("exponent", [0.0, 0.25, -0.375, 0.625, 2.75])
@pytest.mark.parametrize("output", [0, 1])
def test_programmed_phase_complex_state_and_source(
    exponent: float, output: int
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QBit
    from qualtran.bloqs.basic_gates import Hadamard, ZPowGate
    from qualtran.bloqs.bookkeeping import Allocate
    from qualtran.bloqs.rotations.programmable_ancilla_rotation import (
        ZPowProgrammedAncilla,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    epsilon = 0.001
    source = ZPowProgrammedAncilla(exponent, epsilon)
    operations = []

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        operations.append(bloq)
        ports = {"q": (output,)}
        if isinstance(bloq, Allocate):
            assert bloq.dtype == QBit() and not inputs
            return (), {"reg": (output,)}
        if isinstance(bloq, Hadamard):
            assert inputs == ports
            return (Instruction("h", (output,)),), ports
        if isinstance(bloq, ZPowGate):
            assert inputs == desired == ports
            assert (bloq.exponent, bloq.eps) == (exponent, epsilon)
            return (
                Instruction("phase", (output,), params={"theta": pi * bloq.exponent}),
            ), ports
        return None

    expected_gates, ports = walk_source_graph(source, {}, {"q": (output,)}, leaf)
    assert ports == {"q": (output,)}
    assert operations[1:] == [
        Allocate(QBit()),
        Hadamard(),
        ZPowGate(exponent, eps=epsilon),
    ]
    gates = programmed_phase_ir(output, exponent=exponent, epsilon=epsilon)
    assert gates == expected_gates
    assert Counter(gate.name for gate in gates) == {"h": 1, "phase": 1}
    spectator = torch.tensor([0.8, 0.6j], dtype=torch.complex128)
    zero = torch.tensor([1, 0], dtype=torch.complex128)
    source_state = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    order = (zero, spectator) if output == 0 else (spectator, zero)
    initial = torch.kron(*order)
    expected = torch.kron(
        *((source_state, spectator) if output == 0 else (spectator, source_state))
    )
    actual = forced_branch(initial, gates, ())
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=0)
    assert abs(source_state[0].item() - 1 / sqrt(2)) < 2e-12
    assert get_cost_value(source, QubitCount()) == 1
    model = QECGatesCost()
    assert get_cost_value(source, model) == (
        get_cost_value(Hadamard(), model)
        + get_cost_value(ZPowGate(exponent, eps=epsilon), model)
    )


@pytest.mark.parametrize(
    "output,exponent,epsilon",
    [(-1, 0.25, 1e-11), (0, float("nan"), 1e-11), (0, 1.7e308, 1e-11), (0, 0.25, 0)],
)
def test_programmed_phase_rejects_invalid_bindings(
    output: int, exponent: float, epsilon: float
) -> None:
    with pytest.raises(ValueError):
        programmed_phase_ir(output, exponent=exponent, epsilon=epsilon)


def test_append_prepares_the_resource() -> None:
    from dataclasses import dataclass

    from flagquantum.core.ir import CircuitIR
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    @dataclass(frozen=True)
    class Prepare:
        def append_to(self, circuit: DynamicCircuit) -> DynamicCircuit:
            circuit.compose(CircuitIR(1, programmed_phase_ir(0, exponent=-0.25)))
            return circuit

    circuit = DynamicCircuit(1, dtype=torch.complex128)
    circuit.append(Prepare())
    result = run_dynamic(circuit, shots=1, strategy="trajectory")
    expected = torch.tensor([1, (1 - 1j) / sqrt(2)], dtype=torch.complex128) / sqrt(2)
    torch.testing.assert_close(result.final_states[0], expected, atol=2e-12, rtol=0)
