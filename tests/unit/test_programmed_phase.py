"""Locked resource-state preparation and its source costing boundary."""

from cmath import exp
from collections import Counter
from itertools import product
from math import pi, sqrt
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.phase import programmed_phase
from flagquantum.compiler.registers.phase.programmed_phase import (
    programmed_phase_ir,
    programmed_phase_rotation_ir,
)
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
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


@pytest.mark.parametrize("exponent", [-0.375, 0.125])
@pytest.mark.parametrize("slots", [(0,), (3, 0), (4, 1, 3)])
@pytest.mark.parametrize("correct", [False, True])
def test_programmed_rotation_all_raw_branches(exponent, slots, correct):
    # Work occupies the middle wire of an entangled data/spectator state.
    initial = torch.tensor([1, 2j, 0, 0, -0.5j, 1 + 2j, 0, 0], dtype=torch.complex128)
    initial /= initial.norm()
    gates = programmed_phase_rotation_ir(
        0, 1, slots, exponent=exponent, apply_final_correction=correct
    )
    for history in product((0, 1), repeat=len(slots)):
        actual = forced_branch(initial, gates, history)
        failures = next((i for i, bit in enumerate(history) if bit == 0), len(slots))
        if any(history[failures:]):
            torch.testing.assert_close(
                actual, torch.zeros_like(initial), atol=2e-12, rtol=0
            )
            continue
        attempts = min(failures + 1, len(slots))
        failed_phase = (2**failures - 1) * exponent
        net_exponent = exponent if failures < len(slots) or correct else -failed_phase
        expected = initial * exp(1j * pi * failed_phase) * 2 ** (-attempts / 2)
        expected[4:] *= exp(1j * pi * net_exponent)
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=0)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("correct", [False, True])
def test_programmed_rotation_matches_source_resource_schedule(monkeypatch, correct):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import CNOT, MeasureZ, XGate, ZPowGate
    from qualtran.bloqs.rotations.programmable_ancilla_rotation import (
        ZPowProgrammedAncilla,
        ZPowUsingProgrammedAncilla,
    )
    from qualtran.resource_counting import SympySymbolAllocator

    calls = []

    def prepare(output, *, exponent, epsilon):
        calls.append(ZPowProgrammedAncilla(exponent, epsilon))
        return programmed_phase_ir(output, exponent=exponent, epsilon=epsilon)

    monkeypatch.setattr(programmed_phase, "programmed_phase_ir", prepare)
    slots = (4, 1, 3)
    source = ZPowUsingProgrammedAncilla(
        -0.125, eps=0.006, n_rounds=len(slots), apply_final_correction=correct
    )
    gates = programmed_phase_rotation_ir(
        0,
        1,
        slots,
        exponent=source.exponent,
        epsilon=source.eps,
        apply_final_correction=correct,
    )
    assert len(gates) == 5 * len(slots) + correct
    for i, bit in enumerate(slots):
        h, phase, cx, measure, x = gates[5 * i : 5 * (i + 1)]
        active = tuple(sorted((b, 1) for b in slots[:i]))
        assert [gate.name for gate in (h, phase, cx, measure, x)] == [
            "h",
            "phase",
            "cx",
            "measure",
            "x",
        ]
        assert [gate.wires for gate in (h, phase, cx, measure, x)] == [
            (1,),
            (1,),
            (0, 1),
            (1,),
            (1,),
        ]
        assert all(g.metadata.get("conditions", ()) == active for g in (h, phase, cx))
        assert measure.metadata == {"is_dynamic": True, "classical_bit": bit}
        assert x.metadata == {"conditions": tuple(sorted((*active, (bit, 1))))}
        assert phase.params["theta"] == pi * source.exponent * 2**i
    resources = Counter(calls)
    resources.update({CNOT(): len(slots), XGate(): len(slots), MeasureZ(): len(slots)})
    if correct:
        correction = gates[-1]
        assert correction.name == "phase" and correction.wires == (0,)
        assert correction.metadata == {
            "conditions": tuple(sorted((b, 1) for b in slots))
        }
        resources[ZPowGate(correction.params["theta"] / pi, eps=calls[0].eps)] += 1
    assert resources == source.build_call_graph(SympySymbolAllocator())


def test_programmed_rotation_append_trajectory():
    from dataclasses import dataclass

    from flagquantum.core.ir import CircuitIR
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    @dataclass(frozen=True)
    class Rotate:
        def append_to(self, circuit: DynamicCircuit) -> DynamicCircuit:
            circuit.compose(
                CircuitIR(
                    2,
                    programmed_phase_rotation_ir(
                        0,
                        1,
                        (0, 1),
                        exponent=0.125,
                        apply_final_correction=True,
                    ),
                )
            )
            return circuit

    initial = torch.tensor([0.8, 0, 0.6j, 0], dtype=torch.complex128)
    circuit = DynamicCircuit(2, inputs=initial, dtype=torch.complex128)
    circuit.append(Rotate())
    result = run_dynamic(circuit, shots=32, seed=42, strategy="trajectory")
    histories = set()
    for record, actual in zip(
        result.classical_bits.tolist(), result.final_states, strict=True
    ):
        history = tuple(record)
        histories.add(history)
        failures = sum(record)
        expected = initial * exp(1j * pi * (2**failures - 1) / 8)
        expected[2:] *= exp(1j * pi / 8)
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=0)
    assert histories == {(0, 0), (1, 0), (1, 1)}


@pytest.mark.parametrize(
    "data,work,slots,epsilon,correct,exponent",
    [
        (0, 0, (0,), 0.01, False, 0.125),
        (0, 1, (), 0.01, False, 0.125),
        (0, 1, (0, 0), 0.01, False, 0.125),
        (0, 1, (0,), 0.01, 1, 0.125),
        (0, 1, (0,), 0, False, 0.125),
        (0, 1, (0,), 0.01, True, 4e307),
    ],
)
def test_programmed_rotation_rejects_invalid_inputs(
    data, work, slots, epsilon, correct, exponent
):
    with pytest.raises(ValueError):
        programmed_phase_rotation_ir(
            data,
            work,
            slots,
            exponent=exponent,
            epsilon=epsilon,
            apply_final_correction=correct,
        )
