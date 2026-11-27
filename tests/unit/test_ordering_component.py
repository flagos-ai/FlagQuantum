"""Typed comparison trees preserve the frozen source's physical execution."""

from collections import Counter
from dataclasses import replace
from importlib import import_module
from itertools import product

import pytest
import torch

from tests._reference_dependencies import unavailable_reference

try:
    _source = import_module("qualtran.bloqs.arithmetic.comparison")
    _costs = import_module("qualtran.resource_counting")
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

from flagquantum.compiler.registers.comparison.ordering_component import (
    greater_than_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _align_private_names(
    actual: tuple[Instruction, ...], expected: tuple[Instruction, ...], logical: int
) -> tuple[Instruction, ...]:
    def history(gates: tuple[Instruction, ...], wire: int) -> tuple[object, ...]:
        return tuple(
            (
                gate.name,
                gate.wires.index(wire),
                tuple((i, q) for i, q in enumerate(gate.wires) if q < logical),
            )
            for gate in gates
            if wire in gate.wires
        )

    # Independent tree branches may allocate private flags in different order.
    # Match their entire histories; logical operands remain fixed throughout.
    names = {wire: wire for wire in range(logical)}
    private = {wire for gate in actual for wire in gate.wires if wire >= logical}
    for wire in sorted({q for gate in expected for q in gate.wires if q >= logical}):
        candidates = [
            q
            for q in private - set(names.values())
            if history(expected, wire) == history(actual, q)
        ]
        assert len(candidates) == 1, (wire, candidates)
        names[wire] = candidates[0]
    assert private == set(names.values()) - set(range(logical))
    renamed = tuple(
        replace(gate, wires=tuple(names[q] for q in gate.wires)) for gate in expected
    )
    actual_measurements = {
        gate.wires: gate for gate in actual if gate.name == "measure"
    }
    assert len(actual_measurements) == sum(gate.name == "measure" for gate in actual)
    records: dict[int, int] = {}
    for gate in renamed:
        if gate.name == "measure":
            counterpart = actual_measurements[gate.wires]
            records[gate.metadata["classical_bit"]] = counterpart.metadata[
                "classical_bit"
            ]
    assert len(set(records.values())) == len(records) == len(actual_measurements)
    result = []
    for gate in renamed:
        metadata = dict(gate.metadata)
        if "classical_bit" in metadata:
            metadata["classical_bit"] = records[metadata["classical_bit"]]
        if "conditions" in metadata:
            metadata["conditions"] = tuple(
                (records[bit], value) for bit, value in metadata["conditions"]
            )
        result.append(replace(gate, metadata=metadata))
    return tuple(result)


@pytest.mark.parametrize("width", (1, 2, 3, 5, 8))
@pytest.mark.parametrize("adjoint", (False, True))
def test_ordering_source_stream_feedback_and_resources(
    width: int, adjoint: bool
) -> None:
    graph = greater_than_program(width, adjoint=adjoint)
    source = _source.GreaterThan(width, width)
    if adjoint:
        source = source.adjoint()
    compare_layers(graph, source, {f.name for f in graph.functions if f.name != "main"})
    ports = {
        "a": tuple(range(width)),
        "b": tuple(range(width, 2 * width)),
        "target": (2 * width,),
    }
    work = tuple(range(2 * width + 1, 5 * width))
    bound = bind_register_program(graph, tuple(ports.values()), clean=work)
    expected = reference_arithmetic_gates(
        source,
        ports,
        work,
        MeasurementSlots(bound.circuit.instructions),
        preserve_components=True,
    )
    assert_source(
        bound.circuit.instructions,
        _align_private_names(bound.circuit.instructions, expected, 2 * width + 1),
    )
    counts = Counter(g.name for g in bound.circuit.instructions)
    model = _costs.get_cost_value(source, _costs.QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count()
    assert counts["measure"] == model.measurement == 2 * width - 1


@pytest.mark.parametrize("width", (1, 2, 3))
@pytest.mark.parametrize("adjoint", (False, True))
def test_ordering_full_complex_dirty_target_and_every_raw_history(
    width: int, adjoint: bool
) -> None:
    logical = 2 * width + 1
    work = 3 * width - 1
    raw = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 7 - 3 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (logical + 1 + work), dtype=torch.complex128)
    initial[:: 1 << work] = amplitudes
    source = _source.GreaterThan(width, width)
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        1 << logical, 1 << logical
    )
    if adjoint:
        matrix = matrix.mH
    expected = torch.zeros_like(initial)
    expected[:: 1 << work] = (matrix @ amplitudes.reshape(-1, 2)).flatten()
    bound = bind_register_program(
        greater_than_program(width, adjoint=adjoint),
        (tuple(range(width)), tuple(range(width, 2 * width)), (2 * width,)),
        clean=tuple(range(logical + 1, logical + work + 1)),
    )
    for history in product((0, 1), repeat=2 * width - 1):
        torch.testing.assert_close(
            forced_branch(initial, bound.circuit.instructions, history),
            expected / 2 ** ((2 * width - 1) / 2),
            rtol=0,
            atol=3e-12,
        )
