"""Literal controlled doubling retains source calls and source counterexamples."""

from importlib import import_module

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    for dependency in ("numpy", "cirq", "kickmix"):
        import_module(dependency)
    from qualtran import QUInt
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import ModDbl
except ModuleNotFoundError as error:
    unavailable_reference(error)

from flagquantum.compiler.registers.modular.controlled_doubling_component import (
    controlled_modular_doubling_program,
)
from flagquantum.core.ir.program.register_types import unsigned_register_type
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_symbolic_modular_doubling_calls import (
    frozen_doubling_source as frozen_doubling_source,
)
from tests.unit.test_symbolic_modular_doubling_calls import (
    frozen_source as frozen_source,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("adjoint", (False, True))
def test_negative_control_source_wrapper(adjoint: bool) -> None:
    from qualtran import CtrlSpec

    from flagquantum.compiler.formats.legacy_component_resolution import (
        resolve_component_program,
    )
    from flagquantum.compiler.formats.legacy_modular_multiplication_factories import (
        modular_multiplication_factories,
    )
    from tests.unit.test_legacy_arithmetic_factories import _library

    source = ModDbl(QUInt(2), 3)
    source = (source.adjoint() if adjoint else source).controlled(CtrlSpec(cvs=0))
    record = resolve_component_program(
        _library(source, 0), 0, modular_multiplication_factories()
    )
    calls = record.functions[0].blocks[0].operations
    assert len(calls) == 3
    compare_layers(record, source, {op.callee for op in calls if op.callee is not None})


@pytest.mark.parametrize("width,modulus", ((2, 3), (3, 3), (3, 7)))
@pytest.mark.parametrize("adjoint", (False, True))
def test_controlled_doubling_root_and_operand_order(
    width: int, modulus: int, adjoint: bool
) -> None:
    source = ModDbl(QUInt(width), modulus)
    source = (source.adjoint() if adjoint else source).controlled()
    record = controlled_modular_doubling_program(
        unsigned_register_type(width), modulus, adjoint=adjoint
    )
    calls = record.functions[0].blocks[0].operations
    assert len(calls) == 15
    compare_layers(record, source, {op.callee for op in calls if op.callee is not None})


def test_original_inactive_control_layout_counterexample() -> None:
    import numpy as np

    source = ModDbl(QUInt(2), 3).controlled()
    operator = np.asarray(source.decompose_bloq().tensor_contract())
    assert source.call_classically(ctrl=0, x=1) == (0, 1)
    np.testing.assert_array_equal(operator[:, 1], np.eye(8)[:, 2])
    assert source.call_classically(ctrl=0, x=2) == (0, 2)
    np.testing.assert_array_equal(operator[:, 2], np.zeros(8))


@pytest.mark.parametrize("adjoint", (False, True))
def test_literal_primitives_feedback_and_coherent_clean_sector(adjoint: bool) -> None:
    from collections import Counter

    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from flagquantum.compiler.registers.register_binding import bind_register_program
    from tests.unit.sparse_register_execution import execute_sparse_register
    from tests.unit.test_multi_control_component import MeasurementSlots
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    source = ModDbl(QUInt(2), 3)
    source = (source.adjoint() if adjoint else source).controlled()
    record = controlled_modular_doubling_program(
        unsigned_register_type(2), 3, adjoint=adjoint
    )
    ports = {"ctrl": (0,), "x": (1, 2)}
    bound = bind_register_program(
        record,
        tuple(ports.values()),
        clean=tuple(range(3, 14)),
        input_ownership=((1, False), (2, False)),
        allow_workspace_outputs=True,
    )
    allocations = tuple(
        wire
        for event in bound.events
        if event.kind == "allocate"
        for wire in event.wires
    )
    expected_gates = reference_arithmetic_gates(
        source,
        ports,
        allocations,
        MeasurementSlots(bound.circuit.instructions),
        expected_outputs=dict(zip(ports, bound.outputs, strict=True)),
        reuse_allocations=False,
        preserve_components=True,
        keep_exchange_leaves=True,
    )
    assert_source(bound.circuit.instructions, expected_gates)
    counts = Counter(op.name for op in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert counts["measure"] == model.measurement
    assert counts["t"] + counts["tdg"] + 4 * counts["ccx"] == model.total_t_count()

    def pack(words: tuple[int, int], layout: tuple[tuple[int, ...], ...]) -> int:
        return sum(
            ((value >> (len(wires) - i - 1)) & 1) << (bound.circuit.n_wires - wire - 1)
            for value, wires in zip(words, layout, strict=True)
            for i, wire in enumerate(wires)
        )

    # The entire promised active residue sector, plus the clean inactive zero.
    inputs = ((0, 0), (1, 0), (1, 1), (1, 2))
    amplitudes = tuple(complex(i + 1, 3 - i) for i in range(len(inputs)))
    norm = sum(abs(value) ** 2 for value in amplitudes) ** 0.5
    before = {
        pack(words, tuple(ports.values())): amplitude / norm
        for words, amplitude in zip(inputs, amplitudes, strict=True)
    }
    expected = {
        pack((ctrl, (2 * x) % 3 if ctrl else 0), bound.outputs): amplitude / norm
        for (ctrl, x), amplitude in zip(inputs, amplitudes, strict=True)
    }
    for seed in (0, 19):
        assert execute_sparse_register(
            bound.circuit, before, seed=seed
        ) == pytest.approx(expected, abs=3e-12)
    from itertools import product

    import torch

    from tests.unit.measurement_checks import forced_branch

    initial = torch.zeros(1 << bound.circuit.n_wires, dtype=torch.complex128)
    final = torch.zeros_like(initial)
    for basis, amplitude in before.items():
        initial[basis] = amplitude
    for basis, amplitude in expected.items():
        final[basis] = amplitude
    for history in product((0, 1), repeat=counts["measure"]):
        torch.testing.assert_close(
            forced_branch(initial, bound.circuit.instructions, history),
            final / 2 ** (counts["measure"] / 2),
            rtol=0,
            atol=3e-12,
        )


@pytest.mark.parametrize("adjoint", (False, True))
@pytest.mark.parametrize("depth", (0, 1))
def test_source_constructor_roundtrip(adjoint: bool, depth: int) -> None:
    from flagquantum.compiler.formats.legacy_component_resolution import (
        resolve_component_program,
    )
    from flagquantum.compiler.formats.legacy_modular_multiplication_factories import (
        modular_multiplication_factories,
    )
    from flagquantum.compiler.registers.component_definition_match import (
        matches_component_definition,
    )
    from tests.unit.test_legacy_arithmetic_factories import _library

    source = ModDbl(QUInt(2), 3)
    source = (source.adjoint() if adjoint else source).controlled()
    record = resolve_component_program(
        _library(source, depth), 0, modular_multiplication_factories()
    )
    assert matches_component_definition(
        record,
        record.functions[0].name,
        controlled_modular_doubling_program(
            unsigned_register_type(2), 3, adjoint=adjoint
        ),
    )
