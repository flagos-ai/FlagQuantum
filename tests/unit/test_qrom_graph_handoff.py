"""Actual QROM typed graphs retain native execution and source failure stages."""

from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.lookup.qrom_table import QromTable
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_graph import import_register_graph
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import assert_source_dependencies
from tests.unit.test_unary_component_calls import source_slot_names

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")


@pytest.mark.parametrize("grouped,controls", tuple(product((False, True), (0, 1))))
def test_actual_qrom_graph_handoff_preserves_joint_channel_and_resources(
    grouped: bool, controls: int
) -> None:
    from qualtran.bloqs.data_loading.qrom import QROM
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    data = np.array([[0, 1], [1, 0]]) if grouped else np.array([1, 0])
    shapes = ((2,),) if grouped else ((),)
    source = QROM.build_from_data(data, target_shapes=shapes, num_controls=controls)
    table = QromTable.from_data(data, target_shapes=shapes, num_controls=controls)
    ports: dict[str, tuple[int, ...]] = {}
    inputs = []
    logical = 0
    for register in source.signature:
        start = logical
        for _ in register.all_idxs():
            width = int(register.bitsize)
            inputs.append(tuple(range(logical, logical + width)))
            logical += width
        ports[register.name] = tuple(range(start, logical))
    clean = tuple(range(logical, logical + table.workspace_size))
    width = logical + len(clean) + 1
    # The final wire is an external coherent spectator, outside source resources.
    raw = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    joint = torch.complex(raw, 2 - raw)
    joint /= joint.norm()
    initial = torch.zeros(1 << width, dtype=torch.complex128)
    initial.reshape(1 << logical, 1 << len(clean), 2)[:, 0, :] = joint.reshape(-1, 2)
    graph = source.decompose_bloq()
    for direction in (graph, graph.adjoint()):
        program = import_register_graph(direction)
        bound = bind_register_program(program, tuple(inputs), clean=clean)
        assert bound.outputs == tuple(inputs)
        expected = reference_arithmetic_gates(
            direction,
            ports,
            clean,
            dict(zip(clean, range(len(clean)), strict=True)),
            cirq_style=False,
            keep_pauli_leaves=True,
        )
        assert_source_dependencies(
            source_slot_names(bound.circuit.instructions, clean), expected
        )
        circuit = Circuit(width)
        circuit.append(FieldOperation(width, bound.circuit.instructions))
        actual_ir = circuit.to_ir()
        assert CircuitIR.from_json(actual_ir.to_json()) == actual_ir
        counts = Counter(gate.name for gate in actual_ir.instructions)
        for history in product((0, 1), repeat=counts["measure"]):
            actual = forced_branch(initial, actual_ir.instructions, history)
            original = forced_branch(initial, expected, history)
            torch.testing.assert_close(actual, original, atol=2e-12, rtol=0)
            assert (
                torch.count_nonzero(actual.reshape(-1, 1 << len(clean), 2)[:, 1:, :])
                == 0
            )
        events = Counter(event.kind for event in bound.events)
        assert events["allocate"] == events["free"]
        model = get_cost_value(direction, QECGatesCost())
        assert counts["measure"] == model.measurement
        assert counts["t"] + counts["tdg"] == model.total_t_count()
        assert get_cost_value(direction, QubitCount()) == width - 1


def boundary_source(case: str) -> Any:
    from qualtran.bloqs.data_loading.qrom import QROM

    if case == "leading":
        return QROM.build_from_data(np.array([[1, 0, 2]]))
    if case == "trailing":
        return QROM.build_from_data(np.array([[1], [0], [2]]))
    if case == "middle":
        return QROM.build_from_data(np.arange(6).reshape(2, 1, 3))
    if case == "wrong_target":
        return QROM.build_from_data(np.arange(6).reshape(3, 2), target_shapes=((3,),))
    if case == "scalar":
        return QROM.build_from_data(1, [1, 2])
    raise ValueError(f"Unknown source boundary {case}")


@pytest.mark.parametrize(
    "case,error,match",
    (
        ("leading", TypeError, "0-dimensional"),
        ("trailing", TypeError, "0-dimensional"),
        ("middle", TypeError, "0-dimensional"),
        ("wrong_target", IndexError, "out of bounds"),
        ("scalar", IndexError, "too many indices"),
    ),
)
def test_original_decomposition_failures_reach_graph_boundary_unchanged(
    case: str, error: type[Exception], match: str
) -> None:
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = boundary_source(case)
    assert source.signature
    for action in (
        source.decompose_bloq,
        lambda: get_cost_value(source, QECGatesCost()),
        lambda: import_register_graph(source),
    ):
        with pytest.raises(error, match=match):
            action()


def test_original_zero_target_fails_after_shallow_decomposition() -> None:
    from qualtran.bloqs.data_loading.qrom import QROM
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = QROM.build_from_data([0, 0, 0, 0], target_bitsizes=(0,))
    graph = source.decompose_bloq()
    assert tuple(reg.name for reg in source.signature) == ("selection",)
    assert get_cost_value(source, QECGatesCost()).total_t_count() == 0
    # The first decomposition succeeds. Its original XorK(0) child cannot flatten.
    for action in (graph.flatten, lambda: import_register_graph(source)):
        with pytest.raises(
            TypeError, match="missing 1 required positional argument: 'x'"
        ):
            action()
