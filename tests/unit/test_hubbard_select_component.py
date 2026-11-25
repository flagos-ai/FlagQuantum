"""Complete controlled Hubbard SELECT graphs and native execution closure."""

from collections import Counter
from itertools import product

import pytest

from flagquantum.algorithms.primitives.hubbard_select import SelectHubbard
from flagquantum.algorithms.primitives.hubbard_select_component import (
    hubbard_select_program,
)
from flagquantum.algorithms.primitives.hubbard_spin_up import HubbardSpinUpZ
from flagquantum.algorithms.primitives.hubbard_spin_up_component import (
    hubbard_spin_up_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("dimension", [2, 3, 4, 5])
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("adjoint", [False, True])
def test_complete_spin_up_calls_partition_views_and_inverse(
    dimension: int, controlled: bool, adjoint: bool
) -> None:
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard import (
        HubbardSpinUpZ as Source,
    )

    record = hubbard_spin_up_program(
        HubbardSpinUpZ(dimension, controlled), adjoint=adjoint
    )
    original = Source(dimension, dimension, control_val=1 if controlled else None)
    assert verify_program_record(record) == ()
    compare_layers(
        record, original.adjoint() if adjoint else original, atomic_leaves(record)
    )


@pytest.mark.parametrize("dimension", [2, 3, 4, 5])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("controlled", [False, True])
def test_complete_select_all_recursive_calls(
    dimension: int, adjoint: bool, controlled: bool
) -> None:
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard import (
        SelectHubbard as Source,
    )

    record = hubbard_select_program(
        SelectHubbard(dimension, controlled), adjoint=adjoint
    )
    original = Source(dimension, dimension, control_val=1 if controlled else None)
    assert verify_program_record(record) == ()
    compare_layers(
        record,
        original.adjoint() if adjoint else original,
        atomic_leaves(record) | {f.name for f in record.functions if f.registers == ()},
    )


def assert_sparse_equal(
    actual: dict[int, complex], expected: dict[int, complex]
) -> None:
    assert (
        max(
            (
                abs(actual.get(i, 0) - expected.get(i, 0))
                for i in actual.keys() | expected.keys()
            ),
            default=0,
        )
        < 2e-11
    )


@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("adjoint", [False, True])
def test_spin_up_complex_state_every_raw_history(
    controlled: bool, adjoint: bool
) -> None:
    operation = HubbardSpinUpZ(2, controlled)
    ports: dict[str, tuple[int, ...]] = {}
    logical = 0
    for name, size in operation.layout:
        ports[name] = tuple(range(logical, logical + size))
        logical += size
    work, slots = operation.requirements()
    clean = tuple(range(logical, logical + work))
    record = hubbard_spin_up_program(operation, adjoint=adjoint)
    bound = bind_register_program(record, tuple(ports.values()), clean=clean)
    flat = operation._emit(ports, clean, tuple(range(slots)))
    assert Counter(g.name for g in bound.circuit.instructions) == Counter(
        g.name for g in flat
    )
    # Include every coordinate/control pattern with complex target/spectator data.
    initial = {(i * 137 % (1 << logical)) << work: 1j**i / 8 for i in range(64)}
    expected = _branch(
        flat,
        logical + work,
        (0,) * Counter(g.name for g in flat)["measure"],
        initial,
        normalize_measurements=True,
    )
    events = Counter(g.name for g in bound.circuit.instructions)["measure"]
    for history in product((0, 1), repeat=events):
        actual = _branch(
            bound.circuit.instructions,
            logical + work,
            history,
            initial,
            normalize_measurements=True,
        )
        assert_sparse_equal(actual, expected)
        assert (
            sum(abs(a) ** 2 for i, a in actual.items() if i & ((1 << work) - 1)) < 1e-22
        )


@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("controlled", [False, True])
def test_complete_select_complex_branches_work_json_copy_and_append(
    adjoint: bool,
    controlled: bool,
) -> None:
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum import Circuit
    from tests.unit.measurement_certificate import certify_erasure_histories

    operation = SelectHubbard(2, controlled)
    ports: dict[str, tuple[int, ...]] = {}
    logical = 0
    for name, size in operation.layout:
        ports[name] = tuple(range(logical, logical + size))
        logical += size
    work, slots = operation.requirements()
    clean = tuple(range(logical, logical + work))
    record = hubbard_select_program(operation, adjoint=adjoint)
    bound = bind_register_program(record, tuple(ports.values()), clean=clean)
    assert (
        bind_register_program(
            ProgramRecord(record.functions), tuple(ports.values()), clean=clean
        ).circuit
        == bound.circuit
    )
    restored = CircuitIR.from_json(bound.circuit.to_json())
    assert (
        Circuit(restored.n_wires)
        .append(FieldOperation(restored.n_wires, restored.instructions))
        .to_ir()
        .instructions
        == restored.instructions
    )
    # Existing accepted flat source emitter supplies the independent SELECT operator.
    flat = operation._emit(ports, clean, tuple(range(slots)), adjoint=adjoint)
    assert Counter(g.name for g in restored.instructions) == Counter(
        g.name for g in flat
    )
    events = certify_erasure_histories(
        restored.instructions, n_wires=restored.n_wires, zero_wires=clean
    )
    assert events == Counter(g.name for g in restored.instructions)["measure"]
    initial = {(i * 1973 % (1 << logical)) << work: 1j**i / 8 for i in range(64)}
    flat_events = Counter(g.name for g in flat)["measure"]
    expected = _branch(
        flat, logical + work, (0,) * flat_events, initial, normalize_measurements=True
    )
    for history in ((0,) * events, (1,) * events, tuple(i % 2 for i in range(events))):
        actual = _branch(
            restored.instructions,
            logical + work,
            history,
            initial,
            normalize_measurements=True,
        )
        assert_sparse_equal(actual, expected)
        assert (
            sum(abs(a) ** 2 for i, a in actual.items() if i & ((1 << work) - 1)) < 1e-22
        )


def test_original_stored_boundaries_remain_explicit() -> None:
    with pytest.raises(ValueError, match="singleton"):
        hubbard_spin_up_program(HubbardSpinUpZ(1))
    with pytest.raises(ValueError, match="zero control"):
        hubbard_spin_up_program(HubbardSpinUpZ(2, True, control_value=0))
    with pytest.raises(ValueError, match="zero control"):
        hubbard_select_program(SelectHubbard(2, True, control_value=0))


@pytest.mark.parametrize("strategy", ["trajectory", "batched"])
@pytest.mark.parametrize("adjoint", [False, True])
def test_uncontrolled_scalar_carrier_normal_append_execution(
    strategy: str, adjoint: bool
) -> None:
    import torch

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    operation = SelectHubbard(2)
    ports: dict[str, tuple[int, ...]] = {}
    logical = 0
    for name, width in operation.layout:
        ports[name] = tuple(range(logical, logical + width))
        logical += width
    work, slots = operation.requirements()
    clean = tuple(range(logical, logical + work))
    bound = bind_register_program(
        hubbard_select_program(operation, adjoint=adjoint),
        tuple(ports.values()),
        clean=clean,
    )
    count = logical + work
    sparse = {(i * 1973 % (1 << logical)) << work: 1j**i / 8 for i in range(64)}
    flat = operation._emit(ports, clean, tuple(range(slots)), adjoint=adjoint)
    expected_sparse = _branch(
        flat, count, (0,) * 15, sparse, normalize_measurements=True
    )
    initial = torch.zeros(1 << count, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for word, value in sparse.items():
        initial[word] = torch.tensor(value, dtype=torch.complex128)
    for word, value in expected_sparse.items():
        expected[word] = torch.tensor(value, dtype=torch.complex128)
    circuit = DynamicCircuit(count, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(count, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=2, seed=42, strategy=strategy)
    assert result.statistics["measurement_count"] == 30
    for state in result.final_states:
        torch.testing.assert_close(state.flatten(), expected, atol=2e-11, rtol=0)
        assert state.flatten().reshape(-1, 1 << work)[:, 1:].abs().max() < 2e-11
