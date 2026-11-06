"""Concrete sparse rotation preparation graph, amplitudes, and costs."""

from cmath import exp
from collections import Counter
from itertools import product
from math import tau
from typing import Any

import pytest
import torch

from flagquantum.algorithms.primitives.sparse_rotation_preparation import (
    SparseRotationPreparation,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_rotation_preparation import (
    execute_rotation_sparse,
    source_rotation_preparation_gates,
)

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "indices,coefficients,size,width,phase_width",
    [
        ((0, 1), (1 + 1j, 1 - 1j), 2, 1, 2),
        ((2, 1), (1 + 1j, 1 - 1j), 3, 2, 2),
        ((3, 1, 4), (1, 1j, -1), 5, 3, 2),
        ((2, 1), (1 + 1j, 1 - 1j), 3, 4, 8),
    ],
)
def test_source_sparse_preparation(
    indices: tuple[int, ...],
    coefficients: tuple[complex, ...],
    size: int,
    width: int,
    phase_width: int,
) -> None:
    pytest.importorskip("qualtran")
    np = pytest.importorskip("numpy")
    cirq = pytest.importorskip("cirq")
    from qualtran.bloqs.arithmetic.permutation import Permutation
    from qualtran.bloqs.bookkeeping import Partition
    from qualtran.bloqs.state_preparation.sparse_state_preparation_via_rotations import (
        SparseStatePreparationViaRotations,
    )
    from qualtran.bloqs.state_preparation.state_preparation_via_rotation import (
        StatePreparationViaRotations,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = SparseStatePreparationViaRotations(
        indices, coefficients, size, phase_width, width
    )
    native = SparseRotationPreparation(indices, coefficients, size, phase_width)
    work_size, slot_count = native.requirements(width)
    target = tuple(range(width))
    gradient = tuple(range(width, width + phase_width))
    data_size = width + phase_width
    work = tuple(range(data_size, data_size + work_size))
    bits = tuple(range(slot_count))
    ports = {"target_state": target, "phase_gradient": gradient}
    actual = native.emit(target, gradient, work, bits)
    children = []

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        if isinstance(bloq, Partition):
            children.append("split" if bloq.partition else "join")
            op, outputs = bloq.as_cirq_op(
                qubit_manager=None,
                **{
                    name: np.array([cirq.LineQubit(i) for i in qubits])
                    for name, qubits in inputs.items()
                },
            )
            assert op is None
            return (), {
                name: tuple(q.x for q in qubits.flat)
                for name, qubits in outputs.items()
            }
        if isinstance(bloq, StatePreparationViaRotations):
            children.append("dense")
            assert inputs["target_state"] == target[-int(bloq.state_bitsize) :]
            return (
                source_rotation_preparation_gates(
                    bloq, inputs, work[:dense_work], bits
                ),
                inputs,
            )
        if isinstance(bloq, Permutation):
            children.append("permutation")
            assert inputs["x"] == target
            marker, carries = work[0], work[1:width]
            allocation_plan = tuple(
                wire
                for cycle in bloq.cycles
                for i in range(len(cycle) + 1)
                for wire in (*carries, *((marker,) if i == 0 else ()))
            )
            return (
                reference_arithmetic_gates(
                    bloq,
                    inputs,
                    allocation_plan,
                    dict(zip(carries, bits[: width - 1], strict=True)),
                    reuse_allocations=False,
                ),
                inputs,
            )
        return None

    # Bind the source dense phase flag to the same prefix slice as the native stage.
    dense_work, _ = SparseRotationPreparation(
        tuple(range(len(indices))), coefficients, size, phase_width
    ).requirements(width)
    expected, returned = walk_source_graph(source, ports, ports, leaf)
    assert dense_work <= work_size
    assert returned == ports and children == ["split", "dense", "join", "permutation"]
    assert_source(actual, expected)
    assert [g for g in actual if g.name == "rx"] == [
        g for g in expected if g.name == "rx"
    ]
    counts = Counter(gate.name for gate in actual)
    model = get_cost_value(source, QECGatesCost())
    assert counts["measure"] == model.measurement + (source.dense_bitsize + 1) * (
        phase_width - 1
    )
    assert counts["t"] + counts["tdg"] == 4 * counts["measure"]
    assert model.toffoli == (source.dense_bitsize + 1) * (phase_width - 2)

    if phase_width == 8:
        # The unmodified small source can be contracted with the same phases.
        short = SparseStatePreparationViaRotations(
            indices, coefficients, size, 2, width
        )
        full_rows, full_phase = (
            source._dense_stateprep_bloq.rotation_tree.get_rom_vals()
        )
        short_rows, short_phase = (
            short._dense_stateprep_bloq.rotation_tree.get_rom_vals()
        )
        assert full_rows == [[value << 6 for value in row] for row in short_rows]
        assert full_phase == [value << 6 for value in short_phase]
        short_state = torch.zeros(1 << (width + 2), dtype=torch.complex128)
        short_gradient = torch.tensor([1, -1j, -1, 1j], dtype=torch.complex128) / 2
        short_state[:4] = short_gradient
        matrix = torch.tensor(short.tensor_contract(), dtype=torch.complex128)
        short_result = (matrix @ short_state).reshape(-1, 4) @ short_gradient.conj()
        catalyst = tuple(exp(-1j * tau * i / 256) / 16 for i in range(256))
        initial = {i << work_size: value for i, value in enumerate(catalyst)}
        reference_sparse = {
            ((address << 8) | i) << work_size: complex(value) * factor
            for address, value in enumerate(short_result)
            for i, factor in enumerate(catalyst)
        }
        for seed in (0, 1, 27):
            output = execute_rotation_sparse(
                actual, data_size + work_size, initial, seed
            )
            assert (
                max(
                    abs(output.get(i, 0) - reference_sparse.get(i, 0))
                    for i in output.keys() | reference_sparse.keys()
                )
                < 2e-12
            )
        return
    source_matrix = torch.tensor(source.tensor_contract(), dtype=torch.complex128)
    data = torch.arange(1 << data_size, dtype=torch.float64)
    vector = torch.complex(data + 1, data % 7 - 3)
    vector /= vector.norm()
    state = torch.zeros(1 << (data_size + work_size), dtype=torch.complex128)
    state[:: 1 << work_size] = vector
    reference = torch.zeros_like(state)
    reference[:: 1 << work_size] = source_matrix @ vector
    events = counts["measure"]
    histories = (
        product((0, 1), repeat=events)
        if events <= 5
        else ((0,) * events, (1,) * events, tuple(i % 2 for i in range(events)))
    )
    for history in histories:
        torch.testing.assert_close(
            forced_branch(state, actual, history),
            reference * 2 ** (-events / 2),
            rtol=0,
            atol=2e-12,
        )


@pytest.mark.ftqc_reference
def test_source_loses_a_prefix_and_singleton_lacks_target_port() -> None:
    pytest.importorskip("qualtran")
    from qualtran import BloqError
    from qualtran.bloqs.state_preparation.sparse_state_preparation_via_rotations import (
        SparseStatePreparationViaRotations,
    )

    broken = SparseStatePreparationViaRotations((3, 0), (1, 1), 4, 2)
    assert broken._basis_permutation_bloq.cycles == ((0, 3),)
    with pytest.raises(ValueError, match="loses an indexed transition"):
        SparseRotationPreparation((3, 0), (1, 1), 4, 2).requirements(2)
    with pytest.raises(BloqError, match="does not accept Soquets"):
        SparseStatePreparationViaRotations((2,), (1,), 3, 2).decompose_bloq()
    with pytest.raises(ValueError, match="at least two"):
        SparseRotationPreparation((2,), (1,), 3, 2).requirements(2)
