# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Includes the Qualtran v0.7.0 sparse rotation preparation test fixture.

"""Concrete sparse rotation preparation graph, amplitudes, and costs."""

from cmath import exp
from collections import Counter
from hashlib import sha1
from importlib.metadata import version
from itertools import product
from math import sqrt, tau
from pathlib import Path
from typing import Any

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as numpy
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

import pytest
import torch

from examples.algorithms.sparse_rotation_preparation_catalog import (
    SparseRotationExample,
    sparse_rotation_example,
)
from flagquantum.algorithms.primitives.sparse_rotation_preparation import (
    SparseRotationPreparation,
)
from flagquantum.circuit import Circuit
from flagquantum.core.ir import Instruction
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
from tests.unit.test_rotation_resource_model import assert_source_cost

pytestmark = pytest.mark.unit


def assert_sparse_source_stream(
    source: Any, native: SparseRotationPreparation, width: int
) -> tuple[tuple[Instruction, ...], int]:
    try:
        import cirq
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)
    try:
        import numpy as np
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)
    from qualtran.bloqs.arithmetic.permutation import Permutation
    from qualtran.bloqs.bookkeeping.partition import Partition
    from qualtran.bloqs.state_preparation.state_preparation_via_rotation import (
        StatePreparationViaRotations,
    )

    phase_width = native.phase_width
    work_size, slot_count = native.requirements(width)
    target = tuple(range(width))
    gradient = tuple(range(width, width + phase_width))
    data_size = width + phase_width
    work = tuple(range(data_size, data_size + work_size))
    bits = tuple(range(slot_count))
    ports = {"target_state": target, "phase_gradient": gradient}
    example = SparseRotationExample(native, width)
    circuit = Circuit(example.qubit_count)
    circuit.append(example)
    actual = circuit.to_ir().instructions
    assert actual == native.emit(target, gradient, work, bits)
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
            assert isinstance(bloq.cycles, tuple)
            cycles: list[tuple[int, ...]] = []
            for cycle in bloq.cycles:
                assert isinstance(cycle, tuple)
                cycles.append(cycle)
            allocation_plan = tuple(
                wire
                for cycle in cycles
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
        tuple(range(len(native.indices))), native.coefficients, native.size, phase_width
    ).requirements(width)
    expected, returned = walk_source_graph(source, ports, ports, leaf)
    assert dense_work <= work_size
    assert returned == ports and children == ["split", "dense", "join", "permutation"]
    assert_source(actual, expected)
    assert [g for g in actual if g.name == "rx"] == [
        g for g in expected if g.name == "rx"
    ]
    return actual, work_size


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
    from qualtran.bloqs.state_preparation.sparse_state_preparation_via_rotations import (
        SparseStatePreparationViaRotations,
    )
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    source = SparseStatePreparationViaRotations(
        indices, coefficients, size, phase_width, width
    )
    native = SparseRotationPreparation.from_coefficient_map(
        size, dict(zip(indices, coefficients, strict=True)), phase_width
    )
    assert native.indices == source.sparse_indices
    actual, work_size = assert_sparse_source_stream(source, native, width)
    data_size = width + phase_width
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
    from qualtran._infra.composite_bloq import BloqError
    from qualtran.bloqs.state_preparation.sparse_state_preparation_via_rotations import (
        SparseStatePreparationViaRotations,
    )

    broken = SparseStatePreparationViaRotations((3, 0), (1, 1), 4, 2)
    assert broken._basis_permutation_bloq.cycles == ((0, 3),)
    with pytest.raises(BloqError, match="does not accept Soquets"):
        SparseStatePreparationViaRotations((2,), (1,), 3, 2).decompose_bloq()
    with pytest.raises(ValueError, match="at least two"):
        SparseRotationPreparation((2,), (1,), 3, 2).requirements(2)


def test_completed_prefix_prepares_requested_complex_state_on_every_branch() -> None:
    example = SparseRotationExample(SparseRotationPreparation((3, 0), (1, 1j), 4, 3), 2)
    work, _ = example.preparation.requirements(2)
    circuit = Circuit(example.qubit_count).append(example)
    gates = circuit.to_ir().instructions
    phase_gradient = torch.tensor(
        [exp(-1j * tau * k / 8) / sqrt(8) for k in range(8)], dtype=torch.complex128
    )
    logical = torch.tensor([1j, 0, 0, 1], dtype=torch.complex128) / sqrt(2)
    initial = torch.zeros(1 << example.qubit_count, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    initial[: 8 << work : 1 << work] = phase_gradient
    expected[:: 1 << work] = torch.kron(logical, phase_gradient)
    events = sum(gate.name == "measure" for gate in gates)
    assert events > 0
    for history in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(initial, gates, history),
            expected / 2 ** (events / 2),
            rtol=0,
            atol=2e-12,
        )


@pytest.fixture(scope="module")
def locked_sparse_source() -> Any:
    import kickmix
    from qualtran.bloqs.state_preparation import sparse_state_preparation_via_rotations

    assert version("qualtran") == "0.7.0"
    assert getattr(kickmix, "__version__") == "0.1" and hasattr(kickmix, "Simulator")
    source = Path(sparse_state_preparation_via_rotations.__file__)
    for path, digest in (
        (source, "e5d631263c199129e23c246bfab0fd36fd5f774a"),
        (
            source.with_name(f"{source.stem}_test.py"),
            "ea6a6748f0837615f45e7e6082188b5cc7cfc1d9",
        ),
    ):
        raw = path.read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return sparse_state_preparation_via_rotations


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("wide", (False, True))
def test_original_sparse_factories(locked_sparse_source: Any, wide: bool) -> None:
    source = (
        locked_sparse_source._sparse_state_prep_via_rotations_with_large_target_bitsize()
        if wide
        else locked_sparse_source._sparse_state_prep_via_rotations()
    )
    example = sparse_rotation_example(wide=wide)
    native = example.preparation
    assert (
        native.indices,
        native.coefficients,
        native.size,
        native.phase_width,
        example.target_width,
    ) == (
        source.sparse_indices,
        source.nonzero_coeffs,
        source.N,
        source.phase_bitsize,
        source.target_bitsize,
    )
    assert [(port.name, port.bitsize, port.side.name) for port in source.signature] == [
        ("target_state", example.target_width, "THRU"),
        ("phase_gradient", 2, "THRU"),
    ]
    gates, work = assert_sparse_source_stream(source, native, example.target_width)
    assert_source_cost(example.resource_model(), source)
    assert not any(
        wire >= example.data_width + work for gate in gates for wire in gate.wires
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("target_width", (None, 4, 6))
def test_original_sparse_state_grid(
    locked_sparse_source: Any, target_width: int | None
) -> None:
    import attrs

    try:
        import numpy as np
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)
    from qualtran.bloqs.state_preparation.sparse_state_preparation_via_rotations_test import (
        get_prepared_state_vector,
    )

    # Preserve the original floating-point coefficients for angle quantization.
    amplitudes = (
        (-0.42677669529663675 - 0.1767766952966366j),
        0,
        (0.17677669529663664 - 0.4267766952966367j),
        (0.17677669529663675 - 0.1767766952966368j),
        0,
        0,
        (0.07322330470336305 - 0.07322330470336309j),
        (0.4267766952966366 - 0.17677669529663692j),
        0,
        (0.42677669529663664 + 0.17677669529663675j),
        0,
        (0.0732233047033631 + 0.17677669529663678j),
        (-0.07322330470336308 - 0.17677669529663678j),
        0,
    )
    source = locked_sparse_source.SparseStatePreparationViaRotations.from_sparse_array(
        amplitudes, 3
    )
    if target_width is not None:
        source = attrs.evolve(source, target_bitsize=target_width)
    width = int(source.target_bitsize)
    native = SparseRotationPreparation.from_sparse_array(amplitudes, 3)
    assert native.indices == source.sparse_indices
    assert native.coefficients == source.nonzero_coeffs
    example = SparseRotationExample(native, width)
    gates, work = assert_sparse_source_stream(source, native, width)
    assert_source_cost(example.resource_model(), source)
    original = np.pad(amplitudes, (0, (1 << width) - len(amplitudes)))
    result = get_prepared_state_vector(source)
    np.testing.assert_allclose(result, original, atol=2e-12, rtol=0)

    gradient = tuple(exp(-1j * tau * i / 8) / sqrt(8) for i in range(8))
    input_state = {i << work: phase for i, phase in enumerate(gradient)}
    expected = {
        ((address << 3) | i) << work: complex(value) * phase
        for address, value in enumerate(result)
        if abs(value) > 2e-12
        for i, phase in enumerate(gradient)
    }
    for seed in (0, 1, 27):
        actual = execute_rotation_sparse(gates, example.qubit_count, input_state, seed)
        assert (
            max(
                abs(actual.get(index, 0) - expected.get(index, 0))
                for index in actual.keys() | expected.keys()
            )
            < 2e-12
        )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("dtype", ["float32", "complex64", "complex128"])
def test_coefficient_vector_keeps_source_order_dtype_and_gates(
    locked_sparse_source: Any, dtype: str
) -> None:
    try:
        import numpy as np
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)

    values = [0.6, 0.8, 0.0] if dtype == "float32" else [0.6, 0.8j, 0.0]
    vector = np.asarray(values, dtype=dtype)
    source = locked_sparse_source.SparseStatePreparationViaRotations.from_sparse_array(
        vector, 2
    )
    native = SparseRotationPreparation.from_sparse_array(vector, 2)
    assert (native.size, native.indices, native.target_width) == (
        source.N,
        source.sparse_indices,
        source.target_bitsize,
    )
    assert (
        np.asarray(native.coefficients).dtype == np.asarray(source.nonzero_coeffs).dtype
    )
    assert native.coefficients == source.nonzero_coeffs
    dense, _, _ = native._components(native.target_width)
    assert dense.coefficients == source._dense_stateprep_bloq.state_coefficients
    assert_sparse_source_stream(source, native, native.target_width)
    assert_source_cost(
        SparseRotationExample(native, native.target_width).resource_model(), source
    )


@pytest.mark.ftqc_reference
def test_sparse_vector_shape_failures_differ_from_valid_flat_vectors(
    locked_sparse_source: Any,
) -> None:
    try:
        import numpy as np
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)
    from scipy.sparse import csr_array

    source_factory = locked_sparse_source.SparseStatePreparationViaRotations
    row = np.array([[0.0, 0.6, 0.0, 0.8]])
    malformed = source_factory.from_sparse_array(row, 2)
    assert (malformed.N, malformed.target_bitsize, malformed.sparse_indices) == (
        1,
        0,
        (1, 3),
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        SparseRotationPreparation.from_sparse_array(row, 2)
    sparse_row = csr_array(row)
    with pytest.raises(TypeError, match="length is ambiguous"):
        source_factory.from_sparse_array(sparse_row, 2)
    with pytest.raises(TypeError, match="length is ambiguous"):
        SparseRotationPreparation.from_sparse_array(sparse_row, 2)
