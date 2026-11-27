"""Retain source integral filtering, scalar precision and executable table data."""

from importlib import import_module

from tests._reference_dependencies import unavailable_reference

try:
    import cirq
    import numpy as np
    import torch
    from numpy.typing import NDArray
    from qualtran._infra.gate_with_registers import get_named_qubits

    _source_prepare = import_module("qualtran.bloqs.chemistry.sparse.prepare")
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

import pytest

from flagquantum import Circuit
from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation
from flagquantum.algorithms.primitives.sparse_chemistry_data import (
    SparseChemistryData,
    sparse_chemistry_data,
)
from flagquantum.compiler.formats.legacy_component_resolution import ResolvedArgument
from flagquantum.compiler.formats.legacy_sparse_preparation_factories import (
    native_sparse_preparation_factories,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.preparation_reference import preparation_terminals
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_hubbard_hamiltonian_gqsp import _source_sparse_branch
from tests.unit.test_sparse_chemistry_prepare import _branch
from tests.unit.test_sparse_chemistry_prepare_component import (
    bind_used_workspace,
)
from tests.unit.test_sparse_chemistry_prepare_component import (
    frozen_sparse_preparation_source as frozen_sparse_preparation_source,
)
from tests.unit.test_sparse_chemistry_prepare_component import (
    reference_runtime as reference_runtime,
)

PrepareSparse = _source_prepare.PrepareSparse
get_sparse_inputs_from_integrals = _source_prepare.get_sparse_inputs_from_integrals

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _inputs() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    one = np.array([[1.1, 0.2], [0.2, 2.3]], dtype=np.float64)
    two = np.zeros((2,) * 4, dtype=np.float64)
    two[0, 0, 0, 0] = 0.1234567
    return one, two


def _arguments(data: SparseChemistryData) -> dict[str, ResolvedArgument]:
    return {
        "num_spin_orb": 2 * data.spatial_orbitals,
        "num_non_zero": len(data.indices),
        "num_bits_state_prep": data.probability_bits,
        "num_bits_rot_aa": data.rotation_bits,
        "sum_of_l1_coeffs": data.normalization,
        "log_block_size": data.log_block_size,
        "ind_pqrs": data.datasets[:4],
        "theta": data.datasets[4],
        "one_body": data.datasets[5],
        "alt_pqrs": data.datasets[6:10],
        "alt_theta": data.datasets[10],
        "alt_one_body": data.datasets[11],
        "keep": data.datasets[12],
    }


@pytest.mark.parametrize("single", (False, True))
@pytest.mark.parametrize(
    "threshold,rows",
    ((0.0, 4), (-1.0, 9), (float("inf"), 3), (float("-inf"), 9), (float("nan"), 3)),
)
def test_source_integral_precision_and_negative_threshold(
    single: bool, threshold: float, rows: int
) -> None:
    one, two = _inputs()
    inputs = (one.astype(np.float32), two.astype(np.float32)) if single else (one, two)
    source = PrepareSparse.from_hamiltonian_coeffs(
        4,
        inputs[0],
        inputs[1],
        num_bits_state_prep=8,
        drop_element_thresh=threshold,
        log_block_size=1,
    )
    data = sparse_chemistry_data(
        *inputs, probability_bits=8, drop_threshold=threshold, log_block_size=1
    )
    indices, coefficients = get_sparse_inputs_from_integrals(
        inputs[0], inputs[1], threshold
    )
    assert data.indices == tuple(map(tuple, indices.tolist()))
    assert data.coefficients == tuple(float(value) for value in coefficients)
    assert data.normalization == float(source.l1_norm_of_coeffs)
    assert data.datasets == (
        *source.ind_pqrs,
        source.theta,
        source.one_body,
        *source.alt_pqrs,
        source.alt_theta,
        source.alt_one_body,
        source.keep,
    )
    assert len(data.indices) == rows
    factories = native_sparse_preparation_factories()
    forward = factories["qualtran.bloqs.chemistry.sparse.prepare.PrepareSparse"](
        "main", None, _arguments(data)
    )
    assert isinstance(forward, ProgramRecord)
    inverse = factories["qualtran._infra.adjoint.Adjoint"](
        "main", None, {"subbloq": forward.functions[0]}
    )
    assert isinstance(inverse, ProgramRecord)
    for record, reference in ((forward, source), (inverse, source.adjoint())):
        compare_layers(
            record,
            reference,
            preparation_terminals(
                record,
                native_boundaries=(
                    f.name
                    for f in record.functions
                    if f.name.endswith("native_qroam_erase_leaf")
                ),
            ),
        )


def test_float32_eight_bit_cumulative_rounding_matches_source() -> None:
    a, b, c = np.float32(64.5 / 1024), np.float32(0.1), np.float32(0.2)
    last = np.nextafter(
        np.float32(1 - float(a) - float(b) - float(c)), np.float32(np.inf)
    )
    one = np.array([[a, 0], [b, c]], dtype=np.float32)
    two = np.zeros((2,) * 4, dtype=np.float32)
    two[0, 0, 0, 0] = last
    source = PrepareSparse.from_hamiltonian_coeffs(4, one, two, num_bits_state_prep=8)
    data = sparse_chemistry_data(one, two, probability_bits=8)
    assert data.datasets[-1] == source.keep
    assert data.datasets[6:10] == source.alt_pqrs
    assert data.normalization == float(source.l1_norm_of_coeffs)


def test_float32_threshold_uses_source_array_comparison() -> None:
    one, two = _inputs()
    narrow_one, narrow_two = one.astype(np.float32), two.astype(np.float32)
    threshold = float(narrow_two[0, 0, 0, 0]) - 1e-12
    indices, _ = get_sparse_inputs_from_integrals(narrow_one, narrow_two, threshold)
    data = sparse_chemistry_data(narrow_one, narrow_two, drop_threshold=threshold)
    assert data.indices == tuple(map(tuple, indices.tolist()))
    assert len(data.indices) == 3


def test_negative_threshold_source_complex_forward_and_joint_inverse() -> None:
    one, two = _inputs()
    source = PrepareSparse.from_hamiltonian_coeffs(
        4, one, two, num_bits_state_prep=2, drop_element_thresh=-1, log_block_size=1
    )
    data = sparse_chemistry_data(
        one, two, probability_bits=2, drop_threshold=-1, log_block_size=1
    )
    factories = native_sparse_preparation_factories()
    forward = factories["qualtran.bloqs.chemistry.sparse.prepare.PrepareSparse"](
        "main", None, _arguments(data)
    )
    assert isinstance(forward, ProgramRecord)
    backward = factories["qualtran._infra.adjoint.Adjoint"](
        "main", None, {"subbloq": forward.functions[0]}
    )
    assert isinstance(backward, ProgramRecord)
    bound, left = bind_used_workspace(forward, allow_workspace_outputs=True)
    appended = Circuit(bound.circuit.n_wires).append(
        _BoundBlockOperation(bound.circuit.instructions, bound.circuit.n_wires)
    )
    assert appended.to_ir().instructions == bound.circuit.instructions
    assert bound.circuit.from_json(bound.circuit.to_json()) == bound.circuit
    named = get_named_qubits(source.signature)
    visible = tuple(q for reg in source.signature for q in named[reg.name].flat)
    operations = list(
        cirq.decompose(
            source.on_registers(**named),
            keep=lambda op: (
                cirq.is_measurement(op)
                or isinstance(op, cirq.ClassicallyControlledOperation)
                or isinstance(op.gate, cirq.ResetChannel)
                or (len(op.qubits) <= 3 and cirq.has_unitary(op))
            ),
        )
    )
    initial = {0: 0.6 + 0j, 1: 0.8j}
    outputs = tuple(wire for word in bound.outputs for wire in word)
    hidden = set(range(bound.circuit.n_wires)) - set(outputs)
    hidden_mask = sum(1 << (bound.circuit.n_wires - 1 - wire) for wire in hidden)
    count = sum(gate.name == "measure" for gate in bound.circuit.instructions)
    for pattern in (0, 1):
        expected = _source_sparse_branch(
            operations,
            visible,
            {key << (len(visible) - left): value for key, value in initial.items()},
            pattern,
        )
        actual = _branch(
            bound.circuit.instructions,
            bound.circuit.n_wires,
            (pattern,) * count,
            {
                key << (bound.circuit.n_wires - left): value
                for key, value in initial.items()
            },
            normalize_measurements=True,
        )
        assert all(not key & hidden_mask for key in actual)
        decoded = {
            sum(
                1 << (len(outputs) - 1 - i)
                for i, wire in enumerate(outputs)
                if key & (1 << (bound.circuit.n_wires - 1 - wire))
            ): value
            for key, value in actual.items()
        }
        keys = sorted(expected.keys() | decoded.keys())
        torch.testing.assert_close(
            torch.tensor([decoded.get(key, 0) for key in keys], dtype=torch.complex128),
            torch.tensor(
                [expected.get(key, 0) for key in keys], dtype=torch.complex128
            ),
            atol=4e-10,
            rtol=0,
        )
    builder = ComponentProgramBuilder(
        Signature(tuple(Signature(forward.functions[0].registers or ()).lefts()))
    )
    restored = builder.call_program(
        backward, builder.call_program(forward, builder.inputs)
    )
    joint, logical = bind_used_workspace(builder.finalize(restored))
    target = {
        key << (joint.circuit.n_wires - logical): value
        for key, value in initial.items()
    }
    count = sum(gate.name == "measure" for gate in joint.circuit.instructions)
    for pattern in (0, 1):
        actual = _branch(
            joint.circuit.instructions,
            joint.circuit.n_wires,
            (pattern,) * count,
            target,
            normalize_measurements=True,
        )
        for key in actual.keys() | target.keys():
            assert actual.get(key, 0) == pytest.approx(target.get(key, 0), abs=4e-10)
