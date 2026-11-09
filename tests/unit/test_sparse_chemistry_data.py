"""Compare sparse integral representatives and all source QROAM datasets."""

import numpy as np
import pytest

from flagquantum.algorithms.primitives.sparse_chemistry_data import (
    sparse_chemistry_data,
)
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = pytest.mark.unit


def test_threshold_only_filters_two_body_integrals() -> None:
    one = np.asarray(((1.0, 0.0), (0.0, -2.0)))
    two = np.zeros((2,) * 4)
    two[0, 0, 0, 0] = 0.5
    two[1, 1, 1, 1] = -0.75
    data = sparse_chemistry_data(one, two, probability_bits=2, drop_threshold=0.5)
    assert data.indices == ((0, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 1, 1, 1))
    assert data.coefficients == (1, 0, -2, -0.75)
    assert data.datasets[4] == (0, 0, 1, 1)
    assert data.datasets[5] == (0, 0, 0, 1)
    assert data.normalization == 3.625
    assert data.address_width == 2


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    ("spatial", "bits", "threshold", "log"),
    [
        (1, 1, 0.0, 0),
        (2, 1, 0.0, 0),
        (2, 4, 0.0, 1),
        (3, 3, 1.0, None),
        (4, 8, 0.1, 1),
        (6, 4, 0.2, None),
        (7, 5, 5.0, None),
    ],
)
def test_integrals_and_lookup_cost_match_locked_source(
    spatial: int, bits: int, threshold: float, log: int | None
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.sparse.prepare import (
        PrepareSparse,
        get_sparse_inputs_from_integrals,
    )
    from qualtran.bloqs.chemistry.sparse.prepare_test import build_random_test_integrals

    one, two = build_random_test_integrals(spatial, seed=7)
    one[0, 0] = 0
    reference = PrepareSparse.from_hamiltonian_coeffs(
        2 * spatial, one, two, bits, 6, threshold, log
    )
    data = sparse_chemistry_data(
        one,
        two,
        probability_bits=bits,
        rotation_bits=6,
        drop_threshold=threshold,
        log_block_size=log,
    )
    indices, coefficients = get_sparse_inputs_from_integrals(one, two, threshold)
    np.testing.assert_array_equal(data.indices, indices)
    np.testing.assert_array_equal(data.coefficients, coefficients)
    assert data.normalization == reference.l1_norm_of_coeffs
    assert data.rotation_bits == reference.num_bits_rot_aa
    assert data.log_block_size == reference.log_block_size
    assert data.datasets == (
        *reference.ind_pqrs,
        reference.theta,
        reference.one_body,
        *reference.alt_pqrs,
        reference.alt_theta,
        reference.alt_one_body,
        reference.keep,
    )
    assert data.target_layout == tuple(
        (reg.name, reg.bitsize) for reg in reference.qroam_target_registers
    )
    if spatial == 1:
        with pytest.raises(ValueError, match="nonempty orbital words"):
            data.lookup_table()
    else:
        assert_source_model(data.lookup_cost(), reference.build_qrom_bloq())


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("log", [0, 1])
def test_named_lookup_matches_source_gate_stream(log: int) -> None:
    import torch

    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.sparse.prepare import PrepareSparse

    from flagquantum.compiler.registers.layout.partition import partition_register
    from flagquantum.core.ir import CircuitIR
    from flagquantum.dynamic import DynamicCircuit
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_register_subtraction import assert_source
    from tests.unit.test_source_qroam_clean import source_clean_graph

    one = np.asarray(((1.0, -0.75), (-0.75, 0.25)))
    two = np.zeros((2,) * 4)
    data = sparse_chemistry_data(one, two, probability_bits=1, log_block_size=log)
    prepare = PrepareSparse.from_hamiltonian_coeffs(
        4, one, two, num_bits_state_prep=1, log_block_size=log
    )
    source = prepare.build_qrom_bloq()
    assert data.lookup_output_layout == tuple(
        (reg.name, reg.total_bits())
        for reg in (
            *prepare.qroam_target_registers,
            *prepare.qroam_extra_target_registers,
        )
    )
    address = tuple(range(data.address_width))
    data_width = data.address_width + sum(
        width for _, width in data.lookup_output_layout
    )
    outputs = partition_register(
        tuple(range(data.address_width, data_width)), data.lookup_output_layout
    )
    work_size = data.lookup_table().workspace_size(0, (data.address_width,))
    work = tuple(range(data_width, data_width + work_size))
    bits = tuple(range(work_size))
    actual = data.lookup_ir(address, outputs, work, bits)
    circuit = DynamicCircuit(data_width + work_size).compose(
        CircuitIR(data_width + work_size, actual).validate()
    )
    source_outputs = {"selection": address}
    for i, (name, _) in enumerate(data.target_layout):
        source_outputs[f"target{i}_"] = outputs[name]
        if log:
            source_outputs[f"junk_target{i}_"] = outputs[f"jnk_{name}"]
    original, _, _ = source_clean_graph(
        source, {"selection": address}, source_outputs, work, bits
    )
    assert_source(circuit.to_ir().instructions, original)
    if log == 0:
        from itertools import product
        from math import sqrt

        initial = torch.zeros(1 << (data_width + work_size), dtype=torch.complex128)
        expected = torch.zeros_like(initial)
        for row in range(len(data.indices)):
            coefficient = complex(row + 1, 2 - row)
            initial[row << (data_width + work_size - len(address))] = coefficient
            index = row
            for column, (_, width) in zip(
                data.datasets, data.target_layout, strict=True
            ):
                index = (index << width) | column[row]
            expected[index << work_size] = coefficient
        measurements = sum(g.name == "measure" for g in actual)
        for history in product((0, 1), repeat=measurements):
            torch.testing.assert_close(
                forced_branch(initial, actual, history),
                expected / sqrt(2**measurements),
                atol=2e-12,
                rtol=0,
            )
