"""Preserve original THC preprocessing precision through constructor and lookup."""

from importlib import import_module
from itertools import product
from math import sqrt

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
    from numpy.typing import NDArray

    _source = import_module("qualtran.bloqs.chemistry.thc.prepare")
    _qroam = import_module("qualtran.bloqs.data_loading.qroam_clean")
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

import pytest

from flagquantum.algorithms.primitives.thc_chemistry_data import (
    THCChemistryData,
    thc_chemistry_data,
)
from flagquantum.compiler.formats.legacy_component_resolution import ResolvedArgument
from flagquantum.compiler.formats.legacy_thc_preparation_factories import (
    thc_preparation_factories,
)
from flagquantum.compiler.registers.layout.partition import partition_register
from flagquantum.compiler.registers.lookup.qroam_bank_binding import bind_clean_banks
from flagquantum.core.ir.program import ProgramRecord
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_source_qroam_clean import source_clean_graph
from tests.unit.test_sparse_chemistry_prepare import _branch
from tests.unit.test_sparse_chemistry_prepare_component import bind_used_workspace
from tests.unit.test_thc_preparation_source_component import thc_source_terminals
from tests.unit.test_thc_uniform_source_component import (
    pinned_thc_source as pinned_thc_source,
)
from tests.unit.test_thc_uniform_source_component import (
    reference_runtime as reference_runtime,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _witness() -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    a, b, c = np.float32(64.5 / 1024), np.float32(0.1), np.float32(0.2)
    last = np.nextafter(
        np.float32(1 - float(a) - float(b) - float(c)), np.float32(np.inf)
    )
    return (
        np.array([last], dtype=np.float32),
        np.array([[0.3], [0.7]], dtype=np.float32),
        np.array([[a, b], [b, c]], dtype=np.float32),
    )


def _arguments(data: THCChemistryData) -> dict[str, ResolvedArgument]:
    return {
        "num_mu": data.num_mu,
        "num_spin_orb": data.num_spin_orb,
        "keep_bitsize": data.probability_bits,
        "sum_of_l1_coeffs": data.normalization,
        "log_block_size": data.log_block_size,
        **dict(
            zip(
                ("theta", "alt_theta", "alt_mu", "alt_nu", "keep"),
                data.source_datasets,
                strict=True,
            )
        ),
    }


@pytest.mark.parametrize("single", (False, True))
@pytest.mark.parametrize("rank,spatial", ((2, 1), (3, 2), (8, 4)))
def test_original_integral_columns_norm_and_blocking(
    single: bool, rank: int, spatial: int
) -> None:
    rng = np.random.default_rng(7)
    dtype = np.float32 if single else np.float64
    one = rng.normal(size=spatial).astype(dtype)
    leaves = rng.normal(size=(rank, spatial)).astype(dtype)
    central = rng.normal(size=(rank, rank)).astype(dtype)
    source = _source.PrepareTHC.from_hamiltonian_coeffs(
        one, leaves, central, num_bits_state_prep=8
    )
    native = thc_chemistry_data(one, leaves, central, probability_bits=8)
    assert native.source_datasets == (
        source.theta,
        source.alt_theta,
        source.alt_mu,
        source.alt_nu,
        source.keep,
    )
    assert native.normalization == float(source.l1_norm_of_coeffs)
    assert native.log_block_size == source.log_block_size


def test_eight_bit_rounding_and_complete_source_constructor_path() -> None:
    inputs = _witness()
    source = _source.PrepareTHC.from_hamiltonian_coeffs(
        *inputs, num_bits_state_prep=8, log_block_size=1
    )
    native = thc_chemistry_data(*inputs, probability_bits=8, log_block_size=1)
    assert native.source_datasets[-1] == source.keep == (65, 102, 205, 0)
    assert native.normalization == float(source.l1_norm_of_coeffs)
    factories = thc_preparation_factories()
    forward = factories["qualtran.bloqs.chemistry.thc.prepare.PrepareTHC"](
        "main", None, _arguments(native)
    )
    assert isinstance(forward, ProgramRecord)
    inverse = factories["qualtran._infra.adjoint.Adjoint"](
        "main", None, {"subbloq": forward.functions[0]}
    )
    assert isinstance(inverse, ProgramRecord)
    for record, reference in ((forward, source), (inverse, source.adjoint())):
        compare_layers(record, reference, thc_source_terminals(record, 2, 2))
        # The literal root cannot execute an upstream model-only comparator.
        with pytest.raises(ValueError, match="requires component bodies"):
            bind_used_workspace(record, allow_workspace_outputs=True)


@pytest.mark.parametrize("log", (0, 1))
def test_actual_eight_bit_table_complex_lookup_all_histories_and_inverse(
    log: int,
) -> None:
    data = thc_chemistry_data(*_witness(), probability_bits=8, log_block_size=log)
    table = data.lookup_table()
    address = tuple(range(data.address_width))
    bank_width = sum(width for _, width in data.lookup_output_layout)
    logical = len(address) + bank_width
    outputs = partition_register(
        tuple(range(len(address), logical)), data.lookup_output_layout
    )
    work = table.workspace_size(0, (len(address),))
    scratch = tuple(range(logical, logical + work))
    bits = tuple(range(work))
    forward = data.lookup_ir(address, outputs, scratch, bits)
    inverse = data.lookup_ir(address, outputs, scratch, bits, uncompute=True)
    source = _qroam.QROAMClean.build_from_data(
        *data.address_datasets,
        target_bitsizes=tuple(width for _, width in data.target_layout),
        log_block_sizes=(log,),
    )
    banks = bind_clean_banks(outputs, data.target_layout, 1 << log, junk_prefix="junk_")
    source_inputs = {"selection": address}
    source_outputs = dict(source_inputs)
    for i, bank in enumerate(banks):
        source_outputs[f"target{i}_"] = bank[0]
        if log:
            source_outputs[f"junk_target{i}_"] = bank[1]
    reference, _, _ = source_clean_graph(
        source, source_inputs, source_outputs, scratch, bits
    )
    assert_source(forward, reference)
    width = logical + work + 1
    initial = {
        (row << (width - len(address))) | spectator: complex(row + 1, spectator - row)
        for row, spectator in product(range(4), (0, 1))
    }
    norm = sqrt(sum(abs(value) ** 2 for value in initial.values()))
    initial = {key: value / norm for key, value in initial.items()}
    count = sum(gate.name == "measure" for gate in forward)
    assert count == sum(gate.name == "measure" for gate in inverse)
    work_mask = sum(1 << (width - 1 - wire) for wire in scratch)
    for history in product((0, 1), repeat=count):
        actual = _branch(forward, width, history, initial)
        assert actual == pytest.approx(
            _branch(reference, width, history, initial), abs=2e-12
        )
        assert all(not key & work_mask for key in actual)
        for outcome in product((0, 1), repeat=count):
            restored = _branch(inverse, width, outcome, actual)
            assert restored == pytest.approx(
                {key: value * 2**-count for key, value in initial.items()}, abs=2e-12
            )
