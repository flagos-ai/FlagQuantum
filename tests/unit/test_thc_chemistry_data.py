"""THC alias row order, named QROAM dependencies, and complete complex branches."""

from itertools import product
from math import sqrt

import numpy as np
import pytest

from flagquantum.algorithms.primitives.thc_chemistry_data import thc_chemistry_data
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler.registers.layout.partition import partition_register
from flagquantum.compiler.registers.lookup.qroam_bank_binding import bind_clean_banks
from flagquantum.core.ir import CircuitIR
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_relative_controlled_swap import source_relative_swap
from tests.unit.test_source_qroam_clean import source_clean_graph
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = pytest.mark.unit


def _tensors(m, spatial):
    rng = np.random.default_rng(17)
    one = rng.normal(size=spatial)
    eta = rng.normal(size=(m, spatial))
    raw = rng.normal(size=(m, m))
    return one, eta, (raw + raw.T) / 2


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "m,spatial,bits,log",
    [
        (1, 1, 1, None),
        (2, 2, 2, 0),
        (3, 2, 4, 1),
        (4, 2, 8, None),
        (8, 4, 12, None),
        (10, 4, 4, None),
    ],
)
def test_alias_columns_normalization_and_address_permutation(m, spatial, bits, log):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.arithmetic.conversions.contiguous_index import ToContiguousIndex
    from qualtran.bloqs.chemistry.thc.prepare import PrepareTHC

    tensors = _tensors(m, spatial)
    actual = thc_chemistry_data(*tensors, probability_bits=bits, log_block_size=log)
    source = PrepareTHC.from_hamiltonian_coeffs(
        *tensors, num_bits_state_prep=bits, log_block_size=log
    )
    assert actual.source_datasets == (
        source.theta,
        source.alt_theta,
        source.alt_mu,
        source.alt_nu,
        source.keep,
    )
    assert actual.normalization == source.l1_norm_of_coeffs
    assert actual.log_block_size == source.log_block_size
    assert actual.target_layout == tuple(
        (register.name, register.total_bits())
        for register in source.qroam_target_registers
    )
    index = ToContiguousIndex(m.bit_length(), actual.address_width)
    for i, (mu, nu) in enumerate(actual.source_rows):
        address = index.on_classical_vals(mu=mu, nu=nu)["s"]
        assert tuple(column[address] for column in actual.address_datasets) == tuple(
            column[i] for column in actual.source_datasets
        )
    assert_source_model(actual.lookup_cost(source_order=True), source.build_qrom_bloq())


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("m,log", [(2, 0), (3, 1)])
def test_bound_contiguous_qroam_against_pinned_leaves_and_all_branches(m, log):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.data_loading.qroam_clean import QROAMClean

    # Nonconstant data makes the target bits and the retained banks observable.
    data = thc_chemistry_data(*_tensors(m, 2), probability_bits=2, log_block_size=log)
    if m == 3:
        assert data.address_datasets != data.source_datasets
    table = data.lookup_table()
    addr = tuple(range(data.address_width))
    bank_width = sum(width for _, width in data.lookup_output_layout)
    logical = len(addr) + bank_width
    outputs = partition_register(
        tuple(range(len(addr), logical)), data.lookup_output_layout
    )
    work = table.workspace_size(0, (len(addr),))
    scratch = tuple(range(logical, logical + work))
    bits = tuple(range(20, 20 + work))
    forward = data.lookup_ir(addr, outputs, scratch, bits)
    inverse = data.lookup_ir(addr, outputs, scratch, bits, uncompute=True)

    # Instantiate the unmodified source QROAM with the permuted input columns.
    source = QROAMClean.build_from_data(
        *data.address_datasets,
        target_bitsizes=tuple(width for _, width in data.target_layout),
        log_block_sizes=(log,),
    )
    banks = bind_clean_banks(outputs, data.target_layout, 1 << log, junk_prefix="junk_")
    source_inputs = {"selection": addr}
    source_outputs = dict(source_inputs)
    for i, bank in enumerate(banks):
        source_outputs[f"target{i}_"] = bank[0]
        if len(bank) > 1:
            source_outputs[f"junk_target{i}_"] = tuple(
                wire for cell in bank[1:] for wire in cell
            )
    reference, load_length, swaps = source_clean_graph(
        source, source_inputs, source_outputs, scratch, bits
    )
    assert_source(forward, reference)
    reverse = (
        *(
            gate
            for swap in reversed(swaps)
            for gate in source_relative_swap(swap["ctrl"][0], swap["x"], swap["y"])
        ),
        *reference[:load_length],
    )
    assert_source(inverse, reverse)
    assert_source_model(data.lookup_cost(), source)

    width = logical + work + 1  # A coherent spectator.
    counts = count_emitted_instructions(CircuitIR(width, forward).validate())
    events = counts.unconditional.get("measure", 0)
    assert events == sum(gate.name == "measure" for gate in inverse)
    assert counts.t_count > 0
    logical_input = {
        (index << (width - len(addr)))
        | spectator: complex(index + 1, 2 * spectator - index)
        for index, spectator in product(range(1 << len(addr)), (0, 1))
    }
    norm = sqrt(sum(abs(value) ** 2 for value in logical_input.values()))
    logical_input = {index: value / norm for index, value in logical_input.items()}
    for history in product((0, 1), repeat=events):
        state = _branch(forward, width, history, logical_input)
        assert state == pytest.approx(
            _branch(reference, width, history, logical_input), abs=2e-12
        )
        for index in state:
            address = index >> (width - len(addr))
            if address < len(data.source_rows):
                for (name, _), column in zip(
                    data.target_layout, data.address_datasets, strict=True
                ):
                    loaded = 0
                    for wire in outputs[name]:
                        loaded = (loaded << 1) | ((index >> (width - 1 - wire)) & 1)
                    assert loaded == column[address]
            assert not index & (((1 << work) - 1) << 1)
        for outcome in product((0, 1), repeat=events):
            restored = _branch(inverse, width, outcome, state)
            assert restored == pytest.approx(
                {index: value * 2**-events for index, value in logical_input.items()},
                abs=2e-12,
            )


@pytest.mark.parametrize(
    "one,eta,zeta,options,error",
    [
        ([], [[1]], [[1]], {}, "nonempty vector"),
        ([1], [[1]], [[1, 0]], {}, "square central"),
        ([1], [[1, 0]], [[1]], {}, "leaf shape"),
        ([1j], [[1]], [[1]], {}, "real and finite"),
        ([1], [[float("inf")]], [[1]], {}, "real and finite"),
        ([1], [[1]], [[1]], {"probability_bits": 0}, "bit precision"),
        ([1], [[1]], [[1]], {"log_block_size": -1}, "block exponent"),
    ],
)
def test_thc_alias_data_validates_shape_and_precision(one, eta, zeta, options, error):
    with pytest.raises(ValueError, match=error):
        thc_chemistry_data(one, eta, zeta, **options)
