"""Check DF indexed metadata and the pinned QROAM gates and branch amplitudes."""

from itertools import product
from math import sqrt

import pytest

from flagquantum.algorithms.primitives.double_factorization_indexed_data import (
    DoubleFactorizationIndexedData,
)
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler.registers.partition import partition_register
from flagquantum.dynamic import DynamicCircuit
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_relative_controlled_swap import source_relative_swap
from tests.unit.test_source_qroam_clean import source_clean_graph
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("log", [0, 1])
def test_indexed_data_matches_locked_stream_and_restores_all_banks(log):
    pytest.importorskip("qualtran")
    import numpy as np
    from qualtran.bloqs.data_loading.qroam_clean import QROAMClean
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    data = DoubleFactorizationIndexedData((4, 2, 1, 4, 3), (1, 3, 5, 7, 0), 3, log)
    expected = ((0, 1, 1, 1, 1), (3, 1, 0, 3, 2), (1, 3, 5, 7, 0), (0, 4, 6, 7, 11))
    names = ("l_ne_zero", "xi", "rot_data", "offset")
    target_widths = (1, 2, 3, 4)
    source = QROAMClean.build_from_data(
        *(np.asarray(column) for column in expected),
        target_bitsizes=target_widths,
        log_block_sizes=(log,),
    )
    logical = sum(width for _, width in data.layout)
    ports = partition_register(tuple(range(logical)), data.layout)
    work = tuple(range(logical, logical + data.workspace_size()))
    width = logical + len(work)
    forward, inverse = (
        DynamicCircuit(width)
        .append(data.make_on(ports, workspace=work, adjoint=adjoint))
        .to_ir()
        for adjoint in (False, True)
    )
    inputs = {"selection": ports["l"]}
    outputs = inputs | {f"target{i}_": ports[name] for i, name in enumerate(names)}
    if log:
        outputs.update(
            {f"junk_target{i}_": ports[f"junk_{name}"] for i, name in enumerate(names)}
        )
    original, lookup_length, exchanges = source_clean_graph(
        source, inputs, outputs, work, tuple(range(len(work)))
    )
    assert_source(forward.instructions, original)
    source_inverse = (
        *(
            gate
            for exchange in reversed(exchanges)
            for gate in source_relative_swap(
                exchange["ctrl"][0], exchange["x"], exchange["y"]
            )
        ),
        *original[:lookup_length],
    )
    assert_source(inverse.instructions, source_inverse)
    counts = count_emitted_instructions(forward)
    gates = get_cost_value(source, QECGatesCost())
    assert counts.t_count == gates.total_t_count()
    events = counts.unconditional.get("measure", 0)
    assert events == gates.measurement
    initial = {
        row << (width - len(ports["l"])): complex(1, (-1) ** row) / sqrt(10)
        for row in range(5)
    }
    work_mask = (1 << len(work)) - 1
    for history in product((0, 1), repeat=events):
        loaded = _branch(
            forward.instructions, width, history, initial, normalize_measurements=True
        )
        assert loaded == pytest.approx(
            _branch(original, width, history, initial, normalize_measurements=True),
            abs=2e-12,
        )
        for basis in loaded:
            row = basis >> (width - len(ports["l"]))
            assert not basis & work_mask
            for name, column in zip(names, expected, strict=True):
                bits = ports[name]
                assert (basis >> (width - bits[-1] - 1)) & (
                    (1 << len(bits)) - 1
                ) == column[row]
        assert _branch(
            inverse.instructions, width, history, loaded, normalize_measurements=True
        ) == pytest.approx(initial, abs=2e-12)


def test_single_full_rank_one_body_has_no_rank_or_offset_wires():
    data = DoubleFactorizationIndexedData((1,), (1,), 1)
    assert data.layout == (("l", 0), ("l_ne_zero", 1), ("rot_data", 1))
    ports = partition_register((0, 1), data.layout)
    assert data.workspace_size() == 0
    circuit = DynamicCircuit(2).append(data.make_on(ports, workspace=()))
    assert _branch(circuit.to_ir().instructions, 2, ()) == {1: 1 + 0j}
