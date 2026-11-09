"""Alias PREPARE source composition, coherent samples and logical junk."""

from collections import Counter
from itertools import product
from math import acos, sin, sqrt

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives.alias_preparation import (
    alias_preparation_ir,
    alias_preparation_requirements,
)
from flagquantum.algorithms.primitives.alias_sampling import (
    alias_sampling_table,
    sparse_alias_sampling_table,
)
from flagquantum.compiler.registers.comparison.less_equal import (
    less_equal_ir,
    less_equal_requirements,
)
from flagquantum.compiler.registers.lookup.source_qrom import source_qrom_workspace_size
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_source_qrom import source_qrom_gates

pytestmark = pytest.mark.unit


def layout(table):
    width = (table.selection_size - 1).bit_length()
    address = (len(table.keep) - 1).bit_length() if table.indices is not None else 0
    work, slots = alias_preparation_requirements(table)
    cursor = iter(range(2 * width + 2 * table.mu + address + 1 + work))

    def take(n):
        return tuple(next(cursor) for _ in range(n))

    selection, sigma, sparse, alt, keep = (
        take(n) for n in (width, table.mu, address, width, table.mu)
    )
    flag = next(cursor)
    return (selection, sigma, alt, keep, flag, take(work), tuple(range(slots))), sparse


def expected_state(table, bound, sparse):
    sel, sigma, alt, keep, flag, work, _ = bound
    logical_size = flag + 1
    out = torch.zeros(1 << (logical_size + len(work)), dtype=torch.complex128)
    rows = len(table.keep)
    phase = 1
    odd = rows // (rows & -rows)
    if odd != 1:
        ratio = odd / (1 << (odd - 1).bit_length())
        phase = sqrt(ratio) * complex(1 / (2 * ratio), sin(acos(1 - 1 / (2 * ratio))))
    for row, sample in product(range(rows), range(1 << table.mu)):
        selected = table.indices[row] if table.indices is not None else row
        other = table.alternates[row]
        swap = table.keep[row] <= sample
        if swap:
            selected, other = other, selected
        parts = (
            (sel, selected),
            (sigma, sample),
            (sparse, row),
            (alt, other),
            (keep, table.keep[row]),
            ((flag,), int(swap)),
        )
        basis = sum(
            ((value >> (len(word) - i - 1)) & 1) << (logical_size + len(work) - q - 1)
            for word, value in parts
            for i, q in enumerate(word)
        )
        out[basis] += phase / sqrt(rows * (1 << table.mu))
    return out


@pytest.mark.parametrize(
    "table",
    [
        alias_sampling_table((1,), precision=1),
        alias_sampling_table((1, 3), precision=1),
        alias_sampling_table((0, 1, 2), precision=0.5),
        sparse_alias_sampling_table({3: 1}, 4, precision=1),
        sparse_alias_sampling_table({3: 1, 0: 3}, 4, precision=0.25),
    ],
)
def test_prepare_restore_logical_junk_and_all_small_histories(table):
    bound, sparse = layout(table)
    expected = expected_state(table, bound, sparse)
    size = expected.numel().bit_length() - 1
    initial = torch.zeros_like(expected)
    initial[0] = 1
    for adjoint, state, desired in (
        (False, initial, expected),
        (True, expected, initial),
    ):
        gates = alias_preparation_ir(
            table, *bound, sparse_index=sparse, adjoint=adjoint
        )
        circuit = DynamicCircuit(size, inputs=state, dtype=torch.complex128)
        circuit.append(FieldOperation(size, gates))
        stream = tuple(circuit.to_ir().instructions)
        events = sum(g.name == "measure" for g in stream)
        if events <= 4:
            for history in product((0, 1), repeat=events):
                torch.testing.assert_close(
                    forced_branch(state, stream, history),
                    desired / sqrt(2**events),
                    atol=3e-12,
                    rtol=0,
                )
        else:
            actual = run_dynamic(circuit, shots=8, seed=71, strategy="trajectory")
            torch.testing.assert_close(
                actual.final_states.reshape(8, -1),
                desired.expand(8, -1),
                atol=3e-12,
                rtol=0,
            )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("sparse_mode", [False, True])
def test_source_parent_graph_leaves_and_complex_output(sparse_mode):
    pytest.importorskip("qualtran")
    import cirq
    import numpy as np
    from qualtran.bloqs.arithmetic import LessThanEqual
    from qualtran.bloqs.basic_gates import CSwap, OnEach
    from qualtran.bloqs.data_loading.qrom import QROM
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )
    from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
        SparseStatePreparationAliasSampling,
        StatePreparationAliasSampling,
    )
    from qualtran.cirq_interop.testing import GateHelper
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    if sparse_mode:
        weights = {3: 1, 0: 3}
        table = sparse_alias_sampling_table(weights, 4, precision=0.25)
        source = SparseStatePreparationAliasSampling.from_sparse_dict(
            weights, 4, precision=0.25
        )
    else:
        table = alias_sampling_table((0, 1, 2), precision=0.5)
        source = StatePreparationAliasSampling.from_probabilities(
            [0, 1, 2], precision=0.5
        )
    bound, sparse = layout(table)
    selection, sigma, alt, keep, flag, workspace, bits = bound
    source_stages = sorted(source.decompose_bloq().bloq_instances, key=lambda b: b.i)
    assert tuple(type(stage.bloq) for stage in source_stages) == (
        PrepareUniformSuperposition,
        OnEach,
        QROM,
        LessThanEqual,
        CSwap,
    )

    gates = alias_preparation_ir(table, *bound, sparse_index=sparse)
    datasets = tuple(tuple(map(int, data)) for data in source.qrom_bloq.data)
    address = sparse if sparse_mode else selection
    lookup_size = source_qrom_workspace_size(0, (len(address),), datasets)
    ports = {"selection": address}
    targets = (selection, alt, keep) if sparse_mode else (alt, keep)
    ports.update({f"target{i}_": word for i, word in enumerate(targets)})
    lookup = source_qrom_gates(
        source.qrom_bloq, ports, workspace[:lookup_size], bits[:lookup_size]
    )
    size_cmp, slot_cmp = less_equal_requirements(table.mu, table.mu)
    swap = reference_arithmetic_gates(
        source_stages[4].bloq,
        {"ctrl": (flag,), "x": alt, "y": selection},
        (),
        {},
        preserve_components=True,
    )
    compare_native = less_equal_ir(
        keep,
        sigma,
        flag,
        workspace[:size_cmp],
        bits[:slot_cmp],
    )
    measured = {
        g.wires[0]: g.metadata["classical_bit"]
        for g in compare_native
        if g.name == "measure"
    }
    compare = reference_arithmetic_gates(
        source_stages[3].bloq,
        {"x": keep, "y": sigma, "target": (flag,)},
        workspace[:size_cmp],
        measured,
        cirq_style=True,
    )
    compare_start = len(gates) - len(swap) - len(compare)
    assert_source(gates[-len(swap) :], swap)
    assert_source(gates[compare_start : compare_start + len(compare)], compare)
    assert_source(gates[compare_start - len(lookup) : compare_start], lookup)
    counts = Counter(g.name for g in gates)
    rows = len(table.keep)
    high = ((rows // (rows & -rows)) - 1).bit_length()
    uniform_and = 3 * high - 1 if high else 0
    lookup_meas = sum(g.name == "measure" for g in lookup)
    assert counts["measure"] == uniform_and + lookup_meas + slot_cmp
    assert counts["t"] + counts["tdg"] == 4 * counts["measure"] + 7 * len(selection)

    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count(
        ts_per_cswap=7, ts_per_rotation=0
    )
    assert counts["measure"] == model.measurement
    assert counts["rz"] == model.rotation

    helper = GateHelper(source)
    expected = expected_state(table, bound, sparse).reshape(1 << (flag + 1), -1)[:, 0]
    reference = (
        cirq.Simulator(dtype=np.complex128)
        .simulate(
            helper.circuit,
            qubit_order=helper.operation.qubits,
        )
        .final_state_vector
    )
    torch.testing.assert_close(
        torch.from_numpy(reference), expected, atol=3e-12, rtol=0
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    ("weights", "precision"),
    [((1, 3), 1.0), ((1, 1, 1), 0.25), ((1, 2, 3), 0.5), ((1, 2, 3, 4), 0.3)],
)
def test_dense_alias_resource_model(weights, precision):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.state_preparation import StatePreparationAliasSampling

    from flagquantum.algorithms.primitives.alias_preparation import (
        source_dense_alias_preparation_cost,
    )
    from tests.unit.test_first_quantization_resource_model import assert_source_model

    table = alias_sampling_table(weights, precision=precision)
    source = StatePreparationAliasSampling.from_probabilities(
        weights, precision=precision
    )
    model = source_dense_alias_preparation_cost(table)
    assert_source_model(model, source)
    assert_source_model(model, source.adjoint())


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    ("weights", "size", "precision"),
    [
        ({3: 1, 0: 3}, 4, 0.25),
        ({6: 1, 2: 2, 0: 4}, 7, 0.25),
        ({0: 1, 3: 1, 9: 3, 1: 2}, 10, 0.125),
        ({15: 4, 1: 1}, 16, 0.125),
    ],
)
def test_sparse_alias_resource_model(weights, size, precision):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
        SparseStatePreparationAliasSampling,
    )

    from flagquantum.algorithms.primitives.alias_preparation import (
        source_sparse_alias_preparation_cost,
    )
    from tests.unit.test_first_quantization_resource_model import assert_source_model

    table = sparse_alias_sampling_table(weights, size, precision=precision)
    source = SparseStatePreparationAliasSampling.from_sparse_dict(
        weights, size, precision=precision
    )
    model = source_sparse_alias_preparation_cost(table)
    assert_source_model(model, source)
    assert_source_model(model, source.adjoint())


@pytest.mark.ftqc_reference
def test_sparse_singleton_has_no_source_cost():
    pytest.importorskip("qualtran")
    from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
        SparseStatePreparationAliasSampling,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from flagquantum.algorithms.primitives.alias_preparation import (
        source_sparse_alias_preparation_cost,
    )

    table = sparse_alias_sampling_table({3: 1}, 4, precision=1)
    source = SparseStatePreparationAliasSampling.from_sparse_dict(
        {3: 1}, 4, precision=1
    )
    with pytest.raises(KeyError, match="sparse_index"):
        get_cost_value(source, QECGatesCost())
    with pytest.raises(ValueError, match="at least two support rows"):
        source_sparse_alias_preparation_cost(table)
