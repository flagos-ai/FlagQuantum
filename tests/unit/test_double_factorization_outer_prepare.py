"""Check the DF factor distribution, the locked lookup stream and its inverse."""

from collections import defaultdict
from itertools import product
from math import acos, sin, sqrt

import pytest

from flagquantum.algorithms.primitives.double_factorization_outer_prepare import (
    PrepareDoubleFactorizationOuter,
)
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler.registers.uniform_preparation import (
    uniform_preparation_ir,
    uniform_preparation_requirements,
)
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_relative_controlled_swap import source_relative_swap
from tests.unit.test_single_factorization_outer_prepare import _circuits
from tests.unit.test_source_qroam_clean import source_clean_graph
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = pytest.mark.unit


def test_paper_eigenvalue_weights_match_explicit_factor_distribution():
    prepare = PrepareDoubleFactorizationOuter.from_eigenvalues(
        (1.0, -3.0, 2.0),
        ((3.0, -1.0), (-2.0,), (0.0, 0.0)),
        probability_bits=3,
        log_block_size=1,
    )
    assert prepare == PrepareDoubleFactorizationOuter.from_weights(
        (6.0, 4.0, 1.0, 0.0), probability_bits=3, log_block_size=1
    )
    assert prepare.normalization == 11.0


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("weights,log", [((1,), 0), ((4, 0, 2), 0), ((1, 3, 1, 7), 1)])
def test_factor_alias_complex_branches_and_clean_inverse(weights, log):
    pytest.importorskip("qualtran")
    import cirq
    import numpy as np
    from qualtran.bloqs.data_loading.qroam_clean import QROAMClean
    from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox
    from qualtran.linalg.lcu_util import (
        preprocess_probabilities_for_reversible_sampling,
    )

    prep = PrepareDoubleFactorizationOuter.from_weights(
        weights, probability_bits=1, log_block_size=log
    )
    (forward, inverse), work, ports = _circuits(prep)
    rows = len(weights)
    alternate, keep, _ = preprocess_probabilities_for_reversible_sampling(
        weights, sub_bit_precision=1
    )
    datasets = {"keep": keep}
    if rows > 1:
        datasets = {"alt_l": alternate, **datasets}
    widths = {name: len(ports[name]) for name in datasets}
    assert prep.table.datasets == tuple(tuple(data) for data in datasets.values())
    if rows > 1:
        source = QROAMClean.build_from_data(
            *(np.asarray(data) for data in datasets.values()),
            target_bitsizes=tuple(widths.values()),
            log_block_sizes=(log,),
        )
        lookup_work = prep.table.workspace_size(0, (len(ports["l"]),))
        logical = forward.n_wires - work - 1
        scratch = tuple(range(logical, logical + work))
        inputs = {"selection": ports["l"]}
        outputs = inputs | {
            f"target{i}_": ports[name] for i, name in enumerate(datasets)
        }
        if log:
            outputs.update(
                {
                    f"junk_target{i}_": ports[f"junk_{name}"]
                    for i, name in enumerate(datasets)
                }
            )
        expected_lookup, lookup_length, exchanges = source_clean_graph(
            source, inputs, outputs, scratch[:lookup_work], tuple(range(lookup_work))
        )
        _, uniform_work, uniform_slots = uniform_preparation_requirements(rows)
        initial_gates = uniform_preparation_ir(
            rows, ports["l"], scratch[:uniform_work], tuple(range(uniform_slots))
        )
        start = len(initial_gates) + len(ports["sigma"])
        assert_source(
            forward.instructions[start : start + len(expected_lookup)], expected_lookup
        )
        expected_unlookup = (
            *(
                gate
                for exchange in reversed(exchanges)
                for gate in source_relative_swap(
                    exchange["ctrl"][0], exchange["x"], exchange["y"]
                )
            ),
            *expected_lookup[:lookup_length],
        )
        final = len(inverse.instructions) - start
        assert_source(
            inverse.instructions[final - len(expected_unlookup) : final],
            expected_unlookup,
        )

    matrices = {}
    if log:
        for size in set(widths.values()):
            wires = cirq.LineQubit.range(1 + 2 * size)
            matrices[size] = cirq.unitary(
                cirq.Circuit(cirq.decompose(CSwapApprox(size).on(*wires)))
            )
    initial = {0: (1 + 1j) / 2, 1: (1 - 1j) / 2}
    expected = {}
    distribution = defaultdict(float)
    for row, sigma in product(range(rows), range(2)):
        values = dict(l=row, sigma=sigma, succ_l=1)
        phase = sqrt(0.75) * complex(2 / 3, sin(acos(1 / 3))) if rows == 3 else 1
        for name, column in datasets.items():
            if not log:
                values[name] = column[row]
                continue
            size = widths[name]
            left, right = column[row & ~1 : (row & ~1) + 2]
            basis = ((row & 1) << (2 * size)) | (left << size) | right
            vector = matrices[size][:, basis]
            result = int(np.argmax(np.abs(vector)))
            mask = (1 << size) - 1
            phase *= vector[result]
            values[name] = (result >> size) & mask
            values[f"junk_{name}"] = result & mask
        choose = keep[row] <= sigma
        values["choose_alt"] = int(choose)
        if choose and rows > 1:
            values["l"], values["alt_l"] = alternate[row], row
        distribution[values["l"]] += 1 / (2 * rows)
        word = 0
        for name, size in prep.layout:
            word = (word << size) | values[name]
        for spectator, coefficient in initial.items():
            expected[(word << (work + 1)) | spectator] = (
                phase * coefficient / sqrt(2 * rows)
            )
    assert [distribution[row] for row in range(rows)] == pytest.approx(
        [value / sum(weights) for value in weights], abs=1 / (2 * rows)
    )
    for ir, before, after in (
        (forward, initial, expected),
        (inverse, expected, initial),
    ):
        counts = count_emitted_instructions(ir)
        events = counts.unconditional.get("measure", 0)
        assert (counts.t_count, events, work) == {
            1: (4, 1, 2),
            3: (38, 6, 4),
            4: (30, 1, 2),
        }[rows]
        for history in product((0, 1), repeat=events):
            assert _branch(
                ir.instructions,
                ir.n_wires,
                history,
                before,
                normalize_measurements=True,
            ) == pytest.approx(after, abs=2e-12)
