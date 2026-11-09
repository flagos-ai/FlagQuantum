"""Validate the B8 alias marginal and the full outer PREPARE branches."""

from collections import defaultdict
from itertools import product
from math import acos, sin, sqrt

import pytest

from flagquantum.algorithms.primitives.single_factorization_outer_prepare import (
    PrepareSingleFactorizationOuter,
)
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler.registers.layout.partition import partition_register
from flagquantum.dynamic import DynamicCircuit
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = pytest.mark.unit


def _prepare(count, log):
    return PrepareSingleFactorizationOuter.from_coefficients(
        (0.0, -0.5, 0.0),
        tuple(
            (0.0, -value, 0.0)
            for value in ((0,) if count == 2 else (1, 2, 1)[: count - 1])
        ),
        probability_bits=1,
        log_block_size=log,
    )


def _circuits(prepare):
    logical = sum(size for _, size in prepare.layout)
    work, _ = prepare.requirements()
    width = logical + work + 1
    ports = partition_register(tuple(range(logical)), prepare.layout)
    scratch = tuple(range(logical, logical + work))
    circuits = tuple(
        DynamicCircuit(width)
        .append(prepare.make_on(ports, workspace=scratch, adjoint=adjoint))
        .to_ir()
        for adjoint in (False, True)
    )
    return circuits, work, ports


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("rows,log", [(1, 0), (2, 0), (3, 0), (4, 1)])
def test_outer_prepare_against_locked_lookup_and_paper_weights(rows, log):
    pytest.importorskip("qualtran")
    import cirq
    import numpy as np
    from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox
    from qualtran.linalg.lcu_util import (
        preprocess_probabilities_for_reversible_sampling,
    )

    prepare = _prepare(rows, log)
    (forward, inverse), work, _ = _circuits(prepare)
    weights = (1, 0) if rows == 2 else (1, 1, 4, 1)[:rows]
    alternate, keep, _ = preprocess_probabilities_for_reversible_sampling(
        weights, sub_bit_precision=1
    )
    assert prepare.normalization == sum(weights)
    targets = [
        (name, size)
        for name, size in prepare.layout[4:]
        if not name.startswith("junk_")
    ]
    datasets = dict(
        zip((name for name, _ in targets), prepare.table.datasets, strict=True)
    )
    if rows > 1:
        assert datasets["alt_l"] == tuple(alternate)
    assert datasets["keep"] == tuple(keep)
    assert datasets["l_ne_zero"] == (0,) + (1,) * (rows - 1)
    assert datasets["alt_l_ne_zero"] == tuple(int(value != 0) for value in alternate)

    matrices = {}
    if log:
        for _, size in targets:
            wires = cirq.LineQubit.range(1 + 2 * size)
            matrices[size] = cirq.unitary(
                cirq.Circuit(cirq.decompose(CSwapApprox(size).on(*wires)))
            )
    initial = {0: (1 + 1j) / 2, 1: (1 - 1j) / 2}
    expected = {}
    marginal = defaultdict(float)
    for row, sigma in product(range(rows), range(2)):
        values = {"l": row, "sigma": sigma, "succ_l": 1}
        phase = sqrt(0.75) * complex(2 / 3, sin(acos(1 / 3))) if rows == 3 else 1
        for name, size in targets:
            if log:
                start = row & ~1
                left, right = datasets[name][start : start + 2]
                basis = ((row & 1) << (2 * size)) | (left << size) | right
                vector = matrices[size][:, basis]
                destination = int(np.argmax(np.abs(vector)))
                phase *= vector[destination]
                mask = (1 << size) - 1
                values[name] = (destination >> size) & mask
                values[f"junk_{name}"] = destination & mask
            else:
                values[name] = datasets[name][row]
        choose = keep[row] <= sigma
        values["choose_alt"] = int(choose)
        if choose:
            if rows > 1:
                values["l"], values["alt_l"] = alternate[row], row
            values["l_ne_zero"], values["alt_l_ne_zero"] = (
                values["alt_l_ne_zero"],
                values["l_ne_zero"],
            )
        assert values["l_ne_zero"] == int(values["l"] != 0)
        marginal[values["l"]] += 1 / (2 * rows)
        word = 0
        for name, size in prepare.layout:
            word = (word << size) | values[name]
        for spectator, coefficient in initial.items():
            expected[(word << (work + 1)) | spectator] = (
                phase * coefficient / sqrt(2 * rows)
            )
    assert [marginal[i] for i in range(rows)] == pytest.approx(
        [value / sum(weights) for value in weights], abs=1 / (2 * rows)
    )

    for ir, source, target in (
        (forward, initial, expected),
        (inverse, expected, initial),
    ):
        counts = count_emitted_instructions(ir)
        events = counts.unconditional.get("measure", 0)
        assert (counts.t_count, events, work) == {
            1: (11, 1, 2),
            2: (18, 1, 2),
            3: (49, 7, 4),
            4: (45, 1, 2),
        }[rows]
        for history in product((0, 1), repeat=events):
            assert _branch(
                ir.instructions,
                ir.n_wires,
                history,
                source,
                normalize_measurements=True,
            ) == pytest.approx(target, abs=2e-12)
