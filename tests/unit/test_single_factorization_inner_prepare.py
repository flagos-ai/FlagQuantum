"""Verify the B11 signal, both directions, and the retained signed QROAM banks."""

from itertools import product
from math import sqrt

import pytest

from flagquantum.algorithms.primitives.single_factorization_inner_prepare import (
    PrepareSingleFactorizationInner,
)
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler.registers.partition import partition_register
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "spatial,factor_count,blocked",
    [(1, 1, False), (1, 2, False), (2, 1, False), (2, 2, False), (3, 3, True)],
)
def test_inner_prepare_complex_alias_signal(spatial, factor_count, blocked):
    pytest.importorskip("qualtran")
    import cirq
    import numpy as np
    from qualtran.bloqs.swap_network.swap_with_zero import SwapWithZero
    from qualtran.linalg.lcu_util import (
        preprocess_probabilities_for_reversible_sampling,
    )

    pairs = tuple((p, q) for q in range(spatial) for p in range(q + 1))
    rows = tuple(
        tuple(
            (-1) ** (factor + i) * (1 + ((i + factor) % len(pairs)))
            for i in range(len(pairs))
        )
        for factor in range(factor_count)
    )
    prep = PrepareSingleFactorizationInner.from_coefficients(
        spatial,
        rows[0],
        rows[1:],
        probability_bits=1,
        log_blocks=(1, 1) if blocked else (0, 0),
    )
    logical = sum(size for _, size in prep.layout)
    work, _ = prep.requirements()
    wires = logical + work
    ports = partition_register(tuple(range(logical)), prep.layout)
    circuits = tuple(
        DynamicCircuit(wires)
        .append(
            prep.make_on(ports, workspace=tuple(range(logical, wires)), adjoint=adjoint)
        )
        .to_ir()
        for adjoint in (False, True)
    )
    layout = dict(prep.layout)
    targets = [
        name
        for name in ("alt_p", "alt_q", "keep", "theta", "alt_theta")
        if name in layout
    ]
    banks = {name: [] for name in targets}
    for row in rows:
        alternate, keep, _ = preprocess_probabilities_for_reversible_sampling(
            tuple(map(abs, row)), sub_bit_precision=1
        )
        for i, (alt, threshold) in enumerate(zip(alternate, keep, strict=True)):
            p, q = pairs[alt]
            values = dict(
                alt_p=p,
                alt_q=q,
                keep=threshold,
                theta=int(row[i] < 0),
                alt_theta=int(row[alt] < 0),
            )
            for name in targets:
                banks[name].append(values[name])
    assert prep.table.datasets == tuple(tuple(banks[name]) for name in targets)
    matrices = {}
    if blocked:
        for size in {layout[name] for name in targets}:
            qubits = cirq.LineQubit.range(2 + 4 * size)
            source = SwapWithZero((1, 1), size, (2, 2))
            op = source.on_registers(
                selection0_=qubits[:1],
                selection1_=qubits[1:2],
                targets=np.array(qubits[2:]).reshape(2, 2, size),
            )
            matrices[size] = cirq.Circuit(cirq.decompose(op)).unitary(
                qubit_order=qubits
            )

    def pack(values):
        word = 0
        for name, size in prep.layout:
            word = (word << size) | values.get(name, 0)
        return word << work

    initial = {
        pack({"l": factor}): complex(1, (-1) ** factor) / sqrt(2 * factor_count)
        for factor in range(factor_count)
    }
    expected = {}
    for factor, address, sigma in product(
        range(factor_count), range(len(pairs)), range(2)
    ):
        p, q = pairs[address]
        values = {
            "l": factor,
            "p": p,
            "q": q,
            "s": address,
            "sigma": sigma,
            "succ_pq": 1,
        }
        phase = 1 + 0j
        for name in targets:
            if not blocked:
                values[name] = banks[name][factor * len(pairs) + address]
                continue
            size = layout[name]
            mask = (1 << size) - 1
            index = ((factor & 1) << 1) | (address & 1)
            for i, j in product(range(2), repeat=2):
                outer, inner = (factor & ~1) + i, (address & ~1) + j
                cell = (
                    banks[name][outer * len(pairs) + inner]
                    if outer < factor_count and inner < len(pairs)
                    else 0
                )
                index = (index << size) | cell
            column = matrices[size][:, index]
            dest = int(np.argmax(np.abs(column)))
            phase *= column[dest]
            values[name] = (dest >> (3 * size)) & mask
            values[f"junk_{name}"] = dest & ((1 << (3 * size)) - 1)
        choose = values["keep"] <= sigma
        values["choose_alt"] = int(choose)
        if choose:
            for name in ("p", "q", "theta"):
                alt_name = f"alt_{name}"
                if alt_name in values:
                    values[name], values[alt_name] = values[alt_name], values[name]
        expected[pack(values)] = (
            phase * initial[pack({"l": factor})] / sqrt(2 * len(pairs))
        )
    assert sum(abs(value) ** 2 for value in expected.values()) == pytest.approx(1)

    for ir, before, after in (
        (circuits[0], initial, expected),
        (circuits[1], expected, initial),
    ):
        assert isinstance(ir, CircuitIR)
        counts = count_emitted_instructions(ir)
        events = counts.unconditional.get("measure", 0)
        assert (counts.t_count, events, work) == {
            (1, 1): (11, 1, 2),
            (1, 2): (11, 1, 2),
            (2, 1): (53, 8, 4),
            (2, 2): (65, 11, 4),
            (3, 3): (219, 25, 7),
        }[spatial, factor_count]
        histories = (
            product((0, 1), repeat=events)
            if events <= 11
            else (
                (pattern * events)[:events] for pattern in ((0,), (1,), (0, 1), (1, 0))
            )
        )
        for history in histories:
            assert _branch(
                ir.instructions, wires, history, before, normalize_measurements=True
            ) == pytest.approx(after, abs=2e-12)
