"""Check signed factor aliases, offset arithmetic and branch restoration."""

from itertools import product
from math import cos, sin, sqrt, tau

import pytest

from flagquantum.algorithms.primitives.double_factorization_inner_prepare import (
    PrepareDoubleFactorizationInner,
)
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler.registers.partition import partition_register
from flagquantum.dynamic import DynamicCircuit
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "rows,log",
    [
        (((-1.0,),), 0),
        (((1.0, -3.0), (-2.0,)), 0),
        (((1.0, -3.0, 2.0), (2.0, -1.0), (-3.0, 1.0, -2.0)), 1),
    ],
)
def test_inner_aliases_keep_factor_local_signs_and_invert(rows, log):
    pytest.importorskip("qualtran")
    import cirq
    import numpy as np
    from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox
    from qualtran.linalg.lcu_util import (
        preprocess_probabilities_for_reversible_sampling,
    )

    prepare = PrepareDoubleFactorizationInner.from_eigenvalues(
        rows[0], rows[1:], probability_bits=1, rotation_bits=2, log_block_size=log
    )
    data = prepare.indexed_data()
    logical = sum(size for _, size in prepare.layout)
    work, _ = prepare.requirements()
    # Keep the original factor as a spectator after the scratch register.
    spectators = (len(rows) - 1).bit_length()
    width = logical + work + spectators
    ports = partition_register(tuple(range(logical)), prepare.layout)
    workspace = tuple(range(logical, logical + work))
    forward, inverse = (
        DynamicCircuit(width)
        .append(prepare.make_on(ports, workspace=workspace, adjoint=adjoint))
        .to_ir()
        for adjoint in (False, True)
    )
    target_names = [
        name for name in ("alt_p", "keep", "theta", "alt_theta") if name in ports
    ]
    columns = {name: [] for name in target_names}
    for row in rows:
        alt, keep, _ = preprocess_probabilities_for_reversible_sampling(
            tuple(map(abs, row)), sub_bit_precision=1
        )
        values = dict(
            alt_p=alt,
            keep=keep,
            theta=[int(v < 0) for v in row],
            alt_theta=[int(row[i] < 0) for i in alt],
        )
        for name in target_names:
            columns[name].extend(values[name])
    assert prepare.table.datasets == tuple(tuple(column) for column in columns.values())
    assert data.ranks == tuple(map(len, rows))

    matrices = {}
    if log:
        for size in {len(ports[name]) for name in target_names}:
            qubits = cirq.LineQubit.range(1 + 2 * size)
            matrices[size] = cirq.Circuit(
                cirq.decompose(CSwapApprox(size).on(*qubits))
            ).unitary(qubit_order=qubits)

    def pack(values, factor):
        word = 0
        for name, size in prepare.layout:
            word = (word << size) | values.get(name, 0)
        return (word << (work + spectators)) | factor

    initial, expected = {}, {}
    offset = 0
    factor_switches = 0
    extent = sum(map(len, rows))
    address_mask = (1 << (extent - 1).bit_length()) - 1
    for factor, (row, rotation_word) in enumerate(
        zip(rows, data.rotation_words, strict=True)
    ):
        metadata = dict(xi=len(row) - 1, offset=offset, rot_data=rotation_word)
        coefficient = complex(1, (-1) ** factor) / sqrt(2 * len(rows))
        initial[pack(metadata, factor)] = coefficient
        capacity = 1 << (len(row) - 1).bit_length()
        theta = tau * rotation_word / 4
        rotation = (cos(theta / 2), sin(theta / 2))
        good = len(row) * rotation[0] ** 2 / capacity
        for orbital, ancilla, sigma in product(range(capacity), range(2), range(2)):
            success = int(orbital < len(row) and ancilla == 0)
            value = (
                coefficient
                * rotation[ancilla]
                * (1 - 4 * good + 2 * success)
                / sqrt(2 * capacity)
            )
            if abs(value) <= 1e-12:
                continue
            address = (offset + orbital) & address_mask
            values = metadata | dict(
                p=orbital,
                rotation=ancilla,
                succ_p=success,
                address=address,
                sigma=sigma,
            )
            phase = 1 + 0j
            for name, column in columns.items():
                if not log:
                    values[name] = column[address]
                    continue
                size = len(ports[name])
                left, right = column[address & ~1 : (address & ~1) + 2]
                basis = ((address & 1) << (2 * size)) | (left << size) | right
                amplitudes = matrices[size][:, basis]
                result = int(np.argmax(np.abs(amplitudes)))
                phase *= amplitudes[result]
                values[name] = (result >> size) & ((1 << size) - 1)
                values[f"junk_{name}"] = result & ((1 << size) - 1)
            choose = values["keep"] <= sigma
            values["choose_alt"] = int(choose)
            if choose and success:
                if "alt_p" in values:
                    values["p"], values["alt_p"] = values["alt_p"], values["p"]
                factor_switches += int(values["theta"] != values["alt_theta"])
                values["theta"], values["alt_theta"] = (
                    values["alt_theta"],
                    values["theta"],
                )
            if success:
                assert values["p"] < len(row)
                assert values["theta"] == int(row[values["p"]] < 0)
            else:
                assert values["p"] == orbital
            expected[pack(values, factor)] = phase * value
        offset += len(row)
    if len(rows) > 1:
        assert factor_switches
    assert sum(abs(value) ** 2 for value in expected.values()) == pytest.approx(1)

    for ir, before, after in (
        (forward, initial, expected),
        (inverse, expected, initial),
    ):
        counts = count_emitted_instructions(ir)
        events = counts.unconditional.get("measure", 0)
        assert (counts.t_count, events, work) == {
            1: (19, 3, 2),
            2: (54, 10, 3),
            3: (129, 22, 7),
        }[len(rows)]
        histories = (
            product((0, 1), repeat=events)
            if events <= 10
            else (
                (pattern * events)[:events] for pattern in ((0,), (1,), (0, 1), (1, 0))
            )
        )
        for history in histories:
            assert _branch(
                ir.instructions, width, history, before, normalize_measurements=True
            ) == pytest.approx(after, abs=2e-12)


def test_rank_metadata_lookup_inner_and_adjoint_roundtrip():
    prepare = PrepareDoubleFactorizationInner.from_eigenvalues(
        (1.0, -3.0),
        ((-2.0,),),
        probability_bits=1,
        rotation_bits=2,
    )
    indexed = prepare.indexed_data()
    layout = (
        *indexed.layout,
        *(port for port in prepare.layout if port[0] not in dict(indexed.layout)),
    )
    logical = sum(size for _, size in layout)
    ports = partition_register(tuple(range(logical)), layout)
    outer_ports = {name: ports[name] for name, _ in indexed.layout}
    inner_ports = {name: ports[name] for name, _ in prepare.layout}
    width = logical + max(indexed.workspace_size(), prepare.requirements()[0])
    outer_work = tuple(range(logical, logical + indexed.workspace_size()))
    inner_work = tuple(range(logical, logical + prepare.requirements()[0]))
    circuit = DynamicCircuit(width)
    circuit.append(indexed.make_on(outer_ports, workspace=outer_work))
    circuit.append(prepare.make_on(inner_ports, workspace=inner_work))
    circuit.append(prepare.make_on(inner_ports, workspace=inner_work, adjoint=True))
    circuit.append(indexed.make_on(outer_ports, workspace=outer_work, adjoint=True))
    stream = circuit.to_ir().instructions
    initial = {0: (1 + 1j) / 2, 1 << (width - 1): (1 - 1j) / 2}
    events = sum(gate.name == "measure" for gate in stream)
    for pattern in ((0,), (1,), (0, 1), (1, 0)):
        history = (pattern * events)[:events]
        assert _branch(
            stream, width, history, initial, normalize_measurements=True
        ) == pytest.approx(initial, abs=2e-12)
