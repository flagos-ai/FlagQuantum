"""Check clean bank signatures, the padded erase leaf and append bindings."""

from importlib.metadata import version
from itertools import product
from math import prod
from typing import Any

import pytest
import torch

from flagquantum.algorithms.primitives.clean_table_lookup import CleanTableLookup
from flagquantum.compiler.registers.lookup.qroam_clean_metadata import (
    permuted_qroam_erase_data,
    qroam_clean_adjoint_leaf_signature,
    qroam_clean_signature,
)
from flagquantum.compiler.registers.lookup.source_qroam_clean import QroamCleanTable
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_source_qroam_clean import source_clean_graph

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def ports(signature: Any) -> list[tuple[Any, ...]]:
    return [
        (reg.name, reg.bitsize, reg.shape_symbolic, reg.side.name) for reg in signature
    ]


@pytest.mark.parametrize(
    "shape,logs,controls", [((5,), (1,), 0), ((5, 5), (1, 1), 1), ((1,), (0,), 0)]
)
def test_frozen_padded_erase_literals_and_signature(
    shape: tuple[int, ...], logs: tuple[int, ...], controls: int
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("qualtran")
    from qualtran import LeftDangle
    from qualtran.bloqs.data_loading import QROAMClean, QROAMCleanAdjoint

    assert version("qualtran") == "0.7.0"
    values = tuple(range(prod(shape)))
    datasets = (values, tuple(1 + v for v in values))
    widths = (max(1, (prod(shape) - 1).bit_length()), prod(shape).bit_length())
    table = QroamCleanTable(shape, logs, datasets)
    source = QROAMClean.build_from_data(
        *(np.asarray(data).reshape(shape) for data in datasets),
        target_bitsizes=widths,
        log_block_sizes=logs,
        num_controls=controls,
    )
    signature = qroam_clean_signature(shape, widths, logs, controls=controls)
    assert ports(signature) == ports(source.signature)
    wrapper = source.adjoint()
    assert ports(signature.adjoint()) == ports(wrapper.signature)
    leaf = wrapper.qroam_clean_adjoint_bloq
    assert isinstance(leaf, QROAMCleanAdjoint)
    blocks = table.block_shape
    padded = tuple(
        size * block for size, block in zip(table.batch_shape, blocks, strict=True)
    )
    assert leaf.data_shape == padded
    assert ports(
        qroam_clean_adjoint_leaf_signature(padded, widths, logs, controls=controls)
    ) == ports(leaf.signature)
    expected = tuple(
        tuple(tuple(row) for row in data.reshape(-1, prod(blocks)).tolist())
        for data in leaf.data
    )
    assert permuted_qroam_erase_data(table) == expected

    composite = wrapper.decompose_bloq()
    leaves = [
        inputs
        for node, inputs, _ in composite.iter_bloqsoqs()
        if isinstance(node.bloq, QROAMCleanAdjoint)
    ]
    assert len(leaves) == 1
    for i in range(len(widths)):
        bank = leaves[0][f"target{i}_"]
        assert bank.shape == blocks
        assert tuple((soq.binst, soq.reg.name, soq.idx) for soq in bank.flat) == (
            (LeftDangle, f"target{i}_", ()),
            *((LeftDangle, f"junk_target{i}_", (j,)) for j in range(prod(blocks) - 1)),
        )


def test_symbolic_ports_for_two_datasets() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.data_loading import QROAMClean
    from sympy import symbols

    n, m, a, b, k, log_m, c = symbols("n m a b k l c", positive=True, integer=True)
    source = QROAMClean.build_from_bitsize(
        (n, m), (a, b), log_block_sizes=(k, log_m), num_controls=c
    )
    native = qroam_clean_signature((n, m), (a, b), (k, log_m), controls=c)
    assert ports(native) == ports(source.signature)
    assert ports(native.adjoint()) == ports(source.adjoint().signature)
    assert ports(
        qroam_clean_adjoint_leaf_signature((n, m), (a, b), (k, log_m), controls=c)
    ) == ports(source.adjoint().qroam_clean_adjoint_bloq.signature)


def test_named_banks_append_and_every_measurement_branch() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("qualtran")
    from qualtran.bloqs.data_loading import QROAMClean

    table = QroamCleanTable((3,), (1,), ((1, 0, 1), (0, 1, 0)))
    lookup = CleanTableLookup(table, (1, 1))
    source = QROAMClean.build_from_data(
        np.array([1, 0, 1]), np.array([0, 1, 0]), log_block_sizes=(1,)
    )
    selection = (5, 2)
    named = {
        "target0_": (1,),
        "target1_": (4,),
        "junk_target0_": (3,),
        "junk_target1_": (0,),
    }
    q = lookup.workspace_size()
    work = tuple(range(6, 6 + q))
    bits = tuple(range(70, 70 + q))
    bank_bits = (81, 29, 63, 97)
    args: dict[str, Any] = dict(
        controls=(),
        selections=(selection,),
        outputs=named,
        workspace=work,
        classical_bits=bits,
    )
    forward = lookup.make_on(**args)
    erase = lookup.make_on(**args, bank_bits=bank_bits)
    circuit = DynamicCircuit(6 + q).append(forward)
    expected, _, _ = source_clean_graph(
        source, {"selection": selection}, {"selection": selection, **named}, work, bits
    )
    assert_source(circuit.to_ir().instructions, expected)
    gates = circuit.append(erase).to_ir().instructions
    assert [
        g.metadata["classical_bit"]
        for g in gates
        if g.name == "measure" and g.metadata["classical_bit"] in bank_bits
    ] == list(bank_bits)
    state = torch.zeros(1 << (6 + q), dtype=torch.complex128)
    for j, amplitude in enumerate((1 + 2j, 2 - 1j, -1 + 2j, 3j)):
        basis = sum(
            (j >> (1 - i) & 1) << (5 + q - wire) for i, wire in enumerate(selection)
        )
        state[basis] = torch.tensor(amplitude, dtype=torch.complex128)
    state /= state.norm()
    count = sum(g.name == "measure" for g in gates)
    for outcomes in product((0, 1), repeat=count):
        torch.testing.assert_close(
            forced_branch(state, gates, outcomes),
            state / (2 ** (count / 2)),
            rtol=0,
            atol=1e-12,
        )
