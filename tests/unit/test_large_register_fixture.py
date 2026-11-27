"""Compare the measured large fixture against the locked graph and tensor."""

from itertools import product
from typing import Any

import pytest
import torch

from flagquantum.core.ir import Instruction
from tests._reference_dependencies import unavailable_reference
from tests.unit.large_register_fixture import large_fixture_ir, large_fixture_signature
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.register_source_graph import SourcePorts, SourceResult
from tests.unit.register_source_workspace import source_workspace_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def controlled_source_atom(
    bloq: Any, inputs: SourcePorts, desired: SourcePorts
) -> SourceResult | None:
    from qualtran._infra.controlled import Controlled
    from qualtran.bloqs.for_testing.atom import TestAtom

    if not isinstance(bloq, Controlled) or not isinstance(bloq.subbloq, TestAtom):
        return None
    try:
        from cirq import CNOT, unitary
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)

    tensor = torch.as_tensor(bloq.tensor_contract().copy())
    torch.testing.assert_close(
        tensor, torch.as_tensor(unitary(CNOT), dtype=tensor.dtype), atol=0, rtol=0
    )
    assert not desired or all(inputs[name] == wires for name, wires in desired.items())
    return (Instruction("cx", (*inputs["ctrl"], *inputs["q"])),), inputs


@pytest.mark.parametrize("n_select,repetitions", [(2, 1), (3, 2), (4, 5), (3, 0)])
def test_large_fixture_gate_polarities_and_coherent_channel(
    n_select: int, repetitions: int
) -> None:
    pytest.importorskip("qualtran")
    from qualtran._infra.bloq import DecomposeTypeError
    from qualtran.bloqs.for_testing.large_bloq import LargeBloq
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    source = LargeBloq(n_select, repetitions)
    assert [(r.name, r.bitsize) for r in large_fixture_signature(n_select)] == [
        (r.name, r.bitsize) for r in source.signature
    ]
    select, target = tuple(range(n_select)), n_select
    work = tuple(range(n_select + 1, 2 * n_select)) if repetitions else ()
    n_measure = (n_select - 1) * repetitions
    gates = large_fixture_ir(select, target, repetitions, work, tuple(range(n_measure)))
    inputs = {"select": select, "target": (target,)}
    expected, lifetimes = source_workspace_gates(
        source, inputs, work, (), custom_leaf=controlled_source_atom
    )
    assert_source(gates, expected)
    if n_measure:
        assert len(lifetimes) == 2 * n_measure
        with pytest.raises(DecomposeTypeError, match="TestAtom is atomic"):
            get_cost_value(source, QECGatesCost())
    else:
        assert not lifetimes
    assert get_cost_value(source, QubitCount()) == n_select + 1 + len(work)
    assert sum(g.name in {"t", "tdg"} for g in gates) == 4 * n_measure
    assert sum(g.name == "measure" for g in gates) == n_measure

    indices = torch.arange(1, (2 << n_select) + 1, dtype=torch.float64)
    initial = torch.complex(indices, 7 - indices)
    initial /= initial.norm()
    matrix = torch.as_tensor(source.tensor_contract().copy(), dtype=initial.dtype)
    vacuum = torch.zeros(1 << len(work), dtype=initial.dtype)
    vacuum[0] = 1
    coherent_in = torch.kron(initial, vacuum)
    coherent_out = torch.kron(matrix @ initial, vacuum)
    basis = tuple(product((0, 1), repeat=n_select + 1))
    mapped = matrix.abs().argmax(dim=0).tolist()
    assert torch.count_nonzero(matrix) == len(basis)
    assert torch.all(matrix[mapped, range(len(basis))] == 1)
    scratch = (0,) * len(work)
    assert certified_and_channel_map(
        gates, tuple(word + scratch for word in basis)
    ) == (tuple(basis[index] + scratch for index in mapped), n_measure)
    histories = (
        product((0, 1), repeat=n_measure)
        if n_measure <= 4
        else ((0,) * n_measure, (1,) * n_measure)
    )
    for history in histories:
        actual = forced_branch(coherent_in, gates, history)
        torch.testing.assert_close(
            actual, coherent_out * 2 ** (-n_measure / 2), atol=5e-13, rtol=0
        )
