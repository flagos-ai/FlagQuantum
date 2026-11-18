"""Bind complete SwapWithZero source objects with padded and multiple axes."""

from collections import Counter
from dataclasses import replace
from math import prod
from typing import Any

import pytest
import torch

from examples.algorithms.swap_network_catalog import SwapWithZeroExample
from flagquantum import Circuit
from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_relative_controlled_swap import assert_source_dependencies
from tests.unit.test_swap_network_file import source_network as source_network

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "widths,word,shape",
    [
        ((2,), 1, (3,)),
        ((1, 1), 1, (2, 2)),
        ((1, 0), 1, (2, 1)),
        ((3,), 2, (2,)),
    ],
)
def test_symbolic_swap_zero_resolution_complete_operators(
    source_network: Any, widths: tuple[int, ...], word: int, shape: tuple[int, ...]
) -> None:
    from attrs import evolve
    from qualtran.bloqs.swap_network.swap_with_zero import SwapWithZero
    from sympy import symbols

    # These are the fields of the original symbolic count fixture. Resolution
    # is explicit; the three original example factories themselves are concrete.
    symbolic = SwapWithZero(
        symbols(f"s:{len(widths)}"), symbols("b"), symbols(f"N:{len(shape)}")
    )
    with pytest.raises(ValueError, match="symbolic"):
        _ = symbolic._swap_sequence
    source = evolve(
        symbolic,
        selection_bitsizes=widths,
        target_bitsize=word,
        n_target_registers=shape,
    )
    captured = import_register_emitter(source)
    emitter = replace(
        captured,
        snapshots=(
            CircuitIR.from_json(captured.snapshots[0].to_json()),
            CircuitIR.from_json(captured.snapshots[1].to_json()),
        ),
    )
    ports = tuple((r.name, int(r.total_bits())) for r in source.signature)
    logical = sum(bits for _, bits in ports)
    assert logical == sum(widths) + prod(shape) * word
    assert emitter.requirements() == (0, 0)
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    model = get_cost_value(source, QECGatesCost())
    assert model.t == 4 * word * (prod(shape) - 1)
    assert model.clifford == (22 * word - 1) * (prod(shape) - 1)
    assert model.measurement == model.and_bloq == model.toffoli == 0
    assert get_cost_value(source, QubitCount()) == logical
    binding = BlackBoxPrepare(ports, (), emitter)
    operators = []
    for adjoint in (False, True):
        native = SwapWithZeroExample(widths, word, shape, adjoint)
        actual = Circuit(logical).append(native).to_ir()
        restored = (
            Circuit(logical)
            .append(
                binding.make_on(
                    selection=tuple(range(logical)),
                    junk=(),
                    workspace=(),
                    classical_bits=(),
                    adjoint=adjoint,
                )
            )
            .to_ir()
        )
        assert isinstance(actual, CircuitIR) and isinstance(restored, CircuitIR)
        assert_source_dependencies(actual.instructions, restored.instructions)
        counts = Counter(g.name for g in actual.instructions)
        assert counts["t"] + counts["tdg"] == 4 * word * (prod(shape) - 1)
        assert counts["cx"] == (6 * word - 1) * (prod(shape) - 1)
        assert counts["measure"] == counts["reset"] == 0
        matrices = []
        for ir in (actual, restored):
            columns = []
            for column in range(1 << logical):
                initial = torch.zeros(1 << logical, dtype=torch.complex128)
                initial[column] = 1
                columns.append(forced_branch(initial, ir.instructions, ()))
            matrices.append(torch.stack(columns, dim=1))
        torch.testing.assert_close(matrices[0], matrices[1], atol=8e-12, rtol=0)
        operators.append(matrices[0])
    forward, backward = operators
    torch.testing.assert_close(
        backward @ forward,
        torch.eye(1 << logical, dtype=torch.complex128),
        atol=1e-11,
        rtol=0,
    )


def test_leading_zero_axis_is_an_unchanged_source_failure(source_network: Any) -> None:
    from qualtran.bloqs.swap_network.swap_with_zero import SwapWithZero

    source = SwapWithZero((0, 1), 1, (1, 2))
    with pytest.raises(IndexError, match="list index out of range"):
        source.decompose_bloq()
    with pytest.raises(IndexError, match="list index out of range"):
        import_register_emitter(source)
    native = SwapWithZeroExample((0, 1), 1, (1, 2))
    with pytest.raises(ValueError, match="zero"):
        Circuit(native.qubit_count).append(native)
