"""Typed square result ownership and coherent erasure use shared columns."""

from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.arithmetic.integer_square import integer_square_ir
from flagquantum.compiler.registers.arithmetic.integer_square_component import (
    integer_square_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_weighted_columns_component import normalized_feedback

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", (1, 2, 3, 4, 5, 8))
def test_square_source_ports_and_native_streams(width: int) -> None:
    from qualtran.bloqs.arithmetic.multiplication import Square
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    source = Square(width)
    graph = integer_square_program(width)
    signature = graph.functions[0].registers
    assert signature is not None
    assert [(r.name, r.dtype.quantum_bits, r.side.name) for r in signature] == [
        (r.name, r.dtype.num_qubits, r.side.name) for r in source.signature
    ]
    word = tuple(range(width))
    result = tuple(range(width, 3 * width))
    work = tuple(range(3 * width, 3 * width + width * (width - 1)))
    forward = bind_register_program(
        graph, (word,), clean=(*result, *work), allow_workspace_outputs=True
    )
    inverse = bind_register_program(
        integer_square_program(width, adjoint=True),
        (word, result),
        clean=work,
        input_ownership=tuple((wire, False) for wire in result),
    )
    reference = integer_square_ir(word, result, work)
    assert forward.outputs == (word, result)
    assert inverse.outputs == (word,)
    for bound in (forward, inverse):
        assert_source(normalized_feedback(bound.circuit.instructions), reference)
    assert get_cost_value(source, QECGatesCost()).toffoli == width * (width - 1)


@pytest.mark.parametrize("width", (1, 2, 3))
def test_square_coherent_owned_image_all_small_histories(width: int) -> None:
    word = tuple(range(width))
    output = tuple(range(width, 3 * width))
    work = tuple(range(3 * width + 1, 3 * width + 1 + width * (width - 1)))
    size = 3 * width + 1 + len(work)
    before = torch.zeros(1 << size, dtype=torch.complex128)
    after = torch.zeros_like(before)
    coefficients = torch.arange(1, (1 << (width + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(coefficients, 3 - coefficients / 2)
    amplitudes /= amplitudes.norm()
    for raw, spectator in product(range(1 << width), (0, 1)):
        prefix = raw << (2 * width + 1 + len(work))
        before[prefix | spectator << len(work)] = amplitudes[2 * raw + spectator]
        after[prefix | ((raw * raw) << (1 + len(work))) | spectator << len(work)] = (
            amplitudes[2 * raw + spectator]
        )
    forward = bind_register_program(
        integer_square_program(width),
        (word,),
        clean=(*output, *work),
        allow_workspace_outputs=True,
    )
    inverse = bind_register_program(
        integer_square_program(width, adjoint=True),
        (word, output),
        clean=work,
        input_ownership=tuple((wire, False) for wire in output),
    )
    histories = (
        tuple(product((0, 1), repeat=len(work)))
        if width <= 2
        else ((0,) * len(work), (1,) * len(work))
    )
    for first, second in product(histories, repeat=2):
        state = forced_branch(before, forward.circuit.instructions, first)
        torch.testing.assert_close(
            state, after / 2 ** (len(work) / 2), atol=3e-12, rtol=0
        )
        restored = forced_branch(state, inverse.circuit.instructions, second)
        torch.testing.assert_close(
            restored, before / 2 ** len(work), atol=3e-12, rtol=0
        )
