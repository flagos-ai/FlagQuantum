"""Stored fractional squares retain the original b port and owned result."""

from collections import Counter
from collections.abc import Callable
from itertools import product
from typing import cast

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.fixed_point.fixed_point_square import (
    fixed_point_square_ir,
    fixed_point_square_workspace_size,
)
from flagquantum.compiler.registers.fixed_point.fixed_point_square_component import (
    fixed_point_square_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import verify_program_record
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_fixed_point_square import truncated_square
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_weighted_columns_component import normalized_feedback

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
square_oracle = cast(Callable[[int, int], int], truncated_square)


@pytest.mark.parametrize("width", (3, 4, 5, 6, 8, 10))
@pytest.mark.parametrize("expanded", (False, True))
def test_source_ports_and_paper_primitive_dependencies(
    width: int, expanded: bool
) -> None:
    from qualtran.bloqs.arithmetic.multiplication import SquareRealNumber
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    original = SquareRealNumber(width)
    a, b, output = (tuple(range(i * width, (i + 1) * width)) for i in range(3))
    count = fixed_point_square_workspace_size(width)
    work = tuple(range(3 * width, 3 * width + count))
    for adjoint in (False, True):
        record = fixed_point_square_program(width, adjoint=adjoint)
        assert verify_program_record(record) == ()
        ports = record.functions[0].registers
        assert ports is not None
        assert [
            (r.name, r.dtype.quantum_bits, dict(r.dtype.parameters)) for r in ports
        ] == [
            (r.name, r.bitsize, {"fractional_width": r.dtype.num_frac, "signed": False})
            for r in original.signature
        ]
        assert [r.side.name for r in ports] == [
            "THRU",
            "THRU",
            "LEFT" if adjoint else "RIGHT",
        ]
        if expanded:
            record, _ = expand_component_calls(record)
        bound = bind_register_program(
            record,
            (a, b, output) if adjoint else (a, b),
            clean=work if adjoint else (*output, *work),
            allow_workspace_outputs=not adjoint,
            input_ownership=tuple((wire, False) for wire in output) if adjoint else (),
        )
        assert bound.outputs == ((a, b) if adjoint else (a, b, output))
        assert_source(
            normalized_feedback(bound.circuit.instructions),
            fixed_point_square_ir(a, output, work),
        )
        assert all(
            not set(gate.wires).intersection(b) for gate in bound.circuit.instructions
        )
        counts = Counter(gate.name for gate in bound.circuit.instructions)
        assert counts["t"] + counts["tdg"] == 4 * count
        assert counts["measure"] == counts["reset"] == count
    assert get_cost_value(original, QECGatesCost()).toffoli == width * width // 2 - 4


@pytest.mark.parametrize("width", (3, 4))
def test_complex_dual_inputs_spectator_inverse_and_clean_work(width: int) -> None:
    a, b, output = (tuple(range(i * width, (i + 1) * width)) for i in range(3))
    count = fixed_point_square_workspace_size(width)
    work = tuple(range(3 * width, 3 * width + count))
    size = 3 * width + count + 1
    forward = bind_register_program(
        fixed_point_square_program(width),
        (a, b),
        clean=(*output, *work),
        allow_workspace_outputs=True,
    )
    inverse = bind_register_program(
        fixed_point_square_program(width, adjoint=True),
        (a, b, output),
        clean=work,
        input_ownership=tuple((wire, False) for wire in output),
    )
    before: dict[int, complex] = {}
    after: dict[int, complex] = {}
    for left, right, spectator in product(range(1 << width), range(1 << width), (0, 1)):
        amplitude = complex(left + right + 1, left - 2 * right + spectator)
        prefix = ((left << width) | right) << (width + count + 1)
        before[prefix | spectator] = amplitude
        after[prefix | (square_oracle(left, width) << (count + 1)) | spectator] = (
            amplitude
        )
    norm = sum(abs(value) ** 2 for value in before.values()) ** 0.5
    before = {i: value / norm for i, value in before.items()}
    after = {i: value / norm for i, value in after.items()}
    for bound, initial, expected in (
        (forward, before, after),
        (inverse, after, before),
    ):
        for seed in (3, 29):
            actual = execute_sparse_register(
                CircuitIR(size, bound.circuit.instructions), initial, seed=seed
            )
            assert all(
                abs(actual.get(i, 0) - expected.get(i, 0)) < 5e-12
                for i in actual.keys() | expected.keys()
            )
    if width == 3:
        initial_vector = torch.zeros(1 << size, dtype=torch.complex128)
        expected_vector = torch.zeros_like(initial_vector)
        for i, value in before.items():
            initial_vector[i] = torch.tensor(value, dtype=torch.complex128)
        for i, value in after.items():
            expected_vector[i] = torch.tensor(value, dtype=torch.complex128)
        histories = tuple(product((0, 1), repeat=count))
        for first, second in product(histories, repeat=2):
            state = forced_branch(initial_vector, forward.circuit.instructions, first)
            torch.testing.assert_close(
                state, expected_vector / 2 ** (count / 2), atol=5e-12, rtol=0
            )
            restored = forced_branch(state, inverse.circuit.instructions, second)
            torch.testing.assert_close(
                restored, initial_vector / 2**count, atol=5e-12, rtol=0
            )


def test_normal_append_uses_stored_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    from flagquantum.compiler.registers.fixed_point import fixed_point_square

    def refuse(*args: object, **kwargs: object) -> object:
        raise AssertionError("Flat square root must be replaceable")

    monkeypatch.setattr(fixed_point_square, "fixed_point_square_ir", refuse)
    forward = bind_register_program(
        fixed_point_square_program(3),
        ((0, 1, 2), (3, 4, 5)),
        clean=(6, 7, 8, 9, 10),
        allow_workspace_outputs=True,
    )
    inverse = bind_register_program(
        fixed_point_square_program(3, adjoint=True),
        ((0, 1, 2), (3, 4, 5), (6, 7, 8)),
        clean=(9, 10),
        input_ownership=((6, False), (7, False), (8, False)),
    )
    state = torch.zeros(1 << 12, dtype=torch.complex128)
    state[(7 << 9) | (5 << 6) | 1] = torch.tensor(
        (1 + 2j) / 5**0.5, dtype=torch.complex128
    )
    circuit = DynamicCircuit(12, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(12, forward.circuit.instructions))
    circuit.append(FieldOperation(12, inverse.circuit.instructions))
    actual = run_dynamic(circuit, shots=4, seed=13, strategy="trajectory")
    torch.testing.assert_close(
        actual.final_states.reshape(4, -1), state.expand(4, -1), atol=5e-12, rtol=0
    )


@pytest.mark.parametrize("width", (True, 0, 1, 2, -1))
def test_invalid_width(width: int) -> None:
    with pytest.raises(ValueError):
        fixed_point_square_program(width)


def test_invalid_direction() -> None:
    with pytest.raises(ValueError, match="boolean"):
        fixed_point_square_program(3, adjoint=1)  # type: ignore[arg-type]
