"""Typed Product and SumOfSquares preserve shared columns and owned images."""

from collections import Counter
from itertools import product
from typing import Any, Literal

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.arithmetic.integer_product import (
    integer_product_ir,
    integer_product_requirements,
)
from flagquantum.compiler.registers.arithmetic.integer_product_component import (
    integer_product_program,
)
from flagquantum.compiler.registers.arithmetic.integer_square import (
    sum_of_squares_ir,
    sum_of_squares_requirements,
)
from flagquantum.compiler.registers.arithmetic.sum_of_squares_component import (
    sum_of_squares_program,
)
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_weighted_columns_component import normalized_feedback

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
Kind = Literal["product", "squares"]
CASES: tuple[tuple[Kind, int, int], ...] = (
    ("product", 1, 1),
    ("product", 1, 4),
    ("product", 4, 1),
    ("product", 2, 3),
    ("product", 4, 6),
    ("squares", 1, 1),
    ("squares", 1, 3),
    ("squares", 2, 1),
    ("squares", 2, 3),
    ("squares", 3, 4),
)


def graph(kind: Kind, n: int, m: int, adjoint: bool = False) -> ProgramRecord:
    factory = integer_product_program if kind == "product" else sum_of_squares_program
    return factory(n, m, adjoint=adjoint)


def layout(kind: Kind, n: int, m: int) -> tuple[tuple[int, ...], int, int]:
    if kind == "product":
        result, work = integer_product_requirements(n, m)
        return (n, m), result, work
    result, work = sum_of_squares_requirements(n, m)
    return (n,) * m, result, work


@pytest.mark.parametrize("kind,n,m", CASES)
@pytest.mark.parametrize("expanded", (False, True))
def test_owned_ports_calls_and_primitive_dependencies(
    kind: Kind, n: int, m: int, expanded: bool
) -> None:
    from qualtran.bloqs.arithmetic.multiplication import Product, SumOfSquares
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    source = Product(n, m) if kind == "product" else SumOfSquares(n, m)
    module = graph(kind, n, m)
    assert verify_program_record(module) == ()
    registers = module.functions[0].registers
    assert registers is not None
    assert [
        (r.name, r.dtype.quantum_bits, r.shape, r.side.name) for r in registers
    ] == [(r.name, r.bitsize, r.shape, r.side.name) for r in source.signature]
    assert (
        sum(
            op.opcode == "call"
            and op.callee is not None
            and op.callee.startswith("__fq_component_")
            for op in module.functions[0].blocks[0].operations
        )
        == 1
    )
    widths, output_width, work_width = layout(kind, n, m)
    offset = 0
    words = []
    for width in widths:
        words.append(tuple(range(offset, offset + width)))
        offset += width
    output = tuple(range(offset, offset + output_width))
    work = tuple(range(offset + output_width, offset + output_width + work_width))
    ports = tuple(words)
    reference = (
        integer_product_ir(words[0], words[1], output, work)
        if kind == "product"
        else sum_of_squares_ir(tuple(words), output, work)
    )
    for adjoint in (False, True):
        module = graph(kind, n, m, adjoint)
        if expanded:
            module, _ = expand_component_calls(module)
        bound = bind_register_program(
            module,
            (*ports, output) if adjoint else ports,
            clean=work if adjoint else (*output, *work),
            allow_workspace_outputs=not adjoint,
            input_ownership=tuple((wire, False) for wire in output) if adjoint else (),
        )
        assert bound.outputs == (ports if adjoint else (*ports, output))
        assert_source(normalized_feedback(bound.circuit.instructions), reference)
        counts = Counter(g.name for g in bound.circuit.instructions)
        assert counts["t"] + counts["tdg"] == 4 * work_width
        assert counts["measure"] == counts["reset"] == work_width
    model = get_cost_value(source, QECGatesCost()).toffoli
    assert model == (
        2 * n * m - max(n, m) if kind == "product" else m * n * n - n - int(m % 3 == 0)
    )


@pytest.mark.parametrize(
    "kind,n,m",
    (("product", 1, 1), ("product", 2, 2), ("squares", 1, 3), ("squares", 2, 2)),
)
def test_complete_complex_images_and_inverse_work_zero(
    kind: Kind, n: int, m: int
) -> None:
    widths, output_width, work_width = layout(kind, n, m)
    offset = 0
    words = []
    for width in widths:
        words.append(tuple(range(offset, offset + width)))
        offset += width
    output = tuple(range(offset, offset + output_width))
    work = tuple(range(offset + output_width, offset + output_width + work_width))
    ports = tuple(words)
    forward = bind_register_program(
        graph(kind, n, m), ports, clean=(*output, *work), allow_workspace_outputs=True
    )
    inverse = bind_register_program(
        graph(kind, n, m, True),
        (*ports, output),
        clean=work,
        input_ownership=tuple((wire, False) for wire in output),
    )
    total_width = offset + output_width + work_width + 1
    before: dict[int, complex] = {}
    after: dict[int, complex] = {}
    for values in product(*(range(1 << width) for width in widths)):
        raw = 0
        for value, width in zip(values, widths, strict=True):
            raw = (raw << width) | value
        image = (
            values[0] * values[1]
            if kind == "product"
            else sum(value * value for value in values)
        )
        for spectator in (0, 1):
            prefix = raw << (output_width + work_width + 1)
            amplitude = complex(raw + 1, 2 - spectator - raw / 3)
            before[prefix | spectator] = amplitude
            after[prefix | image << (work_width + 1) | spectator] = amplitude
    norm = sum(abs(value) ** 2 for value in before.values()) ** 0.5
    before = {i: value / norm for i, value in before.items()}
    after = {i: value / norm for i, value in after.items()}
    for bound, initial, expected in (
        (forward, before, after),
        (inverse, after, before),
    ):
        for seed in (2, 17):
            actual = execute_sparse_register(
                CircuitIR(total_width, bound.circuit.instructions), initial, seed=seed
            )
            assert all(
                abs(actual.get(i, 0) - expected.get(i, 0)) < 4e-12
                for i in actual.keys() | expected.keys()
            )
    if work_width <= 1:
        initial_vector = torch.zeros(1 << total_width, dtype=torch.complex128)
        expected_vector = torch.zeros_like(initial_vector)
        for i, coefficient in before.items():
            initial_vector[i] = torch.tensor(coefficient, dtype=torch.complex128)
        for i, coefficient in after.items():
            expected_vector[i] = torch.tensor(coefficient, dtype=torch.complex128)
        for first, second in product(product((0, 1), repeat=work_width), repeat=2):
            state = forced_branch(initial_vector, forward.circuit.instructions, first)
            torch.testing.assert_close(
                state, expected_vector / 2 ** (work_width / 2), atol=3e-12, rtol=0
            )
            restored = forced_branch(state, inverse.circuit.instructions, second)
            torch.testing.assert_close(
                restored, initial_vector / 2**work_width, atol=3e-12, rtol=0
            )


@pytest.mark.parametrize("kind", ("product", "squares"))
def test_append_replaces_flat_root(kind: Kind, monkeypatch: pytest.MonkeyPatch) -> None:
    from flagquantum.compiler.registers.arithmetic import (
        integer_product,
        integer_square,
    )

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Stored results must use shared column calls")

    monkeypatch.setattr(integer_product, "integer_product_ir", refuse)
    monkeypatch.setattr(integer_square, "sum_of_squares_ir", refuse)
    m = 1 if kind == "product" else 3
    widths, result_width, work_width = layout(kind, 1, m)
    start = sum(widths)
    words = tuple((wire,) for wire in range(start))
    output = tuple(range(start, start + result_width))
    work = tuple(range(start + result_width, start + result_width + work_width))
    ports = words
    bound = bind_register_program(
        graph(kind, 1, m), ports, clean=(*output, *work), allow_workspace_outputs=True
    )
    total = start + result_width + work_width + 1
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    for raw in range(1 << start):
        initial[raw << (total - start)] = torch.tensor(
            complex(raw + 1, 2 - raw), dtype=torch.complex128
        )
    initial /= initial.norm()
    circuit = DynamicCircuit(total, inputs=initial, dtype=torch.complex128)
    operation = FieldOperation(total, bound.circuit.instructions)
    circuit.append(operation)
    circuit.append(operation)
    result = run_dynamic(circuit, shots=4, seed=23, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(4, -1), initial.expand(4, -1), atol=3e-12, rtol=0
    )


@pytest.mark.parametrize("factory", (integer_product_program, sum_of_squares_program))
@pytest.mark.parametrize("n,m", ((True, 1), (1, False), (0, 1), (1, 0), (-1, 1)))
def test_invalid_shapes_fail_before_building(factory: Any, n: int, m: int) -> None:
    with pytest.raises(ValueError):
        factory(n, m)
    with pytest.raises(ValueError):
        factory(1, 1, adjoint=1)
