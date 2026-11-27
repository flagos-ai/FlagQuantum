"""Packed polynomial views preserve all raw encodings and complex spectators."""

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.finite_field.field_polynomial_component import (
    field_polynomial_component_program,
)
from flagquantum.compiler.registers.finite_field.field_polynomial_layout import (
    FieldPolynomialLayout,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from flagquantum.simulation.finite_field_polynomial_encoding import (
    FiniteFieldPolynomialEncoding,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("p,m,degree", ((2, 3, 1), (3, 1, 1), (5, 1, 1), (3, 2, 0)))
@pytest.mark.parametrize("reverse", (False, True))
def test_complete_raw_source_operator_aliases_and_typed_append(
    p: int, m: int, degree: int, reverse: bool
) -> None:
    try:
        import cirq
        from qualtran._infra.data_types import QGF, QGFPoly
        from qualtran.bloqs.gf_poly_arithmetic.gf_poly_split_and_join import (
            GFPolyJoin,
            GFPolySplit,
        )
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)

    encoding = FiniteFieldPolynomialEncoding(degree, FiniteFieldEncoding(p, m))
    width = int(encoding.field.bitsize)
    layout = FieldPolynomialLayout(degree, width)
    program = field_polynomial_component_program(
        layout,
        encoding.field.register_type(),
        reverse=reverse,
        packed_dtype=encoding.register_type(),
    )
    source = (GFPolyJoin if reverse else GFPolySplit)(QGFPoly(degree, QGF(p, m)))
    registers = program.functions[0].registers
    assert registers is not None
    assert [(r.name, r.bitsize, r.shape, r.side.value) for r in registers] == [
        (r.name, r.bitsize, r.shape, r.side.value) for r in source.signature
    ]
    wires = tuple(reversed(range(layout.logical_width)))
    words = layout.split(wires)
    bound = bind_register_program(program, words if reverse else (wires,))
    sdk_wires = np.array([cirq.LineQubit(i) for i in wires], dtype=object)
    gate, outputs = source.as_cirq_op(
        None, reg=sdk_wires.reshape((degree + 1, width)) if reverse else sdk_wires
    )
    assert gate is None
    assert bound.outputs == ((wires,) if reverse else words)
    assert tuple(q.x for q in outputs["reg"].flat) == wires
    assert bound.circuit.instructions == ()
    assert bound.events == () and bound.classical_outputs == ()
    operator = torch.as_tensor(
        source.tensor_contract(), dtype=torch.complex128
    ).reshape(1 << layout.logical_width, 1 << layout.logical_width)
    torch.testing.assert_close(
        operator, torch.eye(operator.shape[0], dtype=torch.complex128), atol=0, rtol=0
    )
    count = layout.logical_width + 2
    weights = torch.randn(
        operator.shape[0],
        2,
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(533),
    )
    weights /= weights.norm()
    state = torch.zeros(1 << count, dtype=torch.complex128)
    state.reshape(operator.shape[0], 4)[:, (0, 2)] = weights
    expected = torch.zeros_like(state)
    expected.reshape(operator.shape[0], 4)[:, (0, 2)] = operator @ weights
    actual = (
        Circuit(count, inputs=state, dtype=torch.complex128)
        .append(FieldOperation(count, bound.circuit.instructions))
        .state()
        .flatten()
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
