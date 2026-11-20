"""Typed packed views preserve source coefficient order, aliases and identity."""

from dataclasses import replace
from math import prod
from typing import cast

import pytest
import torch

try:
    import numpy as np
except ModuleNotFoundError:
    pytest.skip(
        "NumPy is needed for polynomial component views", allow_module_level=True
    )

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.finite_field.field_polynomial_component import (
    field_polynomial_component_program,
)
from flagquantum.compiler.registers.finite_field.field_polynomial_layout import (
    FieldPolynomialLayout,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from flagquantum.simulation.finite_field_polynomial_encoding import (
    FiniteFieldPolynomialEncoding,
)
from tests.unit.test_packed_field_polynomial_program import polynomial_type

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("degree,width", [(0, 1), (1, 2), (4, 3)])
@pytest.mark.parametrize("shape", [(), (2, 2), (0,)])
def test_nested_split_join_row_major_aliases_and_source_signature(
    degree: int, width: int, shape: tuple[int, ...]
) -> None:
    from qualtran import QGF, QGFPoly
    from qualtran.bloqs.gf_poly_arithmetic.gf_poly_split_and_join import GFPolySplit

    field = FiniteFieldEncoding(2, width)
    packed = FiniteFieldPolynomialEncoding(degree, field).register_type()
    assert (
        packed.parameters
        == (("polynomial_degree", degree),) + field.register_type().parameters
    )
    layout = FieldPolynomialLayout(degree, width)
    split = field_polynomial_component_program(
        layout, field.register_type(), packed_dtype=packed
    )
    join = field_polynomial_component_program(
        layout, field.register_type(), reverse=True, packed_dtype=packed
    )
    source = GFPolySplit(QGFPoly(degree, QGF(2, width)))
    for native, original in zip(
        split.functions[0].registers or (), source.signature, strict=True
    ):
        assert (native.name, native.bitsize, native.shape, native.side.name) == (
            original.name,
            original.bitsize,
            original.shape,
            original.side.name,
        )
    signature = Signature((Register("p", packed, shape),))
    builder = ComponentProgramBuilder(signature)
    words = np.asarray(builder.inputs["p"], dtype=object)
    outputs = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        coefficients = builder.call_program(split, {"reg": words[index]})["reg"]
        outputs[index] = builder.call_program(join, {"reg": coefficients})["reg"]
    record = builder.finalize({"p": outputs if shape else outputs[()]})
    assert verify_program_record(record) == ()
    size = layout.logical_width
    inputs = tuple(tuple(range(i * size, (i + 1) * size)) for i in range(words.size))
    if not inputs:
        with pytest.raises(ValueError, match="at least one physical wire"):
            bind_register_program(record, inputs)
    bound = bind_register_program(record, inputs, clean=(size * words.size,))
    assert bound.outputs == inputs
    assert not bound.circuit.instructions
    assert not bound.events and not bound.classical_outputs
    if not shape and size <= 4:
        # A spectator is populated and one caller-owned spare wire starts clean.
        count = size + 2
        state = torch.zeros(1 << count, dtype=torch.complex128)
        real = torch.arange(1, (1 << (size + 1)) + 1, dtype=torch.float64)
        state[::2] = torch.complex(real, real.flip(0))
        state /= state.norm()
        result = (
            Circuit(count, inputs=state, dtype=state.dtype)
            .append(FieldOperation(count, bound.circuit.instructions))
            .state()
            .flatten()
        )
        torch.testing.assert_close(result, state, atol=1e-12, rtol=0)
        assert torch.count_nonzero(result[1::2]) == 0


def test_nominal_coefficient_field_identity_and_mismatches() -> None:
    from tests.unit.test_field_frobenius_component import dtype

    coefficient = dtype(3, 11)
    packed = replace(
        polynomial_type(1, 3),
        parameters=(("polynomial_degree", 1),) + coefficient.parameters,
    )
    layout = FieldPolynomialLayout(1, 3)
    record = field_polynomial_component_program(
        layout, coefficient, packed_dtype=packed
    )
    assert record.functions[0].registers is not None
    assert record.functions[0].registers[0].dtype == packed
    for invalid in (
        replace(packed, quantum_bits=5),
        polynomial_type(1, 3),
        polynomial_type(0, 3),
    ):
        with pytest.raises(ValueError, match="metadata must match"):
            field_polynomial_component_program(
                layout, coefficient, packed_dtype=invalid
            )
    with pytest.raises(ValueError, match="metadata must match"):
        field_polynomial_component_program(layout, dtype(3, 13), packed_dtype=packed)
    # Existing call compatibility rejects distinct nominal polynomial fields.
    builder = ComponentProgramBuilder(
        Signature((Register("p", polynomial_type(1, 3)),))
    )
    with pytest.raises(ValueError):
        builder.call_program(record, {"reg": builder.inputs["p"]})


def test_consumed_polynomial_handle_cannot_be_reused() -> None:
    from flagquantum.core.ir.program.register_bindings import RegisterBinding

    packed = polynomial_type(1, 2)
    builder = ComponentProgramBuilder(Signature((Register("p", packed),)))
    before = cast(RegisterBinding, builder.inputs["p"])
    builder.split(before)
    with pytest.raises(ValueError):
        builder.split(before)


@pytest.mark.parametrize("shape", [(), (2,), (0,)])
def test_existing_cast_roundtrip_preserves_polynomial_metadata(
    shape: tuple[int, ...],
) -> None:
    from flagquantum.compiler.registers.layout.component_cast import cast_word_function
    from flagquantum.core.ir.program.register_types import opaque_register_type

    packed = polynomial_type(1, 2)
    opaque = opaque_register_type(4)
    forward = cast_word_function("unpack", packed, opaque, shape)
    reverse = cast_word_function("repack", opaque, packed, shape)
    builder = ComponentProgramBuilder(
        Signature((Register("p", packed, shape),)), (forward, reverse)
    )
    output = builder.call_dict("unpack", {"reg": builder.inputs["p"]})
    output = builder.call_dict("repack", output)
    inputs = tuple(tuple(range(4 * i, 4 * i + 4)) for i in range(prod(shape)))
    bound = bind_register_program(
        builder.finalize({"p": output["reg"]}), inputs, clean=(4 * prod(shape),)
    )
    assert bound.outputs == inputs
    assert not bound.circuit.instructions
    assert not bound.events and not bound.classical_outputs


@pytest.mark.parametrize("reverse", [False, True])
def test_typed_view_control_callback_preserves_controls(reverse: bool) -> None:
    from flagquantum.compiler.registers.control.component_bookkeeping import (
        bookkeeping_component_control_system,
    )
    from flagquantum.core.ir.program.register_types import quantum_bit_type

    coefficient = FiniteFieldEncoding(2, 2).register_type()
    packed = replace(
        polynomial_type(1, 2),
        parameters=(("polynomial_degree", 1),) + coefficient.parameters,
    )
    view = field_polynomial_component_program(
        FieldPolynomialLayout(1, 2),
        coefficient,
        "view",
        reverse=reverse,
        packed_dtype=packed,
    )
    selected, callback = bookkeeping_component_control_system(view, "view")
    assert selected is view
    registers = view.functions[0].registers
    assert registers is not None
    builder = ComponentProgramBuilder(
        Signature((Register("controls", quantum_bit_type(), (2,)), *registers)),
        view.functions,
    )
    controls = (builder.inputs["controls"],)
    returned, (output,) = callback(builder, controls, {"reg": builder.inputs["reg"]})
    assert returned is controls
    record = builder.finalize({"controls": returned[0], "reg": output})
    inputs = ((0,), (1,), (5, 2), (4, 3)) if reverse else ((0,), (1,), (5, 2, 4, 3))
    bound = bind_register_program(record, inputs)
    assert bound.outputs == (
        ((0,), (1,), (5, 2, 4, 3)) if reverse else ((0,), (1,), (5, 2), (4, 3))
    )
    assert not bound.circuit.instructions and not bound.events
