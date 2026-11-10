"""Binary field words use existing typed split/join and physical execution."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_rebindings import split_word_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.compiler.registers.register_programs import lower_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_verifier import executable_register_width
from flagquantum.core.ir.program.registers import RegisterType

pytestmark = pytest.mark.unit


def binary_type(width: int) -> RegisterType:
    return RegisterType(
        "finite_field",
        width,
        parameters=(("characteristic", 2), ("degree", width), ("order", 2**width)),
    )


@pytest.mark.parametrize("width", [1, 2, 3, 8])
def test_binary_field_split_is_a_zero_gate_msb_first_rebinding(width: int) -> None:
    dtype = binary_type(width)
    record = ProgramRecord((split_word_function("main", dtype),))
    assert verify_program_record(record) == ()
    bound = bind_register_program(record, (tuple(reversed(range(width))),))
    assert bound.outputs == tuple((i,) for i in reversed(range(width)))
    assert bound.circuit.instructions == ()
    assert lower_register_program(record).functions[0].blocks[0].operations == ()
    state = torch.randn(
        1 << (width + 1),
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(108430),
    )
    state /= state.norm()
    actual = (
        Circuit(width + 1, inputs=state, dtype=torch.complex128)
        .append(FieldOperation(width + 1, bound.circuit.instructions))
        .state()
        .flatten()
    )
    torch.testing.assert_close(actual, state, rtol=0, atol=0)


@pytest.mark.parametrize(
    "defect",
    [
        "zero",
        "symbolic_degree",
        "classical",
        "prime",
        "order",
        "partial_poly",
        "unknown",
        "bool_degree",
        "duplicate",
    ],
)
def test_binary_field_execution_rejects_unresolved_or_unsupported_profile(
    defect: str,
) -> None:
    dtype = binary_type(2)
    fields = dict(dtype.parameters)
    if defect == "zero":
        dtype = binary_type(0)
    elif defect == "classical":
        dtype = replace(dtype, quantum_bits=0, classical_bits=2)
    elif defect == "symbolic_degree":
        import sympy

        fields["degree"] = sympy.Symbol("m")
        dtype = replace(dtype, parameters=tuple(fields.items()))
    elif defect == "prime":
        fields["characteristic"] = 3
        dtype = replace(dtype, parameters=tuple(fields.items()))
    elif defect == "order":
        fields["order"] = 8
        dtype = replace(dtype, parameters=tuple(fields.items()))
    elif defect == "bool_degree":
        fields["degree"] = True
        dtype = replace(dtype, parameters=tuple(fields.items()))
    else:
        addition = {
            "partial_poly": ("polynomial", 7),
            "unknown": ("unresolved", 1),
            "duplicate": ("degree", 2),
        }[defect]
        dtype = replace(dtype, parameters=(*dtype.parameters, addition))
    with pytest.raises(ValueError, match="unsupported executable"):
        executable_register_width(dtype)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [1, 2, 3, 8])
def test_original_binary_field_projection_and_split_join(width: int) -> None:
    import numpy as np
    from qualtran import QGF
    from qualtran.bloqs.bookkeeping import Join, Split
    from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding

    field = FiniteFieldEncoding(2, width)
    dtype = field.register_type()
    assert executable_register_width(dtype) == width
    source = QGF(2, width)
    assert source.bitsize == width
    for value in range(1 << width):
        bits = Split(source).on_classical_vals(reg=source.gf_type(value))["reg"]
        np.testing.assert_array_equal(bits, source.to_bits(source.gf_type(value)))
        assert int(Join(source).on_classical_vals(reg=bits)["reg"]) == value
        assert field.to_bits(field.gf_type(value)) == list(bits)
