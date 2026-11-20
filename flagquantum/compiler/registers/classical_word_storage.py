"""Bind canonical classical stored words to the shared MSB storage encoding."""

from ...core.ir.program.register_types import classical_bit_type
from ...core.ir.program.register_verifier import executable_classical_register_width
from ...core.ir.program.registers import RegisterType
from .control.control_predicate import ControlRegister, IntegerEncoding


def classical_word_bits(dtype: RegisterType, value: int) -> tuple[int, ...]:
    """Validate physical entry values and reuse existing equality-word encoding.

    Fixed-point storage carries its raw integer. Montgomery storage keeps its
    original raw-width domain. Fields and polynomials carry unsigned stored
    words, with every nonbinary coefficient checked against its field order.
    Numerical callbacks retain Simulation ownership.
    """
    width = executable_classical_register_width(dtype)
    if dtype == classical_bit_type():
        if type(value) not in (int, bool) or value not in (0, 1):
            raise ValueError("external CBit inputs require built-in zero/one bits")
        return (int(value),)
    if type(value) is not int:
        raise ValueError("classical word inputs require a built-in integer")
    parameters = dict(dtype.parameters)
    bound = parameters.get("bound")
    if isinstance(bound, int) and not 0 <= value < bound:
        raise ValueError("classical word value must satisfy its exclusive bound")
    if width == 0:
        if dtype.encoding != "uint" or value != 0:
            raise ValueError("zero-width classical storage requires unsigned zero")
        return ()
    encoding: IntegerEncoding = "unsigned"
    if dtype.encoding == "ones_complement":
        encoding = "ones_complement"
    elif dtype.encoding == "signed" or (
        dtype.encoding == "fixed_point" and parameters["signed"]
    ):
        encoding = "twos_complement"
    bits = ControlRegister(width, (value,), encoding).bits()
    if dtype.encoding in {"finite_field", "finite_field_polynomial"}:
        order = parameters["order"]
        assert type(order) is int
        coefficient_width = (order - 1).bit_length()
        mask = (1 << coefficient_width) - 1
        if any(
            ((value >> offset) & mask) >= order
            for offset in range(0, width, coefficient_width)
        ):
            raise ValueError(
                "classical field coefficients must be below the field order"
            )
    return bits
