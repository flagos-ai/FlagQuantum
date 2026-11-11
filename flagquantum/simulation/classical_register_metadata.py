"""Shared concrete register metadata for numerical codecs and value domains."""

from ..core.ir.program.registers import Register


def concrete_register_metadata(register: Register) -> tuple[int, dict[str, int | bool]]:
    """Read the existing finite numerical metadata without changing its parameters."""
    width = register.bitsize
    if type(width) is not int or width < 0:
        raise ValueError("register codecs require a concrete nonnegative bit width")
    parameters: dict[str, int | bool] = {}
    for name, value in register.dtype.parameters:
        if name == "storage" and value == "classical":
            continue
        if name not in {
            "bound",
            "modulus",
            "fractional_width",
            "signed",
        } or not isinstance(value, int):
            raise ValueError(
                "register codec parameters must be concrete native metadata"
            )
        parameters[name] = value
    return width, parameters
