"""Compose library constructor adaptation with existing native value codecs."""

from ..compiler.formats.legacy_component_resolution import ComponentFactory
from ..compiler.formats.legacy_word_factories import word_factories
from ..core.ir.program.registers import Register, RegisterType
from ..simulation.classical_register_codec import encode_register_value


def _constant_bits(dtype: RegisterType, value: object) -> tuple[int, ...]:
    return tuple(encode_register_value(Register("x", dtype), value))


def bitwise_library_factories() -> dict[str, ComponentFactory]:
    """Bind concrete interchange constants with the existing numerical codec.

    This boundary converts values to owned MSB patterns before Compiler builds
    typed word calls. It imports no source framework or provider constructor.
    """
    return word_factories(encode_constant=_constant_bits)
