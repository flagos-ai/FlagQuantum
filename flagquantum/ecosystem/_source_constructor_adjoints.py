"""Keep exact source-constructor inverse recipes scoped to one resolution."""

from collections.abc import Mapping

from ..compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.registers import Register
from ..errors import SerializationError


def constructor_adjoint_factories(
    forward: Mapping[str, ComponentFactory],
    inverse: Mapping[str, ComponentFactory],
    *,
    fallback: ComponentFactory | None = None,
) -> dict[str, ComponentFactory]:
    """Bind only explicit source pairs; never infer an unknown leaf's inverse."""
    if forward.keys() != inverse.keys():
        raise ValueError("Source constructor directions require identical keys")
    recipes: dict[FunctionRecord, tuple[str, dict[str, ResolvedArgument], bool]] = {}

    def make(key: str, backward: bool) -> ComponentFactory:
        def construct(
            name: str,
            registers: tuple[Register, ...] | None,
            arguments: Mapping[str, ResolvedArgument],
        ) -> FunctionRecord | ProgramRecord:
            factory = inverse[key] if backward else forward[key]
            result = factory(name, registers, arguments)
            root = result.functions[0] if isinstance(result, ProgramRecord) else result
            recipes[root] = key, dict(arguments), backward
            return result

        return construct

    def adjoint(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if set(arguments) == {"subbloq"} and isinstance(child, FunctionRecord):
            if child in recipes:
                key, metadata, backward = recipes[child]
                return make(key, not backward)(name, registers, metadata)
        if fallback is not None:
            return fallback(name, registers, arguments)
        raise SerializationError("Missing exact source constructor inverse context")

    return {
        **{key: make(key, False) for key in forward},
        "qualtran._infra.adjoint.Adjoint": adjoint,
    }
