"""The operator registration table and the one entry point that extends it.

Registration is a pure function from a table to a new table. It never writes into
:data:`flagquantum.core.operator_schema.OPERATOR_SCHEMAS`, which is a read-only
mapping of the built-in opcodes, so a registration cannot change what an already
imported module sees, cannot depend on which registrations ran first, and cannot
disagree with the checked-in ``docs/operator_manifest.json``.

What a registration does *not* do is make an opcode known to the IR. The IR
resolves names through the built-in table only, so a registered opcode reaches a
program the same way any other custom operation does: with an explicit matrix.
That is deliberate -- :class:`flagquantum.core.ir.Instruction` is the serialized
boundary, and an instruction whose name only a runtime table knows could not be
validated by a reader that does not hold that table.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from ..core.operator_schema import (
    ADJOINT_RULES,
    OPERATOR_ALIASES,
    OPERATOR_SCHEMAS,
    OperatorSchema,
)


@dataclass(frozen=True)
class OperatorRegistry:
    """An immutable operator table: the built-ins plus whatever was registered.

    Attributes:
        schemas: Every opcode the table knows, keyed by opcode. Read-only.
    """

    schemas: Mapping[str, OperatorSchema] = field(
        default_factory=lambda: OPERATOR_SCHEMAS
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "schemas", MappingProxyType(dict(self.schemas)))

    @property
    def aliases(self) -> Mapping[str, str]:
        """Alias to opcode, over the built-in table and this one together."""

        merged = dict(OPERATOR_ALIASES)
        merged.update(
            {
                alias: schema.opcode
                for schema in self.schemas.values()
                for alias in schema.aliases
            }
        )
        return MappingProxyType(merged)

    def canonical(self, name: str) -> str:
        """Resolve ``name`` through this table's aliases, or return it unchanged.

        This is the registry's own resolution. :class:`flagquantum.Circuit` and
        :class:`flagquantum.compiler.operator_lowering.OperatorLoweringRegistry`
        resolve through the built-in table, so a registered alias is meaningful
        only to code that was handed this registry.
        """

        normalized = str(name).strip().lower()
        return self.aliases.get(normalized, normalized)

    def with_operator(self, schema: OperatorSchema) -> "OperatorRegistry":
        """Return a new registry that also holds ``schema``.

        Raises:
            ValueError: If the opcode or one of its aliases is already taken by
                this table.
        """

        opcode = str(schema.opcode).strip().lower()
        if opcode in self.schemas or opcode in self.aliases:
            raise ValueError(f"operator schema {opcode!r} is already registered")
        for alias in schema.aliases:
            if alias in self.schemas or alias in self.aliases or alias == opcode:
                raise ValueError(
                    f"operator schema {opcode!r} declares alias {alias!r}, which "
                    "this table already resolves"
                )
        updated = dict(self.schemas)
        updated[opcode] = schema
        return replace(self, schemas=updated)

    def registered(self) -> tuple[str, ...]:
        """The opcodes this table holds that the built-in table does not."""

        return tuple(sorted(set(self.schemas) - set(OPERATOR_SCHEMAS)))


def register_operator(
    opcode: str,
    *,
    arity: int,
    parameters: tuple[str, ...] = (),
    aliases: tuple[str, ...] = (),
    adjoint: str = "matrix_adjoint",
    qubit_convention: tuple[str, ...] | None = None,
    registry: OperatorRegistry | None = None,
) -> OperatorRegistry:
    """Return a new registry that declares one operator.

    The declaration is everything a name needs to be checked, drawn, inverted and
    lowered, and nothing more. It carries no matrix: the operand is supplied at
    the call that emits the operation, which is where its size is known.

    A registration is a declaration, not an extension of the IR's vocabulary, so
    an instruction naming a registered opcode still has to carry its matrix. See
    the module docstring.

    Args:
        opcode: The name to register. Case-insensitive; stored lower case.
        arity: How many qubits one use of the operator spans.
        parameters: The angle names the operator accepts, in order. A registered
            operator declares no parameter frequencies, so it is not reported as
            differentiable: a frequency set is a measurement of the gate, and this
            entry point does not take one.
        aliases: Other names this registry should resolve to ``opcode``.
        adjoint: One of :data:`flagquantum.core.operator_schema.ADJOINT_RULES`.
            The default says the declaration alone does not determine the inverse,
            which is true of every operator whose matrix arrives at the call.
        qubit_convention: The name of each qubit, in order. Omitted means
            positional names ``qubit_0`` to ``qubit_{arity-1}``, because
            registration knows no semantics beyond position; a built-in's names
            say ``control`` and ``target``, which is knowledge this call does
            not have.
        registry: The table to extend. Omitted means the built-in table.

    Returns:
        A new registry. The argument is unchanged.

    Raises:
        ValueError: If the opcode is empty or already taken, the arity is not a
            positive integer, a parameter or alias repeats, the adjoint rule is
            not one of the declared rules, or the declared convention does not
            name exactly ``arity`` qubits.

    Examples:
        >>> from flagquantum.operators import register_operator
        >>> registry = register_operator(
        ...     "cphase_quarter", arity=2, adjoint="self_inverse"
        ... )
        >>> registry.schemas["cphase_quarter"].arity
        2
        >>> registry.canonical("CPHASE_QUARTER")
        'cphase_quarter'
    """

    normalized = str(opcode).strip().lower()
    if not normalized:
        raise ValueError("an operator opcode must be a non-empty name")
    if isinstance(arity, bool) or not isinstance(arity, int) or arity < 1:
        raise ValueError(
            f"operator {normalized!r} needs a positive integer arity, got {arity!r}"
        )
    declared_parameters = tuple(str(name) for name in parameters)
    if len(set(declared_parameters)) != len(declared_parameters):
        raise ValueError(
            f"operator {normalized!r} declares repeated parameter name(s) "
            f"{declared_parameters}"
        )
    declared_aliases = tuple(str(name).strip().lower() for name in aliases)
    if len(set(declared_aliases)) != len(declared_aliases):
        raise ValueError(
            f"operator {normalized!r} declares repeated alias(es) {declared_aliases}"
        )
    if adjoint not in ADJOINT_RULES:
        raise ValueError(
            f"operator {normalized!r} declares adjoint rule {adjoint!r}, which is "
            f"not one of {ADJOINT_RULES}"
        )
    names = (
        tuple(f"qubit_{index}" for index in range(arity))
        if qubit_convention is None
        else tuple(str(name) for name in qubit_convention)
    )
    if len(names) != arity:
        raise ValueError(
            f"operator {normalized!r} spans {arity} qubit(s) and its "
            f"convention names {len(names)} of them: {names}"
        )
    schema = OperatorSchema(
        opcode=normalized,
        aliases=declared_aliases,
        arity=arity,
        parameters=declared_parameters,
        dtype_policy=("complex64", "complex128"),
        semantic_kind="unitary",
        adjoint=adjoint,
        decomposition=(),
        qubit_convention=names,
    )
    base = OperatorRegistry() if registry is None else registry
    return base.with_operator(schema)


__all__ = ["OperatorRegistry", "register_operator"]
