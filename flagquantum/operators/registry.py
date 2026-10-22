"""The operator registration table and the one entry point that extends it.

Registration is a pure function from a table to a new table. It never writes into
:data:`flagquantum.core.operator_schema.OPERATOR_SCHEMAS`, which is a read-only
mapping of the built-in opcodes, so a registration cannot change what an already
imported module sees through that table, cannot depend on which registrations ran
first when tables are built, and cannot disagree with the checked-in
``docs/operator_manifest.json``.

What a registration does *not* do is make an opcode known to the IR. The IR
resolves names through the built-in table only, so a registered opcode reaches a
program the same way any other custom operation does: with an explicit matrix.
That is deliberate -- :class:`flagquantum.core.ir.Instruction` is the serialized
boundary, and an instruction whose name only a runtime table knows could not be
validated by a reader that does not hold that table.

Registration does install the spelling, though: it gives ``Circuit`` a method
named after the declared opcode and each of its aliases, so a declared operator is
written the way a built-in one is. That installation is the one process-global
effect in this module, it is taken at call time and never at import time, and it is
what makes ``Circuit(2).cphase_quarter(0, 1, matrix=...)`` mean the same thing as
``Circuit(2).gate("cphase_quarter", (0, 1), matrix=...)``. A name that ``Circuit``
already carries is refused rather than overwritten, so a declaration cannot shadow
``Circuit.run`` or replace an earlier declaration's spelling.
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

        This is the registry's own resolution by string. ``Circuit`` resolves a
        gate name through the built-in table, but a registration installs a method
        for the opcode and for each alias, and those methods emit the canonical
        opcode, so the declared spelling reaches a program even though the string
        lookup here is what knows the alias.
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


def _spelling_surface(schema: OperatorSchema) -> tuple[object, ...]:
    """The parts of a declaration its installed method actually binds.

    A spelling's method takes an arity, a parameter list and a set of qubit
    keywords, and emits the opcode. Nothing else in the declaration is readable
    from it, so two declarations that agree on these four fields install the same
    method and one of them is a no-op. Fields the method cannot see -- ``adjoint``
    and ``aliases`` -- are left out on purpose: refusing them would make an
    unrelated difference in a second declaration fail a registration that would
    have behaved identically.
    """

    return (
        schema.opcode,
        schema.arity,
        schema.parameters,
        schema.qubit_convention,
    )


def _install_circuit_methods(schema: OperatorSchema) -> None:
    """Give ``Circuit`` a method for the opcode and for each declared alias.

    The import is deferred because ``flagquantum.circuit`` is the module that calls
    this at import time for the built-in opcodes; taking it here keeps the
    dependency pointing one way and off the import path.

    A spelling is installed once. Declaring the same operator again with the same
    calling surface is a no-op, so a notebook cell or a test that re-runs a
    registration is not punished for the earlier one. Two cases are refused instead
    of overwritten: a spelling ``Circuit`` already carries for another reason, such
    as ``run``, and a spelling installed by a declaration with a different calling
    surface, because earlier circuits were built with the meaning that declaration
    gave the name.
    """

    from ..circuit import Circuit
    from ..core._gate_methods import install_gate_method

    for name in (schema.opcode, *schema.aliases):
        for spelling in (name, name.upper()):
            installed = getattr(Circuit, spelling, None)
            if installed is None:
                continue
            declared = getattr(installed, "_fq_operator_schema", None)
            if declared is None:
                raise ValueError(
                    f"operator {schema.opcode!r} cannot be registered: Circuit "
                    f"already carries {spelling!r}, and a declaration does not "
                    "replace an existing Circuit attribute"
                )
            if _spelling_surface(declared) != _spelling_surface(schema):
                raise ValueError(
                    f"operator {schema.opcode!r} is already installed on Circuit as "
                    f"{spelling!r} with a different arity, parameter list or qubit "
                    "names; declare it once, or give this operator a name of its own"
                )
        if getattr(Circuit, name, None) is None:
            install_gate_method(Circuit, name, schema)


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
    """Declare one operator, return a new registry, and install its spelling.

    The declaration is everything a name needs to be checked, drawn, inverted and
    lowered, and nothing more. It carries no matrix: the operand is supplied at
    the call that emits the operation, which is where its size is known.

    A registration is a declaration, not an extension of the IR's vocabulary, so
    an instruction naming a registered opcode still has to carry its matrix. See
    the module docstring.

    The returned table is a value, but the spelling is installed on ``Circuit``:
    the opcode and each alias become methods that take the declared number of
    qubits, the declared qubit names as keywords, and the declared parameter names,
    and that emit the canonical opcode. So after

    ``register_operator("cphase_quarter", arity=2)`` a circuit may say
    ``circuit.cphase_quarter(0, 1, matrix=matrix)`` exactly as it would say
    ``circuit.cx(0, 1)``. Two consequences follow and are both deliberate. The
    installation lasts for the life of the process, because ``Circuit`` installs
    its methods once at import. And a spelling ``Circuit`` already carries -- a
    built-in gate, an alias of one, or anything else such as ``run`` -- is refused
    rather than overwritten.

    Args:
        opcode: The name to register. Case-insensitive; stored lower case.
        arity: How many qubits one use of the operator spans.
        parameters: The angle names the operator accepts, in order. A registered
            operator declares no parameter frequencies, so it is not reported as
            differentiable: a frequency set is a measurement of the gate, and this
            entry point does not take one.
        aliases: Other names this registry should resolve to ``opcode``, and from
            which methods are installed too.
        adjoint: One of :data:`flagquantum.core.operator_schema.ADJOINT_RULES`.
            The default says the declaration alone does not determine the inverse,
            which is true of every operator whose matrix arrives at the call.
        qubit_convention: The name of each qubit, in order, and the keyword each
            of those qubits accepts in the installed method. Omitted means
            positional names ``qubit_0`` to ``qubit_{arity-1}``, because
            registration knows no semantics beyond position; a built-in's names
            say ``control`` and ``target``, which is knowledge this call does
            not have.
        registry: The table to extend. Omitted means the built-in table.

    Returns:
        A new registry. The argument is unchanged.

    Raises:
        ValueError: If the opcode is empty, already taken or already carried by
            ``Circuit``; the arity is not a positive integer; a parameter or alias
            repeats; the adjoint rule is not one of the declared rules; or the
            declared convention does not name exactly ``arity`` qubits.

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
    extended = base.with_operator(schema)
    _install_circuit_methods(schema)
    return extended


__all__ = ["OperatorRegistry", "register_operator"]
