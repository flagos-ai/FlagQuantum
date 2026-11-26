"""Install declared operator spellings through the existing Circuit interface."""

from __future__ import annotations

from importlib import import_module as _import_public_namespace

from flagquantum.core.operators.registry import OperatorRegistry
from flagquantum.core.operators.schema import (
    ADJOINT_RULES,
    OperatorSchema,
)


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

    from flagquantum.circuit import Circuit
    from flagquantum.core._gate_methods import install_gate_method

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


__all__ = ("register_operator",)


# Preserve historical import identities for pickles and public introspection.
for _identity_export in (
    _spelling_surface,
    _install_circuit_methods,
    register_operator,
):
    _identity_export.__module__ = "flagquantum.operators.registry"
del _identity_export

# Bind the historical public namespace for class annotation introspection.

_public_namespace = _import_public_namespace("flagquantum.operators.registry")
for _public_name in (
    "_install_circuit_methods",
    "_spelling_surface",
    "register_operator",
):
    setattr(_public_namespace, _public_name, globals()[_public_name])
del _public_namespace, _public_name
