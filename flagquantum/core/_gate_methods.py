"""The bound gate methods a circuit spelling is made of.

`flagquantum.Circuit` installs one method per built-in opcode and alias when this
package is imported, and :func:`flagquantum.operators.register_operator` installs
one per declared operator and alias when the declaration is made. The machinery
lives here rather than beside the loop so that a class carrying a hundred and
sixty attributes does not also carry the argument binder for them, and so that both
callers install through the same binder instead of agreeing by inspection.

The module takes the class it installs onto as an argument: it is not a second
`Circuit`, and nothing here holds a reference to one.
"""

from __future__ import annotations

from typing import Any

from ..errors import ValidationError
from .operator_schema import OPERATOR_SCHEMAS, OperatorSchema, get_operator_schema

#: Opcodes whose two qubits are a control and a target, so the documented public
#: keywords are ``control``/``target`` rather than the positional pair.
_CONTROLLED_TWO_QUBIT_GATES = {
    "cx",
    "cy",
    "cz",
    "crx",
    "cry",
    "crz",
    "cphase",
}


def _public_qubit_keywords(opcode: str, arity: int) -> tuple[tuple[str, ...], ...]:
    """Return public keyword aliases without changing internal qubit vocabulary."""

    if arity == 1:
        return (("qubit", "target"),)
    if arity == 2 and opcode in _CONTROLLED_TWO_QUBIT_GATES:
        return (("control",), ("target",))
    if arity == 2:
        return (("qubit1", "left"), ("qubit2", "right"))
    if opcode == "ccx":
        return (("control1",), ("control2",), ("target",))
    if opcode == "cswap":
        return (("control",), ("target1",), ("target2",))
    return tuple((f"qubit{index + 1}",) for index in range(arity))


def install_gate_method(
    circuit_class: type,
    name: str,
    schema: OperatorSchema | None = None,
) -> None:
    """Give ``circuit_class`` a method that spells the operator ``name``.

    ``schema`` is the declaration the method binds. Omitting it resolves ``name``
    through the built-in table, which is how the import-time loop installs every
    built-in opcode and alias; ``register_operator`` passes its own schema
    instead, because a registered opcode is not in that table and its arity,
    parameter names and qubit names are knowable only from the declaration.

    The method is installed as both ``name`` and ``name.upper()``.
    """

    resolved = get_operator_schema(name) if schema is None else schema
    if resolved is None:
        raise ValidationError(f"cannot install a method for unknown opcode {name!r}")
    opcode = resolved.opcode
    # A registered operator names its qubits in its declaration, so the declared
    # names are the keywords it accepts. A built-in keeps the documented public
    # keywords, which are not the internal convention verbs.
    qubit_keywords = (
        _public_qubit_keywords(opcode, resolved.arity)
        if opcode in OPERATOR_SCHEMAS
        else tuple((qubit,) for qubit in resolved.qubit_convention)
    )
    parameter_names = resolved.parameters

    def method(self: Any, *args: Any, **kwargs: Any) -> Any:
        arity = resolved.arity
        positional = list(args)
        qubits: list[Any | None] = [None] * arity
        for index, aliases in enumerate(qubit_keywords):
            supplied = [alias for alias in aliases if alias in kwargs]
            if len(supplied) > 1:
                rendered = ", ".join(supplied)
                raise TypeError(
                    f"{name} got multiple names for qubit {index}: {rendered}"
                )
            if supplied:
                qubits[index] = kwargs.pop(supplied[0])
        for index, qubit in enumerate(qubits):
            if qubit is None and positional:
                qubits[index] = positional.pop(0)
        resolved_qubits = []
        for qubit in qubits:
            if qubit is None:
                expected = ", ".join(aliases[0] for aliases in qubit_keywords)
                raise TypeError(f"{name} requires qubit arguments: {expected}")
            resolved_qubits.append(qubit)

        values = positional
        if len(values) > len(parameter_names):
            raise TypeError(
                f"{name} accepts {arity} qubit(s) and "
                f"{len(parameter_names)} parameter(s)"
            )
        # Positional values may cover only a prefix of the schema: the remaining
        # parameters are allowed to arrive as keywords, as in `rx(0, theta=0.5)`.
        # A short `values` is therefore expected here, and the missing names are
        # caught by `IRValidationError` below rather than by `strict=True`.
        for parameter_name, value in zip(parameter_names, values, strict=False):
            if parameter_name in kwargs:
                raise TypeError(f"{name} got multiple values for {parameter_name!r}")
            kwargs[parameter_name] = value
        # The method a spelling installs emits the canonical opcode, so an alias
        # and the opcode it names produce the same instruction name.
        return self.gate(opcode, resolved_qubits, **kwargs)

    method.__name__ = name
    # The declaration the method binds, so a later registration can tell its own
    # spelling from a collision instead of overwriting one.
    setattr(method, "_fq_operator_schema", resolved)
    setattr(circuit_class, name, method)
    setattr(circuit_class, name.upper(), method)


__all__ = ["install_gate_method"]
