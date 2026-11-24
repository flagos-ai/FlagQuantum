"""Lower explicit native definitions for a shared SDK measurement history."""

from collections.abc import Hashable, Iterable, Mapping
from typing import Any

from ...compiler.registers._local_register_binding import _bind_local_register_program
from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.signatures import Signature
from ._register_gate import RegisterCirqGate
from .conversion import _cirq


def _native_definition_module(
    roots: Iterable[FunctionRecord], declared: Mapping[Hashable, FunctionRecord]
) -> dict[str, FunctionRecord]:
    """Collect called helper definitions while retaining selected root ownership."""
    definitions: dict[str, FunctionRecord] = {}
    for function in roots:
        previous = definitions.setdefault(function.name, function)
        if previous != function:
            raise ValueError(f"Conflicting Cirq gate definition {function.name!r}")
    available: dict[str, list[FunctionRecord]] = {}
    for function in declared.values():
        if isinstance(function, FunctionRecord):
            available.setdefault(function.name, []).append(function)
    pending = list(definitions.values())
    while pending:
        function = pending.pop()
        for block in function.blocks:
            for operation in block.operations:
                name = operation.callee
                if name is None or name in definitions:
                    continue
                candidates = available.get(name, [])
                if not candidates:
                    raise ValueError(f"Missing Cirq native helper definition {name!r}")
                helper = candidates[0]
                if any(candidate != helper for candidate in candidates[1:]):
                    raise ValueError(f"Conflicting Cirq gate definition {name!r}")
                definitions[name] = helper
                pending.append(helper)
    return definitions


def _expand_native_definitions(
    operations: tuple[Any, ...], declared: Mapping[Hashable, FunctionRecord]
) -> tuple[Any, ...]:
    """Preserve native definition precedence and isolate local measurement keys."""
    cirq = _cirq()
    selected: list[FunctionRecord | None] = []
    for operation in operations:
        gate = operation.without_classical_controls().gate
        definition = declared.get(gate) if isinstance(gate, Hashable) else None
        if definition is None:
            definition = getattr(gate, "flagquantum_register_definition", None)
        if definition is not None and (
            not isinstance(definition, FunctionRecord) or definition.registers is None
        ):
            raise TypeError(
                "Cirq gate definition requires a typed native FunctionRecord"
            )
        selected.append(definition)
    definitions = _native_definition_module(
        (function for function in selected if function is not None), declared
    )
    keys = {
        key for operation in operations for key in cirq.measurement_key_names(operation)
    }
    keys.update(
        str(key) for operation in operations for key in cirq.control_keys(operation)
    )
    result = []
    for index, (operation, definition) in enumerate(
        zip(operations, selected, strict=True)
    ):
        if definition is None:
            result.append(operation)
            continue
        prefix = f"fq_native_{index}"
        while any(key.startswith(prefix + "_") for key in keys):
            prefix += "_"
        bound = _bind_local_register_program(
            ProgramRecord(tuple(definitions.values())), definition.name
        )
        assert definition.registers is not None
        gate = RegisterCirqGate(
            bound, Signature(definition.registers), feedback_key_prefix=prefix
        )
        result.append(
            gate.on(*operation.qubits).with_classical_controls(
                *operation.classical_controls
            )
        )
    return tuple(result)
