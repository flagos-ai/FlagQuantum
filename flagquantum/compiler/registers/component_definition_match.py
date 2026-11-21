"""Match stored definitions while preserving every field except function names."""

from dataclasses import replace

from ...core.ir.program import FunctionRecord, ProgramRecord, verify_program_record


def _named_definitions(record: ProgramRecord, entry: str) -> tuple[FunctionRecord, ...]:
    functions = {function.name: function for function in record.functions}
    if entry not in functions:
        raise ValueError(f"Unknown component definition {entry!r}")
    diagnostics = verify_program_record(record)
    if diagnostics:
        raise ValueError(f"Invalid component definitions: {diagnostics}")
    names = {entry: "function_0"}
    pending = [entry]
    result = []
    for name in pending:
        function = functions[name]
        for block in function.blocks:
            for operation in block.operations:
                if operation.callee is not None and operation.callee not in names:
                    names[operation.callee] = f"function_{len(names)}"
                    pending.append(operation.callee)
        result.append(
            replace(
                function,
                name=names[name],
                blocks=tuple(
                    replace(
                        block,
                        operations=tuple(
                            (
                                replace(operation, callee=names[operation.callee])
                                if operation.callee is not None
                                else operation
                            )
                            for operation in block.operations
                        ),
                    )
                    for block in function.blocks
                ),
            )
        )
    return tuple(result)


def matches_component_definition(
    record: ProgramRecord, entry: str, template: ProgramRecord
) -> bool:
    """Compare a reachable definition with an existing construction literally.

    Only function names and unreachable definitions are ignored. Port metadata,
    function sharing, SSA values, call order, operands, parameters, feedback,
    allocation and physical context must match exactly. Canonical payload is
    already excluded from Core Program identity. This query neither synthesizes
    nor executes a circuit and makes no general unitary-equivalence claim.
    Invalid modules and missing entries raise ValueError.
    """
    if not template.functions:
        raise ValueError("Component template must contain an entry definition")
    actual = _named_definitions(record, entry)
    expected = _named_definitions(template, template.functions[0].name)
    return record.physical_context == template.physical_context and actual == expected
