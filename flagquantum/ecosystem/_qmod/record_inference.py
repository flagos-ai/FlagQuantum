"""Resolve one unsized record field from an unambiguous whole-record preparation."""

from dataclasses import replace

from .syntax import Function, Parameter, Record, reject


def infer_prepared_record_widths(
    layouts: dict[str, Record], functions: tuple[Function, ...]
) -> bool:
    inferred = {
        name: tuple(field for field in record.fields if field.width is None)
        for name, record in layouts.items()
        if any(field.width is None for field in record.fields)
    }
    if not inferred:
        return False
    widths: dict[str, set[int]] = {name: set() for name in inferred}
    for function in functions:
        records = {port.name.text: port.record for port in function.parameters}
        for statement in function.statements:
            if statement.declaration is not None:
                local = statement.declaration
                records[local.name.text] = local.record
            elif (
                statement.token.text in {"prepare_amplitudes", "prepare_state"}
                and len(statement.arguments) == 3
            ):
                amplitudes, _, target = statement.arguments
                name = records.get(target.token.text)
                size = len(amplitudes.children)
                if (
                    name is not None
                    and name in widths
                    and not target.children
                    and amplitudes.token.text == "["
                    and size >= 2
                    and not size & (size - 1)
                ):
                    widths[name].add(size.bit_length() - 1)

    for name, unknown in inferred.items():
        layout = layouts[name]
        field = unknown[0]
        if (
            len(unknown) != 1
            or any(item.record for item in layout.fields)
            or len(widths[name]) != 1
        ):
            reject(
                field.name,
                "width",
                "record numeric fields require a fixed width or one unambiguous whole-record preparation",
            )
        [width] = widths[name]
        remaining = width - layout.width
        if not 1 <= remaining <= 256 or width > 256:
            reject(
                field.name,
                "width",
                "whole-record preparation cannot fit the fixed fields",
            )
        layouts[name] = replace(
            layout,
            fields=tuple(
                replace(item, width=remaining) if item is field else item
                for item in layout.fields
            ),
        )

    return True


def bind_record_widths(
    layouts: dict[str, Record], functions: tuple[Function, ...]
) -> tuple[Function, ...]:
    def refresh(port: Parameter) -> Parameter:
        if port.record is not None:
            return replace(port, width=layouts[port.record].width)
        return port

    return tuple(
        replace(
            function,
            parameters=tuple(refresh(port) for port in function.parameters),
            statements=tuple(
                (
                    replace(statement, declaration=refresh(statement.declaration))
                    if statement.declaration is not None
                    else statement
                )
                for statement in function.statements
            ),
        )
        for function in functions
    )
