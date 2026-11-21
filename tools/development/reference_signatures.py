# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 make_reference_docs/_components/render_signature.py.

"""Render static Python function and method signatures for reference pages."""

import re
from collections.abc import Callable

from griffe import Function, Object, Parameter, ParameterKind

from tools.development.reference_links import Writable

BESPOKE_OBJECT_NAMES_FOR_CLASSES = {
    "ComponentProgramBuilder": "builder",
    "ProgramRecord": "program",
}


def format_parameter(parameter: Parameter) -> str:
    text = parameter.name
    if parameter.annotation is not None:
        text += f": {parameter.annotation}"
    if parameter.default is not None:
        text += f" = {parameter.default}"
    if parameter.kind in {
        ParameterKind.positional_or_keyword,
        ParameterKind.positional_only,
        ParameterKind.keyword_only,
    }:
        return text
    if parameter.kind is ParameterKind.var_positional:
        return f"*{text}"
    if parameter.kind is ParameterKind.var_keyword:
        return f"**{text}"
    raise ValueError(parameter.kind)


def camel_to_snake(name: str) -> str:
    return re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name).lower()


def get_obj_instance_name(obj: Object) -> str:
    return BESPOKE_OBJECT_NAMES_FOR_CLASSES.get(obj.name, camel_to_snake(obj.name))


def write_property_method_signature(f: Writable, obj: Object, method: Function) -> None:
    assert method.returns, method
    signature = f"{get_obj_instance_name(obj)}.{method.name} -> {method.returns}"
    f.write(f"```python\n{signature}\n```\n\n")


def write_generic_method_signature(
    f: Writable,
    obj: Object,
    method: Function,
    first_arg_name: str | None = "self",
    caller_name: Callable[[Object], str] = get_obj_instance_name,
) -> None:
    parameters = list(method.parameters)
    if first_arg_name is not None:
        first = parameters[0]
        assert first.name == first_arg_name, first
        parameters = parameters[1:]
    if len(parameters) > 1:
        text = "\n" + ",\n".join(f"  {format_parameter(p)}" for p in parameters) + "\n"
    else:
        text = ", ".join(format_parameter(p) for p in parameters)
    signature = f"{caller_name(obj)}.{method.name}({text})"
    if method.returns is not None:
        signature += f" -> {method.returns}"
    f.write(f"```python\n{signature}\n```\n\n")


def write_function_signature(f: Writable, function: Function) -> None:
    parameters = list(function.parameters)
    if len(parameters) > 1:
        text = "\n" + ",\n".join(f"  {format_parameter(p)}" for p in parameters) + "\n"
    else:
        text = ", ".join(format_parameter(p) for p in parameters)
    signature = f"{function.name}({text})"
    if function.returns is not None:
        signature += f" -> {function.returns}"
    f.write(f"```python\n{signature}\n```\n\n")


def write_method_signature(f: Writable, obj: Object, method: Function) -> None:
    if method.overloads:
        return
    decorators = [
        decorator
        for decorator in method.decorators
        if str(decorator.value) != "abc.abstractmethod"
    ]
    if not decorators:
        write_generic_method_signature(f, obj, method)
        return
    if len(decorators) > 1:
        raise ValueError(
            f"{obj.name}.{method.name} has multiple decorators. "
            "Please upgrade `write_method_signature` in the doc generation code to handle this."
        )
    decorator = decorators[0]
    name = str(decorator.value)
    if name in {"property", "cached_property"}:
        write_property_method_signature(f, obj, method)
    elif name == "classmethod":
        write_generic_method_signature(
            f, obj, method, first_arg_name="cls", caller_name=lambda owner: owner.name
        )
    elif name == "staticmethod":
        write_generic_method_signature(
            f, obj, method, first_arg_name=None, caller_name=lambda owner: owner.name
        )
    else:
        raise ValueError(
            f"{obj.name}.{method.name} has decorator {decorator.value!r}. "
            "Please upgrade `write_method_signature` in the doc generation code to handle this."
        )
