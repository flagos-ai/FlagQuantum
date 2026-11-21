# Copyright 2023 Google LLC
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 reference_docs.py.

"""Optional template page adapters for the installed documentation generator."""

from collections.abc import Iterable
from copy import copy
from pathlib import Path
from types import MethodType
from typing import Any

import jinja2
from tensorflow_docs.api_generator import parser
from tensorflow_docs.api_generator.obj_type import ObjType
from tensorflow_docs.api_generator.pretty_docs import (
    ClassPageBuilder,
    ClassPageInfo,
    FunctionPageBuilder,
    FunctionPageInfo,
    ModulePageBuilder,
    ModulePageInfo,
    TypeAliasPageBuilder,
    TypeAliasPageInfo,
)
from tensorflow_docs.api_generator.pretty_docs.class_page import Methods


def filter_and_sort_members(py_object: object, members: Iterable[Any]) -> list[Any]:
    ordering = {name: index for index, name in enumerate(py_object.__dict__)}
    return sorted(
        (member for member in members if member.short_name in ordering),
        key=lambda member: ordering[member.short_name],
    )


def template_mixin(template_name: str) -> type[Any]:
    """Bind the source template policy to the native template asset directory."""
    paths = (str(Path(__file__).parent),)
    return type(
        "ReferenceTemplate",
        (),
        {
            "TEMPLATE": f"templates/{template_name}.jinja",
            "TEMPLATE_SEARCH_PATH": paths,
            "JINJA_ENV": jinja2.Environment(
                trim_blocks=True,
                lstrip_blocks=True,
                autoescape=True,
                loader=jinja2.FileSystemLoader(paths),
            ),
        },
    )


class DescriptionListItem:
    """Supply the source's item formatter for description-list rendering."""

    @staticmethod
    def format(*, name: str, anchor: str, description: str) -> str:
        description_lines = description.splitlines() or ["&nbsp;"]
        return "\n".join(
            [f"{name}{anchor}", f": {description_lines[0]}"]
            + [f"  {line}" for line in description_lines[1:]]
            + ["\n"]
        )


def _description_table_view(
    block: Any, title_template: str | None = None, anchors: bool = True
) -> str:
    title = (
        title_template.format(title=block.title)
        if title_template is not None
        else str(block.title)
    )
    items = []
    for name, description in block.items:
        description = (
            block._dedent_after_first_line(description.strip("\n"))
            if description
            else ""
        )
        anchor = f'<a id="{name}"></a>' if anchors else ""
        items.append(
            DescriptionListItem.format(
                name=f"`{name}`", anchor=anchor, description=description
            )
        )
    # The frozen table template deliberately omits the parsed freeform text.
    return "\n\n" + title + "\n\n" + "".join(items) + "\n"


def _other_member_section(builder: Any) -> str | None:
    members = builder.page_info.other_members
    if not members:
        return None if builder.member_caption == "Class Variables" else ""
    items = []
    for member in members:
        description = [member.doc.brief]
        description.extend(str(part) for part in member.doc.docstring_parts)
        items.append(
            DescriptionListItem.format(
                name=member.short_name,
                anchor=f'<a id="{member.short_name}"></a>',
                description="\n".join(description),
            )
        )
    return f'\n\n<h2 class="add-link">{builder.member_caption}</h2>\n\n' + "".join(
        items
    )


def _local_description_blocks(value: Any) -> Any:
    if isinstance(value, parser.TitleBlock):
        result = copy(value)
        result.table_view = MethodType(_description_table_view, result)
        return result
    if isinstance(value, parser.DocstringInfo):
        return value._replace(
            docstring_parts=[
                _local_description_blocks(part) for part in value.docstring_parts
            ]
        )
    if isinstance(value, list):
        return [_local_description_blocks(item) for item in value]
    if isinstance(value, tuple) and "doc" in getattr(value, "_fields", ()):
        return getattr(value, "_replace")(
            doc=_local_description_blocks(getattr(value, "doc"))
        )
    return value


def _build_page(page: Any) -> str:
    # Copy only owned documentation fields. Parser configuration and documented
    # Python objects retain their original identities.
    for field in (
        "_doc",
        "_methods",
        "_classes",
        "_functions",
        "_modules",
        "_type_alias",
        "_other_members",
        "_bases",
        "attr_block",
    ):
        if field in vars(page):
            setattr(page, field, _local_description_blocks(getattr(page, field)))
    result: str = page.DEFAULT_BUILDER_CLASS(page).build()
    return result


def _collect_module_docs(page: Any) -> None:
    ModulePageInfo.collect_docs(page)
    page._classes = filter_and_sort_members(page.py_object, page._classes)


def _collect_class_docs(page: Any) -> None:
    ClassPageInfo.collect_docs(page)
    page._methods = filter_and_sort_members(page.py_object, page._methods)


def _initialize_class_builder(builder: Any, page_info: Any) -> None:
    ClassPageBuilder.__init__(builder, page_info)
    methods = filter_and_sort_members(
        builder.page_info.py_object, builder.methods.info_dict.values()
    )
    builder.methods = Methods(
        info_dict={method.short_name: method for method in methods},
        constructor=builder.methods.constructor,
    )


def page_info_classes() -> dict[Any, type[Any]]:
    """Compose four concrete SDK plugin classes at the optional boundary.

    TensorFlow-docs supplies untyped runtime bases. The documented factory
    contract accepts their actual Python subclasses, with native typed hooks.
    """
    result = {}
    for kind, name, builder_base, info_base in (
        (ObjType.CLASS, "class", ClassPageBuilder, ClassPageInfo),
        (ObjType.CALLABLE, "function", FunctionPageBuilder, FunctionPageInfo),
        (ObjType.MODULE, "module", ModulePageBuilder, ModulePageInfo),
        (ObjType.TYPE_ALIAS, "type_alias", TypeAliasPageBuilder, TypeAliasPageInfo),
    ):
        builder_hooks: dict[str, Any] = (
            {"__init__": _initialize_class_builder} if kind is ObjType.CLASS else {}
        )
        if kind in (ObjType.CLASS, ObjType.MODULE):
            builder_hooks["build_other_member_section"] = _other_member_section
            builder_hooks["member_caption"] = (
                "Class Variables" if kind is ObjType.CLASS else "Other Members"
            )
        builder = type(
            name.title().replace("_", "") + "TemplateBuilder",
            (template_mixin(name), builder_base),
            builder_hooks,
        )
        hooks: dict[str, Any] = {"DEFAULT_BUILDER_CLASS": builder, "build": _build_page}
        if kind is ObjType.CLASS:
            hooks["collect_docs"] = _collect_class_docs
        elif kind is ObjType.MODULE:
            hooks["collect_docs"] = _collect_module_docs
        result[kind] = type(
            name.title().replace("_", "") + "TemplatePage", (info_base,), hooks
        )
    return result
