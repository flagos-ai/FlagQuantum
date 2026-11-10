"""Owned root locals retain their quantum environment after explicit drop."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from .._hybrid.model import Operation
from .frontend_ports import _annotation_node, _resolve, quantum_port
from .frontend_python import _error
from .program_types import QArray, QBit, QNum

if TYPE_CHECKING:
    from .frontend_program import _Body


def require_root_storage(
    owner: _Body, node: ast.expr | ast.stmt, *, unitary_region: bool = False
) -> None:
    if (
        owner.name != owner.owner.entry
        or owner.aliases is not None
        or owner.region_depth
        and not unitary_region
        or owner.branch_depth
        or owner.loop_variables
    ):
        _error(
            owner.source,
            node,
            "program.storage",
            "hidden ownership changes require the unconditional program root",
        )


def declare_hidden(owner: _Body, node: ast.AnnAssign) -> None:
    require_root_storage(owner, node)
    if not isinstance(node.target, ast.Name):
        _error(
            owner.source,
            node,
            "program.storage",
            "declare one typed hidden register",
        )
    name = node.target.id
    if name in owner.ports or name in owner.classical:
        _error(
            owner.source,
            node,
            "program.storage",
            "hidden names must be unique and cannot shadow ports or parameters",
        )
    annotation = _annotation_node(owner.source, node.annotation)
    port = quantum_port(owner.definition, name, annotation, annotation, "inout")
    if node.value is not None:
        value = node.value
        constructor = {"qbit": QBit, "qnum": QNum, "qarray": QArray}[port.kind]
        if (
            not isinstance(value, ast.Call)
            or value.args
            or value.keywords
            or not isinstance(value.func, (ast.Name, ast.Attribute))
            or _resolve(owner.definition, value.func) is not constructor
        ):
            _error(
                owner.source,
                node,
                "program.storage",
                "initialize a local handle with its matching zero-argument quantum constructor only",
            )
    owner.ports[name] = port
    owner.widths[name] = port.width
    owner.hidden.add(name)
    if port.kind == "qarray":
        owner.array_names.add(name)


def drop_hidden(owner: _Body, node: ast.Call) -> None:
    require_root_storage(owner, node)
    if (
        len(node.args) != 1
        or not isinstance(node.args[0], ast.Name)
        or node.args[0].id not in owner.hidden
    ):
        _error(
            owner.source,
            node,
            "program.drop",
            "drop requires one whole owned hidden register",
        )
    name = node.args[0].id
    value = owner.register(node.args[0])
    owner.emit("qreg.drop", node, (value,))
    owner.dropped.add(name)


def hidden_declarations(owner: _Body) -> tuple[Operation, ...]:
    missing = owner.hidden - owner.dropped
    if missing:
        _error(
            owner.source,
            owner.source.node,
            "program.drop",
            "drop every hidden register before return: " + ", ".join(sorted(missing)),
        )
    return tuple(
        Operation(
            "qreg.hidden",
            results=(owner.registers[name],),
            attributes={"name": name},
            location=owner.source.location(owner.source.node),
        )
        for name in owner.ports
        if name in owner.hidden
    )
