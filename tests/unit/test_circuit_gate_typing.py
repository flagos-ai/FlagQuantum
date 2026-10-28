"""Static and runtime conformance for the shared fluent gate surface."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from flagquantum.circuit import Circuit
from flagquantum.core.operator_schema import OPERATOR_ALIASES, OPERATOR_SCHEMAS
from flagquantum.dynamic import DynamicCircuit

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
GATE_TYPING_STUB = ROOT / "flagquantum" / "core" / "_gate_typing.pyi"
PY_TYPED_MARKER = ROOT / "flagquantum" / "py.typed"


def _typed_workflows() -> tuple[Circuit, DynamicCircuit]:
    static: Circuit = Circuit(3).h(0).cx(0, 1).rx(2, 0.25)
    dynamic: DynamicCircuit = DynamicCircuit(3).h(0).cx(0, 1).rx(2, 0.25).measure(2)
    return static, dynamic


def test_static_and_dynamic_gate_chains_preserve_builder_types() -> None:
    static, dynamic = _typed_workflows()

    assert isinstance(static, Circuit)
    assert type(static) is Circuit
    assert isinstance(dynamic, DynamicCircuit)
    assert [item.name for item in static.to_ir().instructions] == ["h", "cx", "rx"]
    assert [item.name for item in dynamic.to_ir().instructions] == [
        "h",
        "cx",
        "rx",
        "measure",
    ]


def test_gate_typing_stub_covers_the_complete_schema_surface() -> None:
    tree = ast.parse(GATE_TYPING_STUB.read_text(encoding="utf-8"))
    (typing_class,) = (
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_BuiltinGateMethods"
    )
    declared = {
        node.name for node in typing_class.body if isinstance(node, ast.FunctionDef)
    }
    lower = set(OPERATOR_SCHEMAS) | set(OPERATOR_ALIASES)

    assert declared == lower | {name.upper() for name in lower}
    assert all(
        isinstance(node.returns, ast.Name) and node.returns.id == "_GateBuilderT"
        for node in typing_class.body
        if isinstance(node, ast.FunctionDef)
    )


def test_gate_typing_does_not_change_runtime_method_ownership() -> None:
    assert Circuit.__bases__ == (object,)
    assert "h" not in DynamicCircuit.__dict__
    assert "cx" not in DynamicCircuit.__dict__
    assert "rx" not in DynamicCircuit.__dict__
    assert Circuit.h.__module__ == "flagquantum.core._gate_methods"
    assert DynamicCircuit.h is Circuit.h


def test_gate_typing_files_are_declared_as_package_data() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert PY_TYPED_MARKER.is_file()
    assert '"py.typed"' in project
    assert '"core/*.pyi"' in project
