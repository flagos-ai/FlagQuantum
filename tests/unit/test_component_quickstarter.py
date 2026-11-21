"""Execute the complete original quickstarter and native generated components."""

import ast
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import torch

from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.call_graph import SymbolAllocator
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.simulation.unitary import get_unitary
from tests.unit.test_development_tools import REFERENCE

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]
SOURCE = REFERENCE / "dev_tools/bloq-quickstarter.html"
NATIVE = ROOT / "tools/development/component_quickstarter.html"
HARNESS = ROOT / "tests/unit/fixtures/component_quickstarter_dom.cjs"


def render(path: Path, **configuration: Any) -> dict[str, Any]:
    result = subprocess.run(
        ["node", str(HARNESS)],
        input=json.dumps({"path": str(path), **configuration}),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    result_object: dict[str, Any] = json.loads(result.stdout)
    return result_object


def load(code: str, **values: Any) -> dict[str, Any]:
    namespace = {"__name__": "__main__", **values}
    exec(compile(code, "<generated component>", "exec"), namespace)
    return namespace


def source_signature(code: str, **values: Any) -> list[tuple[str, Any, str]]:
    signature = next(
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.FunctionDef) and node.name == "signature"
    )
    result = []
    for call in ast.walk(signature):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "Register"
        ):
            keywords = {keyword.arg: keyword.value for keyword in call.keywords}
            width = eval(
                compile(ast.Expression(keywords["bitsize"]), "<source width>", "eval"),
                {},
                values,
            )
            side = keywords.get("side")
            result.append(
                (
                    ast.literal_eval(call.args[0]),
                    width,
                    side.attr if isinstance(side, ast.Attribute) else "THRU",
                )
            )
    return result


def registers() -> list[dict[str, str]]:
    return [
        {"reg_name": "x", "bitsize": "2", "side": "Thru", "desc": "Input word."},
        {"reg_name": "y", "bitsize": "n", "side": "Left", "desc": "Consumed word."},
        {"reg_name": "z", "bitsize": "3", "side": "Right", "desc": "Created word."},
    ]


def test_complete_source_identity_and_scripts() -> None:
    data = SOURCE.read_bytes()
    assert hashlib.sha1(
        b"blob " + str(len(data)).encode() + b"\0" + data
    ).hexdigest() == ("fed8cda3ee0c55e415342c1750a6ff843acb7552")
    assert re.findall(r"function (\w+)\(", SOURCE.read_text()) == [
        "addRegisterBoxes",
        "getRegisterInputs",
        "generateCode",
    ]
    assert re.findall(r"function (\w+)\(", NATIVE.read_text()) == [
        "addRegisterBoxes",
        "getRegisterInputs",
        "generateCode",
    ]
    for attribute in ("integrity", "crossorigin", "referrerpolicy"):
        assert re.findall(attribute + r'="([^"]+)"', SOURCE.read_text()) == re.findall(
            attribute + r'="([^"]+)"', NATIVE.read_text()
        )


def test_original_and_native_full_signature_and_stubs() -> None:
    configuration = dict(
        className="WordOperation", description="Word operation.", registers=registers()
    )
    source = render(SOURCE, **configuration)
    native = render(NATIVE, **configuration)
    assert source["registers"] == native["registers"] == registers()
    source_type = load(source["code"], n=4)["WordOperation"]
    native_type = load(native["code"], n=4)["WordOperation"]
    with pytest.raises(TypeError, match="bitsize"):
        _ = source_type().signature
    assert source_signature(source["code"], n=4) == [
        (actual.name, actual.bitsize, actual.side.name)
        for actual in native_type().signature
    ]
    for method, arguments in (
        ("build_composite_bloq", (None, None, None, None)),
        ("bloq_counts", ()),
    ):
        with pytest.raises(NotImplementedError):
            getattr(source_type(), method)(*arguments)
    with pytest.raises(NotImplementedError, match="register body"):
        native_type().program()
    with pytest.raises(NotImplementedError, match="resource declaration"):
        native_type().declared_calls(SymbolAllocator())
    assert source_type().short_name() == native_type().short_name() == "WordOperation"
    original_ast = ast.parse(source["code"])
    actual_ast = ast.parse(native["code"])
    original_body = next(
        node
        for node in ast.walk(original_ast)
        if isinstance(node, ast.FunctionDef) and node.name == "build_composite_bloq"
    )
    actual_body = next(
        node
        for node in ast.walk(actual_ast)
        if isinstance(node, ast.FunctionDef) and node.name == "build_registers"
    )
    assert [arg.arg for arg in original_body.args.args] == ["self", "bb", "x", "y", "z"]
    assert [arg.arg for arg in actual_body.args.kwonlyargs] == ["x", "y"]


def test_symbolic_widths_empty_signature_and_description_escaping() -> None:
    from sympy import Symbol

    n = Symbol("n", positive=True, integer=True)
    source = render(SOURCE, className="WordOperation", registers=registers())["code"]
    assert source_signature(source, n=n)[1][1] == n
    code = render(NATIVE, className="WordOperation", registers=registers())["code"]
    assert tuple(load(code, n=n)["WordOperation"]().signature)[1].bitsize == n
    for path in (SOURCE, NATIVE):
        code = render(path, className="EmptyOperation", registers=[])["code"]
        assert len(load(code)["EmptyOperation"]().signature) == 0
    description = 'A """quoted""" description with \\ paths and <markup>.'
    code = render(NATIVE, description=description)["code"]
    assert load(code)["RegisterOperation"].__doc__.startswith(description)


@pytest.mark.parametrize("path", [SOURCE, NATIVE])
def test_input_side_and_description_events(path: Path) -> None:
    class_id = "bloq_name" if path == SOURCE else "component_name"
    result = render(
        path,
        actions=[
            {"id": class_id, "value": "RenamedWord", "event": "keyup"},
            {"register": 0, "name": "bitsize", "value": "4", "event": "change"},
            {"register": 0, "name": "side", "value": "Left", "event": "change"},
            {"register": 0, "name": "desc", "value": "Changed word.", "event": "keyup"},
        ],
    )
    cls = load(result["code"])["RenamedWord"]
    if path == SOURCE:
        with pytest.raises(TypeError, match="bitsize"):
            _ = cls().signature
        assert source_signature(result["code"])[0][1:] == (4, "LEFT")
    else:
        register = tuple(cls().signature)[0]
        assert register.bitsize == 4 and register.side.name == "LEFT"
    assert "Changed word." in cls.__doc__
    assert all(set(events) == {"keyup", "change"} for events in result["events"])


@pytest.mark.parametrize("name", ["", "not-a-class", "class"])
def test_invalid_identifier_is_reported_instead_of_generating_invalid_python(
    name: str,
) -> None:
    original = render(SOURCE, className=name)["code"]
    with pytest.raises(SyntaxError):
        compile(original, "<original invalid identifier>", "exec")
    actual = render(NATIVE, className=name)
    assert actual["code"] == "" and "Python identifiers" in actual["error"]


def test_native_duplicate_names_and_empty_width_are_reported() -> None:
    duplicate = [dict(registers()[0]), dict(registers()[0])]
    for values in (duplicate, [{**registers()[0], "bitsize": ""}]):
        result = render(NATIVE, registers=values)
        assert result["code"] == "" and result["error"]
    result = render(NATIVE, highlight=False)
    assert not result["error"] and "class RegisterOperation" in result["code"]


def test_generated_body_append_amplitudes_workspaces_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation_type = load(render(NATIVE, registers=[registers()[0]])["code"])[
        "RegisterOperation"
    ]

    def body(
        self: Any, builder: ComponentProgramBuilder, *, x: RegisterBinding
    ) -> dict[str, RegisterBinding]:
        bit = quantum_bit_type()
        gate = bit_gate_function(
            "x", (Register("q", bit),), CircuitIR(1, (Instruction("x", (0,)),))
        )
        builder.define(gate)
        bits = builder.split(x)
        outputs = [builder.call_dict("x", {"q": value})["q"] for value in bits]
        return {"x": builder.join(outputs, x.register.dtype)}

    def declared(self: Any, symbols: SymbolAllocator) -> dict[ComponentCost, int]:
        return {ComponentCost("X", 1, GateCost(clifford=1)): 2}

    monkeypatch.setattr(operation_type, "build_registers", body)
    monkeypatch.setattr(operation_type, "declared_calls", declared)
    operation = operation_type(inputs=((0, 1),))
    bound = operation.bind()
    assert bound.outputs == ((0, 1),) and bound.events == ()
    assert bound.circuit.instructions == (
        Instruction("x", (0,)),
        Instruction("x", (1,)),
    )
    state = torch.complex(torch.arange(1, 9).double(), torch.arange(8).double())
    state /= state.norm()
    circuit = Circuit(3, inputs=state, dtype=torch.complex128).append(operation)
    expected = state.reshape(2, 2, 2).flip((0, 1)).flatten()
    torch.testing.assert_close(circuit.state().flatten(), expected, atol=2e-14, rtol=0)
    circuit.append(operation)
    torch.testing.assert_close(circuit.state().flatten(), state, atol=2e-14, rtol=0)
    assert Counter(gate.name for gate in bound.circuit.instructions) == {"x": 2}
    calls = operation.declared_calls(SymbolAllocator())
    assert sum(node.gates.clifford * count for node, count in calls.items()) == 2
    expected_matrix = torch.tensor(
        [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.complex128
    )
    torch.testing.assert_close(
        get_unitary(bound.circuit, dtype=torch.complex128), expected_matrix
    )


def test_generated_sided_body_preserves_explicit_lifetime_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = [
        registers()[0],
        {**registers()[1], "bitsize": "2"},
        {**registers()[2], "bitsize": "2"},
    ]
    operation_type = load(render(NATIVE, registers=configuration)["code"])[
        "RegisterOperation"
    ]

    def body(
        self: Any,
        builder: ComponentProgramBuilder,
        *,
        x: RegisterBinding,
        y: RegisterBinding,
    ) -> dict[str, RegisterBinding]:
        builder.free(y)
        return {"x": x, "z": builder.allocate(2)}

    monkeypatch.setattr(operation_type, "build_registers", body)
    operation = operation_type(
        inputs=((0, 1), (2, 3)), clean=(4, 5), input_ownership=((2, False), (3, False))
    )
    bound = operation.bind()
    assert bound.outputs[0] == (0, 1)
    assert len(bound.outputs[1]) == 2
    assert [event.kind for event in bound.events] == ["free", "allocate"]
    assert bound.circuit.instructions == ()
    state = torch.zeros(64, dtype=torch.complex128)
    state.reshape(2, 2, 2, 2, 2, 2)[:, :, 0, 0, 0, 0] = torch.tensor(
        [[1, 2j], [3j, 4]], dtype=torch.complex128
    )
    state /= state.norm()
    actual = (
        Circuit(6, inputs=state, dtype=torch.complex128)
        .append(operation)
        .state()
        .flatten()
    )
    torch.testing.assert_close(actual, state, atol=0, rtol=0)
    with pytest.raises(ValueError):
        operation_type(inputs=((0, 1), (2, 3)), clean=(4, 5)).bind()


def test_disjoint_sides_share_a_name_and_import_names_are_protected() -> None:
    result = render(
        NATIVE,
        registers=[
            {"reg_name": "x", "bitsize": "1", "side": "Left", "desc": "Input."},
            {"reg_name": "x", "bitsize": "1", "side": "Right", "desc": "Output."},
        ],
    )
    assert not result["error"]
    cls = load(result["code"])["RegisterOperation"]
    assert [register.name for register in cls().signature] == ["x", "x"]
    assert [register.side.name for register in cls().signature] == ["LEFT", "RIGHT"]
    assert render(NATIVE, className="Register")["error"]


def test_add_register_and_regenerate_button_events() -> None:
    result = render(
        NATIVE,
        actions=[
            {"id": "add_register", "event": "click"},
            {"register": 1, "name": "reg_name", "value": "y", "event": "keyup"},
            {"register": 1, "name": "bitsize", "value": "3", "event": "change"},
            {"id": "regenerate", "event": "click"},
        ],
    )
    cls = load(result["code"])["RegisterOperation"]
    assert [(register.name, register.bitsize) for register in cls().signature] == [
        ("x", 1),
        ("y", 3),
    ]
