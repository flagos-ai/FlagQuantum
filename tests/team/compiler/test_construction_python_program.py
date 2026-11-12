"""Typed Python programs lower to the existing checked register/effect profile."""

import hashlib
import linecache
from types import FunctionType

import pytest

from flagquantum.compiler._construction.frontend_program import capture_program
from flagquantum.compiler._construction.frontend_python import _QuantumFunction
from flagquantum.compiler._construction.register_profile import verify_register_program
from flagquantum.compiler._hybrid.verifier import HybridVerificationError

pytestmark = pytest.mark.unit
PREFIX = """from __future__ import annotations
from flagquantum.compiler._construction.frontend_python import qfunc
from flagquantum.compiler._construction.program_types import (
    Const, Output, QNum, QBit, allocate, hadamard_transform,
)
"""


def program_definition(
    source: str, entry: str = "main", *, future: bool = True
) -> _QuantumFunction:
    prefix = (
        PREFIX if future else PREFIX.replace("from __future__ import annotations\n", "")
    )
    text = prefix + source
    filename = f"<up01-{hashlib.sha256(text.encode()).hexdigest()}>"
    linecache.cache[filename] = (
        len(text),
        None,
        text.splitlines(keepends=True),
        filename,
    )
    namespace: dict[str, object] = {}
    exec(compile(text, filename, "exec"), namespace)
    definition = namespace[entry]
    assert isinstance(definition, _QuantumFunction)
    return definition


DIRECT = """@qfunc
def main(x: Output[QNum[2, False, 0]], y: Output[QNum[2]], result: Output[QNum]) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    result |= x*x+y
"""


@pytest.mark.parametrize("future", [False, True])
def test_arithmetic_reuses_scalar_capture_and_register_effects(future: bool) -> None:
    root, functions = capture_program(program_definition(DIRECT, future=future))
    assert verify_register_program(root, functions) is root
    outputs = [
        q.results[0] for q in root.body.blocks[0].operations if q.name == "qreg.declare"
    ]
    assert [q.type.parameters[0] for q in outputs] == [2, 2, 4]
    assert sum(q.name == "qreg.compute" for q in root.body.blocks[0].operations) == 1
    assert sum(q.name == "qreg.gate" for q in root.body.blocks[0].operations) == 4
    again = capture_program(program_definition(DIRECT, future=future))
    assert (root.semantic_identity, tuple(f.semantic_identity for f in functions)) == (
        again[0].semantic_identity,
        tuple(f.semantic_identity for f in again[1]),
    )


def test_host_body_is_never_called_even_if_code_is_replaced() -> None:
    definition = program_definition(DIRECT)
    function = definition.function
    assert isinstance(function, FunctionType)
    namespace: dict[str, object] = {}
    exec("def bomb():\n    raise AssertionError('host body executed')\n", namespace)
    bomb = namespace["bomb"]
    assert isinstance(bomb, FunctionType)
    function.__code__ = bomb.__code__
    root, functions = capture_program(definition)
    verify_register_program(root, functions)


@pytest.mark.parametrize(
    "body",
    [
        "hadamard_transform(x)",
        "allocate(x); allocate(x)",
        "allocate(x); x |= 1",
        "x |= x+1",
        "x |= 4",
        "allocate(3,x)",
        "allocate(True,x)",
        "allocate(0,x)",
        "allocate(257,x)",
        "x ^= 1",
        "allocate(x); if_unknown(x)",
        "allocate(x); phase(x,0.8)",
        "allocate(x); free(x)",
        "allocate(x); print(x)",
        "allocate(x); x |= x-1",
        "allocate(x); x |= x/2",
        "allocate(x); x |= 1.5*x",
        "allocate(x); x |= x**9",
        "allocate(x); x |= x**0",
        "allocate(x); return x",
    ],
)
def test_invalid_ownership_preparation_effects_and_domains_are_refused(
    body: str,
) -> None:
    source = f"@qfunc\ndef main(x: Output[QNum[2]]) -> None:\n    {body}\n"
    with pytest.raises(HybridVerificationError):
        capture_program(program_definition(source))


@pytest.mark.parametrize(
    "annotation",
    [
        "Input[QNum]",
        "QNum",
        "Output[list[int]]",
        "Output[QNum[0]]",
        "Output[QNum[257]]",
        "Output[QNum[2,1,0]]",
        "Output[QNum[2,False,3]]",
        "Output[QNum[True]]",
        "Output[QBit[2]]",
    ],
)
def test_numeric_and_consuming_role_boundaries(annotation: str) -> None:
    with pytest.raises(HybridVerificationError):
        capture_program(
            program_definition(f"@qfunc\ndef main(x: {annotation}):\n    allocate(x)\n")
        )


@pytest.mark.parametrize(
    "helper,call",
    [
        ("def helper(x: Const[QNum]):\n    hadamard_transform(x)", "helper(x)"),
        ("def helper(x: Output[QNum]):\n    allocate(2,x)", "helper(x)"),
        ("def helper(a: Const[QNum],b: Output[QNum]):\n    b |= a", "helper(x,x)"),
        (
            "def helper(a: Output[QNum],b: Output[QNum]):\n    a |= 1\n    b |= 1",
            "helper(y,y)",
        ),
        ("def helper(a: Const[QBit],b: Output[QNum]):\n    b |= a", "helper(x,y)"),
        ("def helper(a: Const[QNum[3]],b: Output[QNum]):\n    b |= a", "helper(x,y)"),
    ],
)
def test_calls_delegate_const_fresh_alias_kind_and_width_guards(
    helper: str, call: str
) -> None:
    source = f"@qfunc\n{helper}\n@qfunc\ndef main(x: Output[QNum[2]],y: Output[QNum]):\n    allocate(x)\n    {call}\n"
    with pytest.raises(HybridVerificationError):
        capture_program(program_definition(source))


def test_recursion_is_refused_before_expansion() -> None:
    source = """@qfunc
def helper(x: Output[QNum]):
    helper(x)
@qfunc
def main(x: Output[QNum]):
    helper(x)
"""
    with pytest.raises(HybridVerificationError, match="program.recursion"):
        capture_program(program_definition(source))


def test_symbol_spelling_cannot_impersonate_a_captured_intrinsic() -> None:
    source = """calls=[]
def allocate(*args):
    calls.append(args)
@qfunc
def main(x: Output[QNum[2]]):
    allocate(x)
"""
    definition = program_definition(source)
    with pytest.raises(HybridVerificationError, match="program.call"):
        capture_program(definition)
    assert isinstance(definition.function, FunctionType)
    assert definition.function.__globals__["calls"] == []


def test_dynamic_attribute_property_is_not_read_as_a_callable() -> None:
    source = """calls=[]
class Probe:
    @property
    def prepare(self):
        calls.append('property called')
        return allocate
probe=Probe()
@qfunc
def main(x: Output[QNum[2]]):
    probe.prepare(x)
"""
    definition = program_definition(source)
    with pytest.raises(HybridVerificationError, match="construction.python.name"):
        capture_program(definition)
    assert isinstance(definition.function, FunctionType)
    assert definition.function.__globals__["calls"] == []


def test_captured_call_specialization_and_effect_budgets_are_bounded() -> None:
    definitions = []
    for i in range(33):
        action = f"f{i+1}(x)" if i < 32 else "allocate(1,x)"
        definitions.append(f"@qfunc\ndef f{i}(x: Output[QNum]):\n    {action}\n")
    definitions.append("@qfunc\ndef main(x: Output[QNum]):\n    f0(x)\n")
    with pytest.raises(HybridVerificationError, match="32"):
        capture_program(program_definition("\n".join(definitions)))
    body = "    hadamard_transform(x)\n" * 16
    with pytest.raises(HybridVerificationError, match="4096"):
        capture_program(
            program_definition(
                "@qfunc\ndef main(x: Output[QNum[256]]):\n    allocate(x)\n" + body
            )
        )
