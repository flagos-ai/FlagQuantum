"""Python source contracts: composition, width inference and early refusal."""

import inspect
import linecache
from collections.abc import Callable

import pytest

from flagquantum.compiler._construction.frontend_python import capture, qfunc
from flagquantum.compiler._hybrid.model import HybridProgram
from flagquantum.compiler._hybrid.verifier import HybridVerificationError

pytestmark = pytest.mark.unit


def polynomial(x: int, y: int) -> int:
    square = x**2
    alias = square
    return alias + y


def square(x: int) -> int:
    return x * x


def composed(x: int, y: int) -> int:
    return square(x) + y


def configured(x: int, scale: int) -> int:
    doubled = 1 + scale
    return x * doubled


@qfunc
def declared(x: int, y: int) -> int:
    return x * x + y


def _width(program: HybridProgram) -> int:
    result = program.body.blocks[0].operations[-1].operands[0]
    return int(result.type.parameters[0])


def test_inference_alias_identity_parameter_order_and_source_locations() -> None:
    program = capture(polynomial, {"y": 2, "x": 2})
    block = program.body.blocks[0]
    assert tuple(arg.id.scope for arg in block.arguments) == ("x", "y")
    multiply = next(op for op in block.operations if op.name == "qint.multiply")
    assert multiply.operands[0] is multiply.operands[1]
    assert _width(program) == 4
    assert all(
        op.location and op.location.filename.endswith(__file__)
        for op in block.operations
    )


def test_explicit_helpers_and_configuration_share_existing_ssa() -> None:
    graph = capture(composed, {"x": 2, "y": 2}, functions=(square,))
    assert _width(graph) == 4
    assert (
        sum(op.name == "qint.multiply" for op in graph.body.blocks[0].operations) == 1
    )
    configured_graph = capture(configured, {"x": 2}, constants={"scale": 2})
    assert _width(configured_graph) == 4  # 3*(1+2) = 9.
    assert any(
        op.name == "arith.constant" and op.attributes["value"] == 3
        for op in configured_graph.body.blocks[0].operations
    )


def test_qfunc_is_an_early_source_snapshot_and_not_a_host_callable() -> None:
    assert _width(capture(declared, {"x": 2, "y": 2})) == 4
    with pytest.raises(TypeError, match="captured quantum definition"):
        declared(3, 2)


def test_qfunc_retains_source_when_late_inspection_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*args: object) -> None:
        raise OSError("source disappeared")

    monkeypatch.setattr(inspect, "getsourcelines", unavailable)
    assert _width(capture(declared, {"x": 2, "y": 2})) == 4
    with pytest.raises(ValueError, match="source is unavailable"):
        capture(polynomial, {"x": 2, "y": 2})


def _cell(source: str, filename: str = "<cs04-cached-cell>") -> Callable[..., object]:
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace: dict[str, object] = {}
    exec(compile(source, filename, "exec"), namespace)
    function = namespace["main"]
    assert callable(function)
    return function


@pytest.mark.parametrize(
    "body,reason",
    [
        ("return x-y", "operator"),
        ("return x//y", "operator"),
        ("return x%y", "operator"),
        ("return x**3", "power"),
        ("return x**True", "power"),
        ("return x[0]", "syntax"),
        ("return x.real", "syntax"),
        ("return -x", "syntax"),
        ("return abs(x)", "call"),
        ("return unknown+x", "name"),
        ("return True+x", "constant"),
        ("return 1.5*x", "constant"),
        ("return 1", "return"),
        ("return", "return"),
        ("x = x+y\nreturn x", "assignment"),
        ("a = b = x\nreturn a", "assignment"),
        ("a,b = x,y\nreturn a", "assignment"),
        ("x += y\nreturn x", "statement"),
        ("if x:\n    return y\nreturn x", "statement"),
        ("for i in range(2):\n    x = x+y\nreturn x", "assignment"),
        ("return [x,y]", "syntax"),
    ],
)
def test_rejected_syntax_has_source_and_actionable_reason(
    body: str, reason: str
) -> None:
    function = _cell(
        "def main(x, y):\n"
        + "".join("    " + line + "\n" for line in body.splitlines())
    )
    with pytest.raises(
        HybridVerificationError, match=rf"python.{reason}: <cs04-cached-cell>:"
    ):
        capture(function, {"x": 2, "y": 2})


def test_cached_cell_source_and_missing_source_are_distinct() -> None:
    fn = _cell("def main(x, y):\n    return x*x+y\n")
    assert _width(capture(fn, {"x": 2, "y": 2})) == 4
    namespace: dict[str, object] = {}
    exec(
        compile("def missing(x):\n    return x*x\n", "<cs04-no-source>", "exec"),
        namespace,
    )
    missing = namespace["missing"]
    assert callable(missing)
    with pytest.raises(ValueError, match="source is unavailable"):
        capture(missing, {"x": 2})


def test_capture_does_not_execute_side_effects() -> None:
    touched = []

    def unsafe(x: int) -> int:
        touched.append(x)
        return x

    with pytest.raises(HybridVerificationError, match="side effects"):
        capture(unsafe, {"x": 2})
    assert not touched


@pytest.mark.parametrize(
    "source",
    [
        "def main(x=2):\n    return x*x\n",
        "def main(*args):\n    return args\n",
        "def main(x, *, y):\n    return x*y\n",
    ],
)
def test_unsupported_signatures_fail_before_body_capture(source: str) -> None:
    with pytest.raises(HybridVerificationError, match="signature"):
        capture(_cell(source), {"x": 2})


def test_recursive_helper_and_unknown_definitions_are_rejected() -> None:
    def recursive(x: int) -> int:
        return recursive(x)

    with pytest.raises(HybridVerificationError, match="recursion"):
        capture(recursive, {"x": 2})
    with pytest.raises(HybridVerificationError, match="register it explicitly"):
        capture(composed, {"x": 2, "y": 2})
    with pytest.raises(HybridVerificationError, match="duplicate definition"):
        capture(polynomial, {"x": 2, "y": 2}, functions=(polynomial,))


@pytest.mark.parametrize(
    "inputs,constants",
    [
        ({"x": 2}, {}),
        ({"x": 2, "y": 2, "z": 2}, {}),
        ({"x": 2, "y": 2}, {"x": 1}),
        ({"x": 2}, {"y": True}),
    ],
)
def test_parameter_coverage_and_constants(
    inputs: dict[str, int], constants: dict[str, int]
) -> None:
    with pytest.raises(HybridVerificationError):
        capture(polynomial, inputs, constants=constants)


@pytest.mark.parametrize("width", [True, 0, -1, 2.5, 257])
def test_input_widths_are_explicit_positive_bounded_integers(width: int) -> None:
    with pytest.raises((TypeError, ValueError)):
        capture(square, {"x": width})


def test_helpers_can_specialize_classical_arguments_without_executing_python() -> None:
    def times(x: int, factor: int) -> int:
        return x * factor

    def main(x: int) -> int:
        return times(x, 3) + times(x, 2)

    graph = capture(main, {"x": 2}, functions=(times,))
    assert _width(graph) == 4


def test_unknown_helper_keyword_arguments_and_missing_arguments() -> None:
    def main(x: int) -> int:
        return square(x=x)

    def absent(x: int) -> int:
        return square()

    with pytest.raises(HybridVerificationError, match="positional arguments"):
        capture(main, {"x": 2}, functions=(square,))
    with pytest.raises(HybridVerificationError, match="needs 1 positional"):
        capture(absent, {"x": 2}, functions=(square,))


def test_nested_expression_and_large_constant_budgets_are_checked() -> None:
    function = _cell("def main(x, y):\n    return " + "x+" * 70 + "y\n")
    with pytest.raises(HybridVerificationError, match="nesting exceeds"):
        capture(function, {"x": 2, "y": 2})
    huge = _cell("def main(x):\n    return x+" + str(1 << 256) + "\n")
    with pytest.raises(HybridVerificationError, match="256-bit"):
        capture(huge, {"x": 2})


def test_quantum_role_annotations_are_not_ignored() -> None:
    function = _cell("def main(x: 'Output[QNum]'):\n    return x\n")
    with pytest.raises(HybridVerificationError, match="annotation"):
        capture(function, {"x": 2})


def test_expanded_helpers_have_a_hard_operation_budget() -> None:
    filename = "<cs04-expanded-calls>"
    text = "def f0(x):\n    return x\n" + "".join(
        f"def f{i}(x):\n    return f{i-1}(x)+f{i-1}(x)\n" for i in range(1, 14)
    )
    linecache.cache[filename] = (
        len(text),
        None,
        text.splitlines(keepends=True),
        filename,
    )
    namespace: dict[str, object] = {}
    exec(compile(text, filename, "exec"), namespace)
    entry = namespace["f13"]
    helpers = tuple(namespace[f"f{i}"] for i in range(13))
    assert callable(entry) and all(callable(helper) for helper in helpers)
    with pytest.raises(HybridVerificationError, match="expanded source exceeds"):
        capture(entry, {"x": 1}, functions=helpers)


def test_source_size_and_node_budgets_precede_lowering() -> None:
    function = _cell('def main(x):\n    "' + "a" * 262144 + '"\n    return x\n')
    with pytest.raises(ValueError, match="source exceeds"):
        capture(function, {"x": 1})
    many = _cell(
        "def main(x):\n"
        + "".join(f"    a{i} = x\n" for i in range(1100))
        + "    return x\n"
    )
    with pytest.raises(HybridVerificationError, match="4096 AST nodes"):
        capture(many, {"x": 1})
