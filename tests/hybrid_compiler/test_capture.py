from __future__ import annotations

from collections.abc import Iterator

import pytest

from flagquantum.compiler._hybrid import (
    HybridCaptureError,
    HybridProgram,
    Operation,
    Region,
    capture_function,
    capture_source,
    tensor_type,
    verify_program,
)

pytestmark = pytest.mark.unit

WEIGHTS = tensor_type("float64", (None, 4))
DATA = tensor_type("float64", (4,))

CATALYST_STYLE_SOURCE = """
def cost(weights, data):
    qp.AngleEmbedding(data, wires=range(4))

    for x in weights:
        for j, p in enumerate(x):
            if p > 0:
                qp.RX(p, wires=j)
            elif p < 0:
                qp.RY(p, wires=j)

        for j in range(4):
            qp.CNOT(wires=[j, jnp.mod((j + 1), 4)])

    return qp.expval(qp.PauliZ(0) + qp.PauliZ(3))
"""


def region_operations(region: Region) -> Iterator[Operation]:
    for block in region.blocks:
        for operation in block.operations:
            yield operation
            for nested in operation.regions:
                yield from region_operations(nested)


def operations(program: HybridProgram) -> Iterator[Operation]:
    yield from region_operations(program.body)


def test_catalyst_style_control_flow_captures_without_catalyst_dependency() -> None:
    program = capture_source(
        CATALYST_STYLE_SOURCE,
        (WEIGHTS, DATA),
        filename="cost_program.py",
    )
    names = [operation.name for operation in operations(program)]

    assert verify_program(program) is program
    assert names.count("scf.for") == 3
    assert names.count("scf.if") == 2
    assert names.count("tensor.extract") == 1
    assert names.count("quantum.rx") == 1
    assert names.count("quantum.ry") == 1
    assert names.count("quantum.cx") == 1
    assert names[-2:] == ["quantum.expectation", "program.return"]


def test_capture_function_reads_source_without_invoking_function() -> None:
    def valid_but_not_executable(data):
        qp.AngleEmbedding(data, wires=range(4))  # noqa: F821
        return qp.expval(qp.PauliZ(0))  # noqa: F821

    program = capture_function(valid_but_not_executable, (DATA,))

    assert program.name == "valid_but_not_executable"
    assert [operation.name for operation in program.body.blocks[0].operations] == [
        "quantum.angle_embedding",
        "quantum.expectation",
        "program.return",
    ]


def test_function_docstring_is_not_a_runtime_effect() -> None:
    source = '''
def documented(data):
    """A cost function whose docstring is capture metadata only."""
    qp.AngleEmbedding(data, wires=range(4))
    return qp.expval(qp.PauliZ(0))
'''

    assert capture_source(source, (DATA,)).name == "documented"


def test_formatting_does_not_change_captured_semantic_identity() -> None:
    compact = """
def cost(data):
 qp.AngleEmbedding(data,wires=range(4))
 return qp.expval(qp.PauliZ(0))
"""
    spaced = """
def cost(data):
    qp.AngleEmbedding(
        data,
        wires=range(4),
    )
    return qp.expval(
        qp.PauliZ(0)
    )
"""

    assert (
        capture_source(compact, (DATA,)).semantic_identity
        == capture_source(spaced, (DATA,)).semantic_identity
    )


def test_local_renaming_does_not_change_captured_semantic_identity() -> None:
    first = """
def cost(weights, data):
    qp.AngleEmbedding(data, wires=range(4))
    for row in weights:
        for index, parameter in enumerate(row):
            if parameter > 0:
                qp.RX(parameter, wires=index)
    return qp.expval(qp.PauliZ(0))
"""
    renamed = """
def cost(a, b):
    qp.AngleEmbedding(b, wires=range(4))
    for x in a:
        for j, p in enumerate(x):
            if p > 0:
                qp.RX(p, wires=j)
    return qp.expval(qp.PauliZ(0))
"""

    assert (
        capture_source(first, (WEIGHTS, DATA)).semantic_identity
        == capture_source(renamed, (WEIGHTS, DATA)).semantic_identity
    )


@pytest.mark.parametrize(
    ("body", "code"),
    (
        ("while True:\n        pass", "control.while"),
        ("for j in range(4):\n        break", "control.loop_exit"),
        ("data[0] = 1.0", "effect.mutation"),
        ("value = data.item()", "tensor.host_conversion"),
        ("value = external(data)", "call.unsupported"),
        ("value = [x for x in data]", "expression.comprehension"),
        ("import pathlib", "effect.import"),
        ("with manager():\n        pass", "effect.context"),
        ("raise RuntimeError()", "effect.exception"),
        ("for j in range(0, 4, 0):\n        pass", "control.range_step"),
    ),
)
def test_unsupported_python_constructs_fail_closed(body: str, code: str) -> None:
    indented = "\n".join(f"    {line}" for line in body.splitlines())
    source = f"def invalid(data):\n{indented}\n"

    with pytest.raises(HybridCaptureError) as caught:
        capture_source(source, (DATA,), filename="invalid.py")

    assert caught.value.diagnostic.code == code
    assert caught.value.diagnostic.location.filename == "invalid.py"
    assert caught.value.diagnostic.location.line >= 2


def test_runtime_predicate_is_captured_structurally_not_evaluated() -> None:
    source = """
def branch(weights, data):
    qp.AngleEmbedding(data, wires=range(4))
    p = weights[0, 0]
    if p > 0:
        qp.RX(p, wires=0)
    else:
        qp.RY(p, wires=0)
    return qp.expval(qp.PauliZ(0))
"""

    program = capture_source(source, (WEIGHTS, DATA))

    assert "scf.if" in [operation.name for operation in operations(program)]


def test_outer_scalar_write_inside_branch_is_explicitly_carried() -> None:
    source = """
def branch(weights, data):
    qp.AngleEmbedding(data, wires=range(4))
    p = weights[0, 0]
    if p > 0:
        p = p + 1
        qp.RX(p, wires=0)
    return qp.expval(qp.PauliZ(0))
"""

    program = capture_source(source, (WEIGHTS, DATA))
    branch = next(
        operation for operation in operations(program) if operation.name == "scf.if"
    )

    assert len(branch.operands) == 3  # predicate, scalar, effect
    assert len(branch.results) == 2
    assert all(len(region.blocks[0].arguments) == 2 for region in branch.regions)
    assert all(
        len(region.blocks[0].operations[-1].operands) == 2 for region in branch.regions
    )
    else_block = branch.regions[1].blocks[0]
    assert else_block.operations[-1].operands[0] == else_block.arguments[0]
