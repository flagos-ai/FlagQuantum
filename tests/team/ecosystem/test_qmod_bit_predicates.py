"""Unary logic respects existing tree depth and owned expression records."""

import pytest

from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.ecosystem._qmod.adapter import SUPPORTED_SOURCE_VERSION, parse_qmod
from flagquantum.ecosystem._qmod.syntax import MAX_EXPRESSION_DEPTH

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "expression",
    [
        "~" * MAX_EXPRESSION_DEPTH + "x",
        "~(" + "+".join(["x"] * MAX_EXPRESSION_DEPTH) + ")",
    ],
)
def test_unary_tree_and_recursive_depth_limits(expression: str) -> None:
    with pytest.raises(HybridVerificationError, match="qmod.budget.depth"):
        parse_qmod(
            f"qfunc main(output x:qbit,output result:qnum) {{allocate(x);result={expression};}}",
            source_version=SUPPORTED_SOURCE_VERSION,
        )


def test_maximum_admitted_unary_chain_uses_existing_operations() -> None:
    expression = "~" * (MAX_EXPRESSION_DEPTH - 1) + "x"
    imported = parse_qmod(
        f"qfunc main(output x:qbit,output result:qnum) {{allocate(x);result={expression};}}",
        source_version=SUPPORTED_SOURCE_VERSION,
    )
    assert (
        sum(
            op.name == "qint.equal"
            for op in imported.functions[0].body.blocks[0].operations
        )
        == MAX_EXPRESSION_DEPTH - 1
    )
