"""Check the frozen arithmetic initializer's complete re-export identity."""

import ast
from hashlib import sha1
from importlib import import_module
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
EXPORTS = (
    "Add",
    "AddK",
    "OutOfPlaceAdder",
    "BitwiseNot",
    "Xor",
    "XorK",
    "BiQubitsMixer",
    "CLinearDepthGreaterThan",
    "Equals",
    "EqualsAConstant",
    "GreaterThan",
    "GreaterThanConstant",
    "LessThanConstant",
    "LessThanEqual",
    "LinearDepthHalfGreaterThan",
    "LinearDepthHalfGreaterThanEqual",
    "LinearDepthHalfLessThan",
    "LinearDepthHalfLessThanEqual",
    "SingleQubitCompare",
    "CAdd",
    "SignedIntegerToTwosComplement",
    "SignExtend",
    "ToContiguousIndex",
    "HammingWeightCompute",
    "InvertRealNumber",
    "MultiplyTwoReals",
    "PlusEqualProduct",
    "Product",
    "ScaleIntByReal",
    "Square",
    "SquareRealNumber",
    "SumOfSquares",
    "Negate",
    "BitonicSort",
    "Comparator",
    "Subtract",
    "SubtractFrom",
    "ArcSin",
    "CHalf",
    "Lt",
    "MultiCToffoli",
)


def test_complete_source_export_order_and_definition_identity() -> None:
    package = import_module("qualtran.bloqs.arithmetic")
    assert package.__file__ is not None
    raw = Path(package.__file__).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "59ca8a5af83c071f824d5c3a7a49d2bb41e97e5c"
    )
    imports = ast.parse(raw).body
    assert all(isinstance(node, ast.ImportFrom) for node in imports)
    observed: list[str] = []
    for node in imports:
        assert isinstance(node, ast.ImportFrom) and node.module is not None
        module = import_module("." * node.level + node.module, package.__name__)
        for alias in node.names:
            assert alias.asname is None
            observed.append(alias.name)
            assert getattr(package, alias.name) is getattr(module, alias.name)
    assert tuple(observed) == EXPORTS
