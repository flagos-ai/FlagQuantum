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


@pytest.mark.parametrize(
    "package_name, blob, expected",
    (
        (
            "qualtran.bloqs.arithmetic",
            "59ca8a5af83c071f824d5c3a7a49d2bb41e97e5c",
            EXPORTS,
        ),
        (
            "qualtran.bloqs.arithmetic.lists",
            "7ed84457f5734a6e2d76fd6d883032fc6f815afa",
            ("HasDuplicates", "SortInPlace", "SymmetricDifference"),
        ),
        (
            "qualtran.bloqs.arithmetic.conversions",
            "129a9c14487f032c3ce60c7fd4f56cd8f5bf75cd",
            (
                "ToContiguousIndex",
                "SignedIntegerToTwosComplement",
                "SignExtend",
                "SignTruncate",
            ),
        ),
    ),
)
def test_complete_source_export_order_and_definition_identity(
    package_name: str, blob: str, expected: tuple[str, ...]
) -> None:
    package = import_module(package_name)
    assert package.__file__ is not None
    raw = Path(package.__file__).read_bytes()
    assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob
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
    assert tuple(observed) == expected
