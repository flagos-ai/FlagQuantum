"""Audit the frozen ECC package exports against their owned primitive modules."""

import ast
from hashlib import sha1
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

# Source package aliases are disposed into existing domain owners. Algorithm
# decompositions and their open boundaries retain their separate acceptance IDs.
EXPORTS = (
    ("ECPoint", "ec_point", "elliptic_curve_point", "ECPoint"),
    ("ECAdd", "ec_add", "elliptic_curve_addition", "EllipticCurveAddition"),
    (
        "ECAddR",
        "ec_add_r",
        "elliptic_curve_fixed_point",
        "EllipticCurveFixedPointAddition",
    ),
    (
        "ECWindowAddR",
        "ec_add_r",
        "elliptic_curve_window_addition",
        "EllipticCurveWindowAddition",
    ),
    (
        "ECPhaseEstimateR",
        "ec_phase_estimate_r",
        "elliptic_curve_phase_estimation",
        "EllipticCurvePhaseEstimation",
    ),
    (
        "FindECCPrivateKey",
        "find_ecc_private_key",
        "elliptic_curve_double_phase",
        "EllipticCurveDoublePhase",
    ),
)


def test_entire_frozen_initializer_has_only_documentation_and_six_aliases() -> None:
    original = pytest.importorskip("qualtran.bloqs.cryptography.ecc")
    assert version("qualtran") == "0.7.0"
    raw = Path(original.__file__).read_bytes()
    assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "537b429e07bb865f6b51f659f766a5be75018ef5"
    )
    tree = ast.parse(raw)
    assert isinstance(tree.body[0], ast.Expr)
    assert isinstance(tree.body[0].value, ast.Constant)
    assert isinstance(tree.body[0].value.value, str)
    assert "Q = [k] P" in tree.body[0].value.value
    imports = tree.body[1:]
    assert all(isinstance(node, ast.ImportFrom) for node in imports)
    aliases: list[tuple[str, str | None, str | None]] = []
    for node in imports:
        assert isinstance(node, ast.ImportFrom)
        assert node.level == 1
        aliases.extend((alias.name, node.module, alias.asname) for alias in node.names)
    assert aliases == [(name, source, None) for name, source, _, _ in EXPORTS]


@pytest.mark.parametrize("source_name,source_module,native_module,native_name", EXPORTS)
def test_each_package_alias_has_one_existing_native_owner(
    source_name: str, source_module: str, native_module: str, native_name: str
) -> None:
    original = pytest.importorskip("qualtran.bloqs.cryptography.ecc")
    original_child = import_module(f"qualtran.bloqs.cryptography.ecc.{source_module}")
    assert getattr(original, source_name) is getattr(original_child, source_name)
    owned = import_module(f"flagquantum.algorithms.primitives.{native_module}")
    native = getattr(owned, native_name)
    assert native.__module__ == owned.__name__
    assert native.__name__ == native_name
