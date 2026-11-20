"""Verify the complete original slow-domain affine callback independently of gates."""

from hashlib import sha1
from itertools import product
from pathlib import Path
from typing import Any

import pytest

from flagquantum.algorithms.primitives.elliptic_curve_addition import (
    EllipticCurveAddition,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source_files() -> None:
    from qualtran.bloqs.cryptography.ecc import ec_add, ec_add_test

    for module, digest in (
        (ec_add, "2caf4f002152e51f1a09048d459c0b839280a4d1"),
        (ec_add_test, "2fa735023da4c1b79d37079a7945aa2929e748b8"),
    ):
        raw = Path(module.__file__).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest


def original_slow_cases() -> list[tuple[int, int, int, int, int, int]]:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.cryptography.ecc import ec_add_test

    axes: list[tuple[tuple[str, ...], Any]] = []
    for marker in ec_add_test.test_ec_add_classical.pytestmark:
        if marker.name == "parametrize":
            names, values = marker.args
            axes.append(
                (
                    (
                        tuple(names.replace(" ", "").split(","))
                        if isinstance(names, str)
                        else tuple(names)
                    ),
                    values,
                )
            )
    assert len(axes) == 3
    cases = []
    for choices in product(*(values for _, values in axes)):
        parameters = {
            name: value
            for (names, _), choice in zip(axes, choices, strict=True)
            for name, value in zip(names, choice, strict=True)
        }
        cases.append(tuple(parameters[name] for name in ("n", "m", "a", "b", "x", "y")))
    assert len(cases) == 504
    return cases


@pytest.mark.parametrize("n,m,a,b,x,y", original_slow_cases())
def test_full_original_slow_matrix_affine_callback(
    n: int, m: int, a: int, b: int, x: int, y: int
) -> None:
    from qualtran import QMontgomeryUInt
    from qualtran.bloqs.cryptography.ecc import ECAdd

    modulus = 17
    encoding = QMontgomeryUInt(n, modulus)
    tangent = modulus if b == 0 else 3 * a * a * pow(2 * b, -1, modulus) % modulus
    raw = {
        name: encoding.uint_to_montgomery(value)
        for name, value in zip(
            ("a", "b", "x", "y", "lam_r"), (a, b, x, y, tangent), strict=True
        )
    }
    source = ECAdd(n, modulus, m)
    expected = source.on_classical_vals(**raw)
    native = EllipticCurveAddition(n, modulus, m)
    assert native.classical_values(**raw) == tuple(
        expected[name] for name in ("a", "b", "x", "y", "lam_r")
    )
    from qualtran.bloqs.cryptography.ecc import ec_add as original_module

    values = raw.copy()
    stages = native._stages()
    for index, original_name in enumerate(
        ("One", "Two", "Three", "Four", "Five", "Six")
    ):
        native_stage = stages[index]
        original_stage_type = getattr(original_module, "_ECAddStep" + original_name)
        original_stage = (
            original_stage_type(n, modulus)
            if original_name in {"One", "Six"}
            else original_stage_type(n, modulus, m)
        )
        inputs = {
            reg.name: values[reg.name] for reg in original_stage.signature.lefts()
        }
        outputs = original_stage.on_classical_vals(**inputs)
        output_names = tuple(reg.name for reg in native_stage.source_signature.rights())
        assert native_stage.classical_values(**inputs) == tuple(
            outputs[name] for name in output_names
        )
        values.update(outputs)
    # The original test also checks the one-level staged callback. Deeper
    # physical arithmetic and its known cleanup defects have separate evidence.
    assert source.decompose_bloq().call_classically(**raw) == tuple(expected.values())


def test_numeric_callback_runs_without_reference_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    original_import = builtins.__import__

    def owned_import(name: str, *args: Any, **kwargs: Any) -> Any:
        assert name.split(".")[0] != "qualtran"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", owned_import)
    op = EllipticCurveAddition(3, 3)
    assert op.classical_values(0, 2, 0, 2, 1) == (0, 2, 2, 0, 1)
