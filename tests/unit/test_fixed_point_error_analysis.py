"""Whole error-analysis notebook, seeded outputs and original failed bounds."""

import builtins
import json
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from hashlib import sha1
from io import StringIO
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from examples.algorithms import fixed_point_error_analysis as native
from tests.unit.test_addition_file_audit import execute

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_complete_source_analysis_and_native_scalar_arithmetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[2]
    raw = (
        root.parent
        / "ftqc-reference-env/qualtran/bloqs/arithmetic/error_analysis_for_fxp_arithmetic.ipynb"
    ).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "57a9927840390329d583a7203c941ece4386ef47"
    )
    original = json.loads(raw)
    notebook = json.loads(
        (root / "examples/notebooks/fixed_point_error_analysis.ipynb").read_text()
    )
    assert len(original["cells"]) == len(notebook["cells"]) == 8
    assert [c["cell_type"] for c in original["cells"]] == [
        c["cell_type"] for c in notebook["cells"]
    ]
    original_output = StringIO()
    with redirect_stdout(original_output):
        reference = execute(original)
    original_import = builtins.__import__

    def native_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name.split(".")[0] in {"qualtran", "cirq", "fxpmath"}:
            raise AssertionError(
                "Native error analysis must use owned integer arithmetic"
            )
        return original_import(name, globals, locals, fromlist, level)

    native_output = StringIO()
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", native_import)
        with redirect_stdout(native_output):
            execute(notebook)
    assert native_output.getvalue() == original_output.getvalue()
    failures = [
        row
        for row in native_output.getvalue().splitlines()
        if row.startswith("Failed!")
    ]
    assert failures == [
        "Failed! d=14, eps=0.001",
        "Failed! d=18, eps=0.0001",
        "Failed! d=21, eps=1e-05",
    ]
    # Compare every seeded result, including later samples after the source's
    # diagnostic wrapper stopped at its first failed approximation bound.
    for eps in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-9):
        for integer_width in (5, 8, 11):
            real_width = native.get_bitsize_for_fxp_mul_with_integer(eps, integer_width)
            assert real_width == reference["get_bitsize_for_fxp_mul_with_integer"](
                eps, integer_width
            )
            rng = np.random.default_rng(int(eps * integer_width * 1e9))
            for _ in range(100):
                a = float(rng.random(1)[0])
                b = int(rng.integers(0, 1 << integer_width))
                expected = reference["mul_with_int_via_repeated_add"](
                    a, b, real_width, integer_width
                )
                assert (
                    native.mul_with_int_via_repeated_add(
                        a, b, real_width, integer_width
                    )
                    == expected.get_val()
                )
        if eps == 1e-9:
            continue
        for helper, bound, offset in (
            ("mul_via_repeated_add", "get_bitsize_for_fxp_mul", -1),
            ("square_via_repeated_add", "get_bitsize_for_fxp_square", 0),
        ):
            d = getattr(native, bound)(eps) + offset
            assert d == reference[bound](eps) + offset
            rng = np.random.default_rng(int(eps * 1e9))
            for _ in range(100):
                args = tuple(float(v) for v in rng.random(2 if offset else 1))
                expected = reference[helper](*args, d).get_val()
                assert getattr(native, helper)(*args, d) == expected
    # Integer one is quantized to the largest pure fraction before multiplication.
    assert native.square_via_repeated_add(0.4106776946403047, 14) == 2754 / 16384
    for width in range(1, 7):
        for raw_word in range(1 << width):
            value = raw_word / (1 << width)
            assert (
                native.square_via_repeated_add(value, width)
                == reference["square_via_repeated_add"](value, width).get_val()
            )
    for width, fraction in ((3, 3), (6, 3), (3, 6)):
        for value in (0.0, 0.125, 0.5, 0.875, np.nextafter(1.0, 0.0)):
            assert (
                native.fxp(value, fraction, d_word=width)
                == reference["fxp"](value, fraction, d_word=width).val
            )
    for invalid in (-0.1, 1.0):
        with pytest.raises(AssertionError):
            native.fxp(invalid, 4)
        with pytest.raises(AssertionError):
            reference["fxp"](invalid, 4)


def test_unsigned_scalar_width_admission() -> None:
    for fraction, width in ((0, None), (-1, None), (3, 0)):
        with pytest.raises(ValueError, match="positive word"):
            native.fxp(0.0, fraction, d_word=width)
