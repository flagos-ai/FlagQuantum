"""Complete frozen chemistry helper numerics and real figure consumers."""

import hashlib
from importlib import import_module
from pathlib import Path

import matplotlib
import numpy as np
import pytest

from examples.algorithms.chemistry_authoring import authoring_integrals
from examples.algorithms.chemistry_scaling import (
    fit_line,
    linear_values,
    plot_power_law,
)
from tests.unit.test_development_tools import REFERENCE

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from qualtran.bloqs.chemistry import chem_tutorials as source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_complete_source_helpers_blob() -> None:
    raw = (REFERENCE / "qualtran/bloqs/chemistry/chem_tutorials.py").read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "e2ae163c0e2d33e072423bf4aa3e388fc1bfe89c"
    )


@pytest.mark.parametrize("name", ["test_rand_ham", "test_fit_linear"])
def test_unmodified_original_numeric_tests(
    name: str, pytestconfig: pytest.Config
) -> None:
    pytestconfig.addinivalue_line("markers", "notebook: original notebook execution")
    raw = (REFERENCE / "qualtran/bloqs/chemistry/chem_tutorials_test.py").read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "2c2e8daa9ee0d13788a43bdbad09f9160fcfcc4d"
    )
    module = import_module("qualtran.bloqs.chemistry.chem_tutorials_test")
    getattr(module, name)()


@pytest.mark.parametrize("power", [1.0, 2.0, 4.5, -0.5])
def test_actual_fit_and_linear_values(power: float) -> None:
    x = np.linspace(1, 200, 10)
    y = 17 * x**power
    np.testing.assert_array_equal(linear_values(x, 2.5, -7), source.linear(x, 2.5, -7))
    np.testing.assert_array_equal(
        fit_line(np.log(x), np.log(y)), source.fit_linear(np.log(x), np.log(y))
    )
    assert np.isclose(fit_line(np.log(x), np.log(y))[0], power)


@pytest.mark.parametrize("spin_orbitals", [0, 1, 2, 3, 4, 9])
@pytest.mark.parametrize("seed", [0, 7, 31])
def test_existing_integral_owner_matches_source(
    spin_orbitals: int, seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = np.random.RandomState(seed)
    monkeypatch.setattr(np.random, "random", generator.random)
    monkeypatch.setattr(np.random, "normal", generator.normal)
    original = source.gen_random_chem_ham(spin_orbitals)
    native = authoring_integrals(spin_orbitals // 2, seed=seed)
    for expected, actual in zip(original, native, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("label", [None, "DF"])
def test_real_plot_line_data_styles_and_rendering(
    label: str | None, tmp_path: Path
) -> None:
    x = np.linspace(1, 200, 10)
    y = 17 * x**2.0
    reference_figure, reference = plt.subplots()
    native_figure, native = plt.subplots()
    try:
        source.plot_linear_log_log(reference, x, y, label=label, color="C2")
        plot_power_law(native, x, y, label=label, color="C2")
        assert len(native.lines) == len(reference.lines) == 2
        for actual, expected in zip(native.lines, reference.lines, strict=True):
            np.testing.assert_array_equal(actual.get_xdata(), expected.get_xdata())
            np.testing.assert_array_equal(actual.get_ydata(), expected.get_ydata())
            assert actual.get_label() == expected.get_label()
            assert actual.get_marker() == expected.get_marker()
            assert actual.get_linestyle() == expected.get_linestyle()
            assert actual.get_color() == expected.get_color()
        assert native.get_xscale() == reference.get_xscale() == "log"
        assert native.get_yscale() == reference.get_yscale() == "log"
        for figure, name in (
            (reference_figure, "reference"),
            (native_figure, "native"),
        ):
            path = tmp_path / (name + ".png")
            figure.savefig(path)
            assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        np.testing.assert_array_equal(
            plt.imread(tmp_path / "native.png"),
            plt.imread(tmp_path / "reference.png"),
        )
    finally:
        plt.close(reference_figure)
        plt.close(native_figure)


@pytest.mark.parametrize("kind", ["empty", "length", "nonfinite"])
def test_solver_failure_matches_source(kind: str) -> None:
    x = np.arange(4, dtype=float)
    y = x.copy()
    if kind == "empty":
        x = y = np.array([], dtype=float)
    elif kind == "length":
        y = np.ones(3, dtype=float)
    else:
        y[0] = np.nan
    with pytest.raises(ValueError) as original:
        source.fit_linear(x, y)
    with pytest.raises(ValueError) as native:
        fit_line(x, y)
    assert str(native.value) == str(original.value)
