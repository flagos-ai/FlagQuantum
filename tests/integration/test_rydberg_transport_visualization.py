"""The view uses the actual native geometry and source timings."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("numpy")
import matplotlib.pyplot as plt
import numpy as np

from examples.visualize_rydberg_transport import transport_figure

pytestmark = pytest.mark.integration


def test_native_transport_view_has_correct_paths_schedule_and_export(
    tmp_path: Path,
) -> None:
    figure = transport_figure(samples=51)
    try:
        geometry, timeline = figure.axes
        paths = {line.get_label(): line for line in geometry.lines}
        np.testing.assert_allclose(
            paths["a (commanded)"].get_xdata(), np.zeros(51), atol=0
        )
        np.testing.assert_allclose(
            paths["b (commanded)"].get_xdata(),
            2 + 2 * np.linspace(0, 1, 51),
            atol=1e-14,
            rtol=0,
        )
        for line in paths.values():
            np.testing.assert_allclose(line.get_ydata(), np.zeros(51), atol=0)
        assert [label.get_text() for label in timeline.get_yticklabels()] == [
            "move b (aod)",
            "pulse a,b (laser)",
        ]
        assert [(bar.get_x(), bar.get_width()) for bar in timeline.patches] == [
            (0.0, 1.0),
            (0.0, 1.0),
        ]
        output = tmp_path / "transport.svg"
        with plt.rc_context({"svg.fonttype": "none"}):
            figure.savefig(output)
        root = ET.parse(output).getroot()
        text = " ".join(node.text or "" for node in root.iter())
        assert "Commanded geometry is not a presence record." in text
        assert "Native source schedule" in text
        assert output.stat().st_size > 1000
    finally:
        plt.close(figure)
