"""Native workflows cover notebook steps beyond source-file compilation."""

import ast

import pytest

from examples.synthesis.community.basic_examples.superposition.main import run_example

pytestmark = pytest.mark.integration


def test_superposition_notebook_build_draw_and_named_readout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_example()
    diagram, probabilities = capsys.readouterr().out.rstrip().rsplit("\n", 1)
    assert diagram.strip()
    assert ast.literal_eval(probabilities) == pytest.approx(
        {(word,): 1 / 8 for word in range(8)}, abs=1e-12
    )
