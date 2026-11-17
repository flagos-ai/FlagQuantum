"""Verify the Hubbard SELECT leaf diagrams and renderer compatibility."""

from hashlib import sha1
from pathlib import Path

import pytest

from examples.algorithms.hubbard_qubitization_resources import notebook_consumer_score
from flagquantum.drawer.register_score_data import export_register_score_json
from flagquantum.drawer.register_score_mpl import draw_register_score
from tests.unit.test_drawer_component_score import assert_source_geometry

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source():
    from qualtran.bloqs.chemistry.hubbard_model.qubitization import select_hubbard

    raw = Path(select_hubbard.__file__).read_bytes()
    assert sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == (
        "6d912f2ae1f63730cc296cfc35f68eb9ccbabb84"
    )


@pytest.mark.parametrize("majorana", ("X", "Y", None))
@pytest.mark.parametrize("control", (None, 0, 1))
@pytest.mark.parametrize("side", (1, 3))
def test_hubbard_leaf_score_matches_source(majorana, control, side):
    from matplotlib import pyplot as plt
    from qualtran import BloqBuilder
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard import (
        HubbardMajorannaOperator,
        HubbardSpinUpZ,
    )
    from qualtran.drawing import get_musical_score_data

    source = (
        HubbardSpinUpZ(side, side, control_val=control)
        if majorana is None
        else HubbardMajorannaOperator(side, side, majorana, control_val=control)
    )
    builder, inputs = BloqBuilder.from_signature(source.signature)
    composite = builder.finalize(**builder.add_d(source, **inputs))
    score = notebook_consumer_score(
        side, majorana=majorana, controlled=control is not None
    )
    assert_source_geometry(score, get_musical_score_data(composite), ordered_ports=True)
    assert score["vlines"][0]["label"] == {"text": ""}
    text = next(
        p["symb_attributes"]["text"]
        for p in score["soqs"]
        if p["seq_x"] == 0 and p["symb_attributes"].get("text", "").startswith("$")
    )
    before = export_register_score_json(score)
    fig, ax = draw_register_score(score)
    try:
        fig.canvas.draw()
        assert text in {artist.get_text() for artist in ax.texts}
        assert export_register_score_json(score) == before
    finally:
        plt.close(fig)
