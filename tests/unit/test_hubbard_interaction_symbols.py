"""Both interaction.py classes retain source layout and renderer semantics."""

import json
from hashlib import sha1
from pathlib import Path

import pytest

from flagquantum.algorithms.primitives.hubbard_resource_model import (
    source_hubbard_interaction_signature,
)
from flagquantum.drawer.hubbard_symbols import hubbard_interaction_wire_symbol
from flagquantum.drawer.register_score import RegisterStep, layout_register_score
from flagquantum.drawer.register_score_data import (
    export_register_score_json,
    register_score_data,
)
from flagquantum.drawer.register_score_mpl import draw_register_score
from tests.unit.test_drawer_component_score import assert_source_geometry

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source():
    from qualtran.bloqs.chemistry.trotter.hubbard import interaction

    raw = Path(interaction.__file__).read_bytes()
    assert sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == (
        "5a1d23cd0eeba8e7cf80b73c7b33b2ff7c07184b"
    )


@pytest.mark.parametrize("hamming_weight", (False, True))
@pytest.mark.parametrize("length", (1, 2, 3, 8))
def test_interaction_complete_layout_json_and_render_match_source(
    hamming_weight, length
):
    from matplotlib import pyplot as plt
    from qualtran import BloqBuilder
    from qualtran.bloqs.chemistry.trotter.hubbard.interaction import (
        Interaction,
        InteractionHWP,
    )
    from qualtran.drawing import get_musical_score_data

    cls = InteractionHWP if hamming_weight else Interaction
    source = cls(length, 0.21, -1.7)
    signature = source_hubbard_interaction_signature(length)
    width = length

    builder, inputs = BloqBuilder.from_signature(source.signature)
    graph = builder.finalize(**builder.add_d(source, **inputs))
    ports = tuple(signature)
    identities = tuple(
        f"{reg.name}-{idx}" for reg in source.signature for idx in reg.all_idxs()
    )
    step = RegisterStep(
        "energy", identities, identities, input_registers=ports, output_registers=ports
    )
    layout = layout_register_score(
        dict.fromkeys(identities, width),
        identities,
        (step,),
        input_registers=ports,
        output_registers=ports,
        outputs=identities,
    )
    data = register_score_data(
        layout,
        register_wire_symbol=hubbard_interaction_wire_symbol,
    )
    assert_source_geometry(data, get_musical_score_data(graph), ordered_ports=True)
    serialized = export_register_score_json(data)
    restored = json.loads(serialized)
    assert restored == data
    expected_labels = {
        port["symb_attributes"]["text"] for port in data["soqs"] if port["seq_x"] == 0
    }
    fig, ax = draw_register_score(restored)
    try:
        fig.canvas.draw()
        assert expected_labels <= {artist.get_text() for artist in ax.texts}
        assert export_register_score_json(restored) == serialized
    finally:
        plt.close(fig)
