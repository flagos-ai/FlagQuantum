"""All hopping.py classes retain source layout and renderer semantics."""

import json
from functools import partial
from hashlib import sha1
from pathlib import Path

import pytest

from flagquantum.algorithms.primitives.hubbard_resource_model import (
    source_hubbard_hopping_plaquette_signature,
    source_hubbard_hopping_tile_signature,
)
from flagquantum.drawer.hubbard_symbols import hubbard_hopping_wire_symbol
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
    from qualtran.bloqs.chemistry.trotter.hubbard import hopping

    raw = Path(hopping.__file__).read_bytes()
    assert sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == (
        "7e9a93856c658f93d728d74f9c3bc6cc357a3c0f"
    )


@pytest.mark.parametrize("kind", ("plaquette", "tile", "hwp"))
@pytest.mark.parametrize("pink", (False, True))
@pytest.mark.parametrize("length", (2, 4, 8))
def test_hopping_complete_layout_json_and_render_match_source(kind, pink, length):
    from matplotlib import pyplot as plt
    from qualtran import BloqBuilder
    from qualtran.bloqs.chemistry.trotter.hubbard.hopping import (
        HoppingPlaquette,
        HoppingTile,
        HoppingTileHWP,
    )
    from qualtran.drawing import get_musical_score_data

    plaquette = kind == "plaquette"
    if plaquette:
        source = HoppingPlaquette(0.21)
        signature = source_hubbard_hopping_plaquette_signature()
        width = 1
    else:
        cls = HoppingTileHWP if kind == "hwp" else HoppingTile
        source = cls(length, 0.21, pink=pink)
        signature = source_hubbard_hopping_tile_signature(length)
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
        register_wire_symbol=partial(
            hubbard_hopping_wire_symbol, plaquette=plaquette, pink=pink
        ),
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
