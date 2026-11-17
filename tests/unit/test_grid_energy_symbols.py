"""Grid energy diagrams retain source geometry, labels and rendered content."""

import json
from functools import partial
from hashlib import sha1
from pathlib import Path

import pytest

from flagquantum.algorithms.primitives.grid_kinetic import source_grid_kinetic_signature
from flagquantum.algorithms.primitives.grid_potential_resource_model import (
    source_grid_pair_potential_signature,
    source_grid_potential_signature,
)
from flagquantum.drawer.chemistry_symbols import grid_energy_wire_symbol
from flagquantum.drawer.register_score import RegisterStep, layout_register_score
from flagquantum.drawer.register_score_data import (
    export_register_score_json,
    register_score_data,
)
from flagquantum.drawer.register_score_mpl import draw_register_score
from tests.unit.test_drawer_component_score import assert_source_geometry

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_sources():
    from qualtran.bloqs.chemistry.trotter.grid_ham import kinetic, potential

    for module, expected in (
        (kinetic, "50dd860c59c0e29ba982654c544d06b75b7e464d"),
        (potential, "6fd2fae98f800302f658ff61dd1d66fd51df9334"),
    ):
        raw = Path(module.__file__).read_bytes()
        assert sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == expected


@pytest.mark.parametrize("term", ("kinetic", "pair", "potential"))
@pytest.mark.parametrize("label", ("V", "Coulomb", ""))
@pytest.mark.parametrize("electrons,grid_points", ((1, 1), (2, 3), (3, 8)))
def test_energy_score_matches_frozen_source_and_survives_json_render(
    term, label, electrons, grid_points
):
    from matplotlib import pyplot as plt
    from qualtran import BloqBuilder
    from qualtran.bloqs.chemistry.trotter.grid_ham import (
        KineticEnergy,
        PairPotential,
        PotentialEnergy,
    )
    from qualtran.drawing import get_musical_score_data

    width = (grid_points - 1).bit_length() + 1
    if term == "kinetic":
        source = KineticEnergy(electrons, grid_points)
        signature = source_grid_kinetic_signature(electrons, grid_points)
    elif term == "pair":
        source = PairPotential(width, ((1, 2, 3, 4),) * 4, label=label)
        signature = source_grid_pair_potential_signature(width)
    else:
        source = PotentialEnergy(electrons, grid_points, label=label)
        signature = source_grid_potential_signature(electrons, grid_points)

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
        register_wire_symbol=partial(grid_energy_wire_symbol, term=term, label=label),
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
