"""Compare the chemistry number-phase score to the frozen source."""

from functools import partial
from hashlib import sha1
from pathlib import Path

import pytest

from flagquantum.algorithms.primitives.chemistry_number_phase import (
    ChemistryNumberPhase,
)
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.drawer.chemistry_symbols import number_phase_wire_symbol
from flagquantum.drawer.register_score import RegisterStep, layout_register_score
from flagquantum.drawer.register_score_data import register_score_data
from tests.unit.test_drawer_component_score import assert_source_geometry

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source():
    from qualtran.bloqs.chemistry import black_boxes

    raw = Path(black_boxes.__file__).read_bytes()
    assert sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == (
        "f9670a2ebaf2fe0973323cf7b532c88142970902"
    )


@pytest.mark.parametrize("values", [(), (0,), (1,), (0, 1), (1, 0, 1), (1,) * 10])
def test_phase_score_preserves_polarity_target_and_full_connector(values):
    from qualtran import BloqBuilder
    from qualtran.bloqs.chemistry.black_boxes import ApplyControlledZs
    from qualtran.drawing import get_musical_score_data

    width = 3
    source = ApplyControlledZs(values, width)
    builder, inputs = BloqBuilder.from_signature(source.signature)
    graph = builder.finalize(**builder.add_d(source, **inputs))
    if values:
        ports = tuple(ChemistryNumberPhase(values, width).signature)
    else:
        ports = (
            Register("ctrls", quantum_bit_type(), (0,)),
            Register("system", opaque_register_type(width)),
        )
    controls = tuple(f"control-{index}" for index in range(len(values)))
    identities = (*controls, "system")
    step = RegisterStep(
        "phase",
        identities,
        identities,
        input_registers=ports,
        output_registers=ports,
    )
    score = layout_register_score(
        dict.fromkeys(controls, 1) | {"system": width},
        identities,
        (step,),
        input_registers=ports,
        output_registers=ports,
        outputs=identities,
    )
    result = register_score_data(
        score, register_wire_symbol=partial(number_phase_wire_symbol, values=values)
    )
    assert_source_geometry(result, get_musical_score_data(graph), ordered_ports=True)


def test_empty_control_source_drawing_does_not_imply_executable_decomposition():
    from qualtran import BloqError
    from qualtran.bloqs.chemistry.black_boxes import ApplyControlledZs
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = ApplyControlledZs((), 3)
    with pytest.raises(BloqError, match="ZGate does not accept Soquets"):
        source.decompose_bloq()
    assert get_cost_value(source, QECGatesCost()).toffoli == -1
    with pytest.raises(ValueError, match="binary control values"):
        ChemistryNumberPhase((), 3)
