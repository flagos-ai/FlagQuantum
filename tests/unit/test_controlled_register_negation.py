"""Source controlled negation with coherent controls and exact branch cleanup."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.arithmetic.constant_add_resource_model import (
    source_controlled_constant_add_resource_formula,
)
from flagquantum.compiler.registers.arithmetic.controlled_addition import (
    controlled_negate_ir,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import (
    layout,
    reference_arithmetic_gates,
)
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("enabled", [0, 1])
@pytest.mark.parametrize("width", [1, 2, 3, 5, 8])
def test_controlled_negation_matches_source_primitives_and_resources(
    width: int, enabled: int
) -> None:
    from qualtran import CtrlSpec, QUInt
    from qualtran.bloqs.arithmetic.negate import Negate
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    target, constant, carries, bits = layout(width)
    control = 3 * width - 1
    source = Negate(QUInt(width)).controlled(CtrlSpec(cvs=enabled))
    expected = reference_arithmetic_gates(
        source,
        {"ctrl": (control,), "x": target},
        (*constant, *carries),
        dict(zip(carries, bits, strict=True)),
        expected_outputs={"ctrl": (control,), "x": target},
    )
    actual = controlled_negate_ir(
        control, target, constant, carries, bits, control_value=enabled
    )
    assert_source(actual, expected)
    counts = Counter(g.name for g in actual)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * (width - 1)
    assert counts["measure"] == model.measurement == width - 1
    increment = source_controlled_constant_add_resource_formula(
        width, 1, control_value=enabled
    )
    assert model.clifford == increment.clifford + width
    assert model.and_bloq == increment.and_bloq
    assert get_cost_value(source, QubitCount()) == increment.logical_qubits == 3 * width
    assert counts["x"] == (2 if enabled == 0 else 0)


@pytest.mark.parametrize("enabled", [0, 1])
@pytest.mark.parametrize("width", [1, 2, 3])
def test_every_controlled_negation_branch_and_involution(
    width: int, enabled: int
) -> None:
    target, constant, carries, bits = layout(width)
    control = 3 * width - 1
    gates = controlled_negate_ir(
        control, target, constant, carries, bits, control_value=enabled
    )
    indices = torch.arange(1 << (width + 2))
    word, control_bit, spectator = indices >> 2, (indices >> 1) & 1, indices & 1
    output = torch.where(control_bit == enabled, (-word) % (1 << width), word)
    initial_locations = (word << (2 * width + 1)) | (control_bit << 1) | spectator
    output_locations = (output << (2 * width + 1)) | (control_bit << 1) | spectator
    weights = torch.complex(indices.double() + 1, 3 * indices.double() - 5)
    weights /= weights.norm()
    state = torch.zeros(1 << (control + 2), dtype=torch.complex128)
    expected = torch.zeros_like(state)
    state[initial_locations] = weights
    expected[output_locations] = weights
    scale = 2 ** ((width - 1) / 2)
    for history in product((0, 1), repeat=width - 1):
        torch.testing.assert_close(
            forced_branch(state, gates, history), expected / scale, atol=1e-12, rtol=0
        )
        torch.testing.assert_close(
            forced_branch(expected, gates, history), state / scale, atol=1e-12, rtol=0
        )


def test_controlled_negation_rejects_invalid_polarity_and_alias() -> None:
    target, constant, carries, bits = layout(2)
    with pytest.raises(ValueError, match="zero or one"):
        controlled_negate_ir(5, target, constant, carries, bits, control_value=2)
    with pytest.raises(ValueError, match="distinct"):
        controlled_negate_ir(target[0], target, constant, carries, bits)
