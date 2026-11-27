"""Exact Boolean cleanup proves arbitrary complex inputs without sampling."""

from itertools import product
from random import Random

import pytest
import torch

from flagquantum import Circuit
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.simulation._clean_releases import require_zero_releases

pytestmark = pytest.mark.unit


def test_nonlinear_boolean_cancellation_preserves_complex_spectator() -> None:
    gates = (
        Instruction("ccx", (0, 1, 2)),
        Instruction("cx", (0, 1)),
        Instruction("ccx", (0, 1, 2)),
        Instruction("cx", (0, 2)),
    )
    ir = CircuitIR(4, gates)
    require_zero_releases(ir, (0, 1, 3), ((4, (2,)),))
    weights = torch.randn(
        4, 2, dtype=torch.complex128, generator=torch.Generator().manual_seed(5163)
    )
    weights /= weights.norm()
    state = torch.zeros(16, dtype=torch.complex128)
    state.reshape(4, 4)[:, :2] = weights
    result = (
        Circuit(4, inputs=state, dtype=torch.complex128).compose(ir).state().flatten()
    )
    assert torch.count_nonzero(result.reshape(4, 4)[:, 2:]) == 0
    expected = state.reshape(2, 2, 4).clone()
    expected[1] = expected[1].flip(0)
    torch.testing.assert_close(result, expected.flatten(), atol=1e-12, rtol=0)


def test_exact_analysis_matches_complete_truth_tables_at_each_release() -> None:
    rng = Random(516)
    choices = (
        Instruction("x", (2,)),
        Instruction("cx", (0, 2)),
        Instruction("cx", (1, 2)),
        Instruction("cx", (0, 1)),
        Instruction("ccx", (0, 1, 2)),
        Instruction("cx", (2, 1)),
    )
    for _ in range(32):
        gates = tuple(rng.choice(choices) for _ in range(8))
        for offset in (0, 4, 8):
            zero = True
            for a, b in product((0, 1), repeat=2):
                bits = [a, b, 0]
                for gate in gates[:offset]:
                    control = all(bits[w] for w in gate.wires[:-1])
                    bits[gate.wires[-1]] ^= int(control)
                zero &= bits[2] == 0
            ir = CircuitIR(3, gates)
            if zero:
                require_zero_releases(ir, (0, 1), ((offset, (2,)),))
            else:
                with pytest.raises(ValueError, match="Nonzero clean release"):
                    require_zero_releases(ir, (0, 1), ((offset, (2,)),))


def test_release_time_is_preserved_and_no_reset_occurs() -> None:
    ir = CircuitIR(2, (Instruction("cx", (0, 1)), Instruction("cx", (0, 1))))
    require_zero_releases(ir, (0,), ((0, (1,)), (2, (1,))))
    with pytest.raises(ValueError, match="instruction 1, wire 1"):
        require_zero_releases(ir, (0,), ((1, (1,)), (2, (1,))))


@pytest.mark.parametrize(
    "gate",
    (Instruction("h", (0,)), Instruction("cx", (0, 1), metadata={"condition": 0})),
)
def test_unknown_phase_or_feedback_never_counts_as_a_proof(gate: Instruction) -> None:
    with pytest.raises(NotImplementedError, match="unconditional X/CX/CCX"):
        require_zero_releases(CircuitIR(2, (gate,)), (0,), ((1, (1,)),))


def test_invalid_layouts_and_expansion_budget_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flagquantum.simulation._clean_releases as analysis

    ir = CircuitIR(3, (Instruction("ccx", (0, 1, 2)),))
    with pytest.raises(ValueError, match="unique circuit wire"):
        require_zero_releases(ir, (0, 0), ())
    with pytest.raises(ValueError, match="valid offsets"):
        require_zero_releases(ir, (0, 1), ((2, (2,)),))
    monkeypatch.setattr(analysis, "_PRODUCT_BUDGET", 0)
    with pytest.raises(NotImplementedError, match="product budget"):
        require_zero_releases(ir, (0, 1), ((1, (2,)),))


def test_literal_product_padding_uses_the_same_analysis() -> None:
    from flagquantum.compiler.registers.finite_field.karatsuba_component import (
        karatsuba_decomposition_program,
    )
    from tests.unit.field_product_checks import bind_product
    from tests.unit.test_field_frobenius_component import dtype

    bound, ports = bind_product(
        karatsuba_decomposition_program(dtype(5, 37), modulus=37)
    )
    releases = tuple(
        (e.instruction_offset, e.wires) for e in bound.events if e.kind == "free"
    )
    assert releases and all(offset is not None for offset, _ in releases)
    points = []
    for offset, wires in releases:
        assert offset is not None
        points.append((offset, wires))
    require_zero_releases(
        bound.circuit, tuple(w for word in ports.values() for w in word), tuple(points)
    )
