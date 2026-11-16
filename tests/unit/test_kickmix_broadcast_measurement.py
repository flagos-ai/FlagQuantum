"""Broadcast HMR preserves lane order, repeated ports and exact branches."""

from itertools import product
from pathlib import Path
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.primitives.measurement import (
    measure_x_clear_register_ir,
    reset_x_register_ir,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import forced_branch

pytestmark = pytest.mark.unit


@pytest.fixture
def kickmix() -> Any:
    km = pytest.importorskip("kickmix")
    assert km.__version__ == "0.1" and hasattr(km, "Simulator")
    assert list(Path(km.__file__).parent.glob("*.so"))
    return km


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "targets,slots",
    [((2, 0), (1, 0)), ((2, 0), (1, 1)), ((2, 2), (1, 0))],
)
def test_broadcast_hmr_matches_source_and_every_branch(
    kickmix: Any, targets: tuple[int, int], slots: tuple[int, int]
) -> None:
    km = kickmix
    builder = km.CircuitBuilder()
    builder.hmr(
        km.array([km.q(wire) for wire in targets]),
        km.array([km.b(slot) for slot in slots]),
    )
    builder.cx(km.b(slots[-1]), km.q(1))
    gates = (
        *measure_x_clear_register_ir(targets, slots),
        Instruction("x", (1,), metadata={"conditions": ((slots[-1], 1),)}),
    )
    program = lower_reversible_phase(CircuitIR(3, gates))
    assert (
        encode_kmx(program).splitlines()[1:]
        == str(builder.finish_circuit()).splitlines()
    )
    assert len(gates) == 7
    basis = torch.arange(8)
    initial = (basis + 1).to(torch.complex128) + 1j * (16 - basis)
    initial /= initial.norm()
    norms = 0.0
    for first, second in product((0, 1), repeat=2):
        expected = torch.zeros_like(initial)
        for q0, q1, q2 in product((0, 1), repeat=3):
            if targets[0] == targets[1]:
                # The second HMR sees the physical zero left by the first.
                exponent = first * q2
                output = 4 * q0
            else:
                exponent = first * q2 + second * q0
                output = 0
            output += 2 * (q1 ^ second)
            expected[output] += (-1) ** exponent * initial[4 * q0 + 2 * q1 + q2] / 2
        result = forced_branch(initial, gates, (first, second))
        torch.testing.assert_close(result, expected, rtol=0, atol=1e-15)
        norms += result.norm().item() ** 2
    assert norms == pytest.approx(1.0)


@pytest.mark.ftqc_reference
def test_reset_broadcast_preserves_spectator_density(kickmix: Any) -> None:
    km = kickmix
    builder = km.CircuitBuilder()
    builder.reset(km.array([km.q(2), km.q(0), km.q(2)]))
    gates = reset_x_register_ir((2, 0, 2))
    assert encode_kmx(lower_reversible_phase(CircuitIR(3, gates))).splitlines()[1:] == (
        str(builder.finish_circuit()).splitlines()
    )
    basis = torch.arange(8)
    initial = (basis + 1).to(torch.complex128) + 1j * (16 - basis)
    initial /= initial.norm()
    program = DynamicCircuit(3, inputs=initial, dtype=torch.complex128).append(
        FieldOperation(3, gates)
    )
    expected = torch.zeros((8, 8), dtype=torch.complex128)
    for q0, q2 in product((0, 1), repeat=2):
        projected = initial[[4 * q0 + q2, 4 * q0 + 2 + q2]]
        expected[:4:2, :4:2] += torch.outer(projected, projected.conj())
    torch.testing.assert_close(
        program.density_matrix()[0], expected, rtol=0, atol=1e-15
    )


def test_empty_broadcast_and_invalid_lengths() -> None:
    assert measure_x_clear_register_ir((), ()) == ()
    assert reset_x_register_ir(()) == ()
    with pytest.raises(ValueError, match="equal lengths"):
        measure_x_clear_register_ir((2, 0), (1,))
