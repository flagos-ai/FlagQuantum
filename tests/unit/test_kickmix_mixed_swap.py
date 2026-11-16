"""Locked mixed SWAP and controlled broadcast stream and execution checks."""

from itertools import product
from pathlib import Path
from typing import Any

import pytest
import torch

from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient,
)
from flagquantum.compiler.registers.primitives._mixed_gates import _XBasisTarget
from flagquantum.compiler.registers.primitives.swap import (
    _broadcast_controlled_swap_ir,
    _broadcast_swap_ir,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR
from tests.unit.measurement_checks import forced_branch

pytestmark = pytest.mark.unit


@pytest.fixture
def kickmix() -> Any:
    km = pytest.importorskip("kickmix")
    assert km.__version__ == "0.1" and hasattr(km, "Simulator")
    assert list(Path(km.__file__).parent.glob("*.so"))
    return km


@pytest.mark.ftqc_reference
def test_swap_xor_order_and_self_target_match_source(kickmix: Any) -> None:
    km = kickmix
    b0, b1 = map(ClassicalFieldCoefficient, range(2))
    gates = _broadcast_swap_ir((b0, b1), (b1, b1))
    builder = km.CircuitBuilder()
    builder.swap(km.array([km.b(0), km.b(1)]), km.array([km.b(1), km.b(1)]))
    source = str(builder.finish_circuit())
    assert source.splitlines()[3:] == ["BIT_INVERT b1 if b1"] * 3
    for first, second in product((0, 1), repeat=2):
        setup = (
            ClassicalXor((), 0, bool(first)).instruction(),
            ClassicalXor((), 1, bool(second)).instruction(),
        )
        prefix = encode_kmx(lower_reversible_phase(CircuitIR(1, setup)))
        native = encode_kmx(lower_reversible_phase(CircuitIR(1, (*setup, *gates))))
        assert native == prefix + source + "\n"
        sim = km.Simulator(batch_size=1, seed=1)
        sim.do(km.Circuit(native))
        assert sim.read_across_shots(km.b(0), out=int) == second
        assert sim.read_across_shots(km.b(1), out=int) == 0


@pytest.mark.ftqc_reference
def test_mixed_swap_and_invalid_late_operands(kickmix: Any) -> None:
    km = kickmix
    b0, b1, b2, b3 = map(ClassicalFieldCoefficient, range(4))
    gates = _broadcast_swap_ir((b0, _XBasisTarget(b1), 0), (b2, _XBasisTarget(b3), 1))
    builder = km.CircuitBuilder()
    builder.swap(
        km.array([km.b(0), km.xb(1), km.q(0)]),
        km.array([km.b(2), km.xb(3), km.q(1)]),
    )
    init = tuple(ClassicalXor((), bit).instruction() for bit in range(4))
    program = lower_reversible_phase(CircuitIR(2, (*init, *gates)))
    assert (
        encode_kmx(program).splitlines()[5:]
        == str(builder.finish_circuit()).splitlines()
    )

    invalid = km.CircuitBuilder()
    with pytest.raises(ValueError, match="Can't swap"):
        invalid.swap(km.array([km.q(0), km.b(0)]), km.array([km.q(1), km.xb(0)]))
    assert str(invalid.finish_circuit()) == "SWAP q0 q1"
    with pytest.raises(ValueError, match="matching writable types"):
        _broadcast_swap_ir((0, b0), (1, _XBasisTarget(b0)))
    with pytest.raises(ValueError, match="equal register widths"):
        _broadcast_swap_ir((0, 1), (2,))


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("mode", ["false", "true", "bit", "qubit"])
def test_controlled_swap_broadcast_matches_source_and_coherence(
    kickmix: Any, mode: str
) -> None:
    km = kickmix
    control: dict[str, int | bool | ClassicalFieldCoefficient] = {
        "false": False,
        "true": True,
        "bit": ClassicalFieldCoefficient(0),
        "qubit": 4,
    }
    source_control = {"false": False, "true": True, "bit": km.b(0), "qubit": km.q(4)}
    left, right = (0, 2), (1, 3)
    gates = _broadcast_controlled_swap_ir(control[mode], left, right)
    builder = km.CircuitBuilder()
    builder.cswap(
        source_control[mode], km.array([km.q(0), km.q(2)]), km.array([km.q(1), km.q(3)])
    )
    prefix = ClassicalXor((), 0).instruction()
    text = encode_kmx(lower_reversible_phase(CircuitIR(5, (prefix, *gates))))
    assert text.splitlines()[2:] == str(builder.finish_circuit()).splitlines()
    indices = torch.arange(32)
    initial = (indices + 1).to(torch.complex128) + 1j * (32 - indices)
    initial /= initial.norm()
    for value in (0, 1):
        enable = {
            "false": torch.zeros_like(indices),
            "true": torch.ones_like(indices),
            "bit": torch.full_like(indices, value),
            "qubit": indices & 1,
        }[mode]
        first_exchange = ((indices >> 4) ^ (indices >> 3)) & enable
        second_exchange = ((indices >> 2) ^ (indices >> 1)) & enable
        expected = initial[indices ^ (first_exchange * 24) ^ (second_exchange * 6)]
        prepared = ClassicalXor((), 0, bool(value)).instruction()
        actual = forced_branch(initial, (prepared, *gates), ())
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-16)
