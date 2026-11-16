"""Locked CX and CZ broadcast dispatch across mixed target kinds."""

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
from flagquantum.compiler.registers.primitives._mixed_gates import (
    _broadcast_cx,
    _broadcast_cz,
    _Control,
    _XBasisTarget,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR, Instruction
from tests.unit.fixtures.kickmix_source_oracle import normalized_stream
from tests.unit.measurement_checks import forced_branch

pytestmark = pytest.mark.unit
_B0 = ClassicalFieldCoefficient(0)
_B1 = ClassicalFieldCoefficient(1)
_B2 = ClassicalFieldCoefficient(2)
_B3 = ClassicalFieldCoefficient(3)


@pytest.fixture
def locked_kickmix() -> Any:
    km = pytest.importorskip("kickmix")
    assert km.__version__ == "0.1" and hasattr(km, "Simulator")
    assert list(Path(km.__file__).parent.glob("*.so"))
    return km


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("phase", [False, True])
def test_broadcast_dispatch_matches_locked_source_matrix(
    locked_kickmix: Any, phase: bool
) -> None:
    km = locked_kickmix
    left: tuple[tuple[tuple[_Control, ...], tuple[Any, ...]], ...] = (
        ((False, True), (False, True)),
        ((_B0, _B1), (km.b(0), km.b(1))),
        ((0, 1), (km.q(0), km.q(1))),
        ((_B0, True), (km.b(0), True)),
    )
    right: list[tuple[tuple[_Control | _XBasisTarget, ...], tuple[Any, ...]]] = [
        ((2, 3), (km.q(2), km.q(3))),
        ((_B1, _B2), (km.b(1), km.b(2))),
        ((3, _B2), (km.q(3), km.b(2))),
    ]
    if phase:
        right.append(((True, False), (True, False)))
    else:
        right.extend(
            (
                ((_XBasisTarget(_B2), _XBasisTarget(_B3)), (km.xb(2), km.xb(3))),
                (
                    (_XBasisTarget(False), _XBasisTarget(True)),
                    (km.xbool(False), km.xbool(True)),
                ),
                ((_XBasisTarget(_B3), 3), (km.xb(3), km.q(3))),
            )
        )
    initials = tuple(ClassicalXor((), bit).instruction() for bit in range(4))
    for (controls, source_controls), (targets, source_targets) in product(left, right):
        builder = km.CircuitBuilder()
        initial = km.q(4)
        builder.x(initial)
        dispatch = builder.cz if phase else builder.cx
        try:
            dispatch(km.array(source_controls), km.array(source_targets))
        except ValueError as error:
            assert not phase and "cannot flip classical bit target" in str(error)
            assert str(builder.finish_circuit()) == "X q4"
            with pytest.raises(ValueError, match="cannot flip a classical target"):
                _broadcast_cx(controls, targets)
            continue
        if phase:
            z_operands = tuple(
                target for target in targets if not isinstance(target, _XBasisTarget)
            )
            assert len(z_operands) == len(targets)
            gates = _broadcast_cz(controls, z_operands)
        else:
            gates = _broadcast_cx(controls, targets)
        ir = CircuitIR(5, (*initials, Instruction("x", (4,)), *gates))
        native = encode_kmx(lower_reversible_phase(ir)).splitlines()[5:]
        source = str(builder.finish_circuit())
        assert normalized_stream("\n".join(native)) == normalized_stream(source)


@pytest.mark.ftqc_reference
def test_ordered_classical_aliases_clear_using_source_opcode(
    locked_kickmix: Any,
) -> None:
    km = locked_kickmix
    gates = _broadcast_cx((_B0, _B1, _B2), (_B1, _B2, _B2))
    source = km.CircuitBuilder()
    source.cx(
        km.array([km.b(0), km.b(1), km.b(2)]), km.array([km.b(1), km.b(2), km.b(2)])
    )
    body = str(source.finish_circuit())
    assert body.endswith("BIT_INVERT b2 if b2")
    for values in product((0, 1), repeat=3):
        setup = tuple(
            ClassicalXor((), bit, bool(value)).instruction()
            for bit, value in enumerate(values)
        )
        prepared = encode_kmx(lower_reversible_phase(CircuitIR(1, setup)))
        native = encode_kmx(lower_reversible_phase(CircuitIR(1, (*setup, *gates))))
        assert native == prepared + body + "\n"
        sim = km.Simulator(batch_size=1, seed=1)
        sim.do(km.Circuit(native))
        result = tuple(sim.read_across_shots(km.b(bit), out=int) for bit in range(3))
        assert result == (values[0], values[1] ^ values[0], 0)


def test_x_basis_kickback_preserves_full_complex_state() -> None:
    gates = _broadcast_cx(
        (0, _B0, True, _B2),
        (_XBasisTarget(_B1), _XBasisTarget(_B1), _XBasisTarget(True), 2),
    )
    indices = torch.arange(16)
    initial = (indices + 1).to(torch.complex128) + 1j * (16 - indices)
    initial /= initial.norm()
    for first, second, third in product((0, 1), repeat=3):
        setup = tuple(
            ClassicalXor((), bit, bool(value)).instruction()
            for bit, value in enumerate((first, second, third))
        )
        negative = ((indices >> 3) & second) ^ (first & second) ^ 1
        expected = initial[indices ^ (third << 1)] * (1 - 2 * negative)
        torch.testing.assert_close(
            forced_branch(initial, (*setup, *gates), ()), expected, atol=1e-15, rtol=0
        )


@pytest.mark.ftqc_reference
def test_x_basis_controls_and_length_mismatch_refused_by_both_builders(
    locked_kickmix: Any,
) -> None:
    km = locked_kickmix
    bad: Any = (_XBasisTarget(_B1),)
    for native, original in (
        (_broadcast_cx, km.CircuitBuilder().cx),
        (_broadcast_cz, km.CircuitBuilder().cz),
    ):
        with pytest.raises(ValueError, match="Z-basis control"):
            native(bad, (0,))
        with pytest.raises(ValueError):
            original(km.array([km.xb(1)]), km.array([km.q(0)]))
        with pytest.raises(ValueError, match="equal lengths"):
            native((True,), (0, 1))
        with pytest.raises(ValueError, match="len"):
            original(km.array([True]), km.array([km.q(0), km.q(1)]))
