"""Repeated noise locations retain exact factory operators and separate storage."""

from collections.abc import Callable

import pytest
import torch

from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.runtime_config import runtime_config
from flagquantum.noise import KrausChannel
from flagquantum.qec.execution import sampling

pytestmark = pytest.mark.unit


def _plan() -> sampling._MeasurementPlan:
    return sampling._MeasurementPlan(
        CircuitIR(n_wires=2, instructions=(Instruction("h", (0,)),)),
        0,
        1,
        (),
        {},
        {},
    )


def _locations() -> tuple[sampling._NoiseLocation, ...]:
    return tuple(
        sampling._NoiseLocation("data", channel, 0, wire, 0, rate)
        for channel, rate in (
            ("bit_flip", 0.005),
            ("phase_flip", 0.007),
            ("y_flip", 0.009),
            ("bit_flip", 0.007),
        )
        for wire in (0, 1)
    )


def test_repeated_rates_build_one_template_per_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts: list[tuple[str, float]] = []
    originals = sampling._DATA_FAULT_CHANNELS

    def counted(
        name: str, factory: Callable[[float], KrausChannel]
    ) -> Callable[[float], KrausChannel]:
        def make(rate: float) -> KrausChannel:
            counts.append((name, rate))
            return factory(rate)

        return make

    monkeypatch.setattr(
        sampling,
        "_DATA_FAULT_CHANNELS",
        {name: counted(name, factory) for name, factory in originals.items()},
    )
    first = sampling._noisy_program(_plan(), _locations())
    assert len(first.instructions) == 9
    assert len(counts) == 4
    second = sampling._noisy_program(_plan(), _locations())
    assert len(counts) == 8
    assert (
        first.instructions[0].matrix[0].data_ptr()
        != second.instructions[0].matrix[0].data_ptr()
    )


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("rate", [0.0, 0.005, 1.0])
@pytest.mark.parametrize("channel", ["bit_flip", "phase_flip", "y_flip"])
def test_operators_match_factory_and_locations_do_not_alias(
    dtype: str, rate: float, channel: sampling._ChannelName
) -> None:
    with runtime_config(complex_dtype=dtype):
        locations = tuple(
            sampling._NoiseLocation("data", channel, 0, wire, 0, rate)
            for wire in (0, 1)
        )
        program = sampling._noisy_program(_plan(), locations)
        expected = sampling._DATA_FAULT_CHANNELS[channel](rate).kraus
    first, second = program.instructions[:2]
    for a, b, original in zip(first.matrix, second.matrix, expected, strict=True):
        assert torch.equal(a, original) and torch.equal(b, original)
        assert a.dtype == b.dtype == original.dtype
        assert a.data_ptr() != b.data_ptr()
    saved = tuple(op.clone() for op in second.matrix)
    first.matrix[0].fill_(42)
    assert all(torch.equal(a, b) for a, b in zip(second.matrix, saved, strict=True))


def test_mixed_rates_preserve_location_order_and_source_instruction() -> None:
    plan = _plan()
    program = sampling._noisy_program(plan, _locations())
    assert program.instructions[-1] is plan.program.instructions[0]
    for instruction, location in zip(
        program.instructions[:-1], _locations(), strict=True
    ):
        assert instruction.name == location.channel
        assert instruction.wires == (location.wire,)
        assert instruction.params == {"probability": location.probability}
        for actual, expected in zip(
            instruction.matrix,
            sampling._DATA_FAULT_CHANNELS[location.channel](location.probability).kraus,
            strict=True,
        ):
            assert torch.equal(actual, expected)


@pytest.mark.parametrize("scalar_dtype", [torch.float32, torch.float64])
def test_default_scalar_precision_matches_original_factory(
    scalar_dtype: torch.dtype,
) -> None:
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(scalar_dtype)
        program = sampling._noisy_program(_plan(), _locations())
        for instruction, location in zip(
            program.instructions[:-1], _locations(), strict=True
        ):
            for actual, expected in zip(
                instruction.matrix,
                sampling._DATA_FAULT_CHANNELS[location.channel](
                    location.probability
                ).kraus,
                strict=True,
            ):
                assert torch.equal(actual, expected)
    finally:
        torch.set_default_dtype(previous)
