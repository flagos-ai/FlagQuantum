"""Directional tree flags preserve source isometries and measured adjoints."""

from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.comparison.tree_component import (
    mixer_program,
    single_qubit_comparison_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("mixer", (False, True))
def test_directional_flags_source_stream_full_isometry_and_every_branch(
    mixer: bool,
) -> None:
    from qualtran.bloqs.arithmetic.comparison import BiQubitsMixer, SingleQubitCompare

    source = BiQubitsMixer() if mixer else SingleQubitCompare()
    factory = mixer_program if mixer else single_qubit_comparison_program
    logical, flags, measurements = (4, 3, 2) if mixer else (2, 2, 1)
    inputs = ((0, 1), (2, 3)) if mixer else ((0,), (1,))
    clean = tuple(range(logical, logical + flags))
    forward = bind_register_program(
        factory(), inputs, clean=clean, allow_workspace_outputs=True
    )
    backward = bind_register_program(
        factory(adjoint=True),
        forward.outputs,
        input_ownership=tuple((q, False) for q in clean),
    )
    input_names = ("x", "y") if mixer else ("a", "b")
    output_names = (
        ("x", "y", "ancilla") if mixer else ("a", "b", "less_than", "greater_than")
    )
    raw = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 7 - 3 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (logical + flags + 1), dtype=torch.complex128)
    initial[:: 1 << (flags + 1)] = amplitudes[::2]
    initial[1 :: 1 << (flags + 1)] = amplitudes[1::2]
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        1 << (logical + flags), 1 << logical
    )
    expected = (matrix @ amplitudes.reshape(-1, 2)).flatten()
    for bound, original, names, ports, before, after, events in (
        (forward, source, input_names, inputs, initial, expected, 0),
        (
            backward,
            source.adjoint(),
            output_names,
            forward.outputs,
            expected,
            initial,
            measurements,
        ),
    ):
        reference = reference_arithmetic_gates(
            original,
            dict(zip(names, ports, strict=True)),
            clean if not events else (),
            MeasurementSlots(bound.circuit.instructions),
            preserve_components=True,
            expected_outputs=dict(
                zip(
                    input_names if events else output_names,
                    inputs if events else forward.outputs,
                    strict=True,
                )
            ),
        )
        assert_source(bound.circuit.instructions, reference)
        for history in product((0, 1), repeat=events):
            torch.testing.assert_close(
                forced_branch(before, bound.circuit.instructions, history),
                after / 2 ** (events / 2),
                rtol=0,
                atol=3e-12,
            )
