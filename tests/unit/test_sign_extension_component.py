"""Typed sign extension and truncation preserve source calls and complex states."""

from collections import Counter

import pytest
import torch

from flagquantum.compiler.registers.bits.sign_extension_component import (
    sign_extension_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.register_types import signed_register_type
from flagquantum.core.ir.program.registers import RegisterType
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "width,extra,encoding",
    [
        (width, extra, encoding)
        for encoding in ("signed", "ones_complement")
        for width, extra in ((1, 1), (2, 1), (2, 3), (3, 1), (3, 4), (8, 8))
        if encoding == "signed" or width > 1
    ],
)
def test_sign_extension_calls_views_ownership_and_source_gates(
    width: int, extra: int, encoding: str
) -> None:
    from qualtran import QInt, QIntOnesComp
    from qualtran.bloqs.arithmetic.conversions.sign_extension import SignExtend
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    dtype, widened = RegisterType(encoding, width), RegisterType(
        encoding, width + extra
    )
    upstream = QInt if encoding == "signed" else QIntOnesComp
    source = SignExtend(upstream(width), upstream(width + extra))
    graph = sign_extension_program(dtype, widened)
    before = (tuple(range(width)),)
    clean = tuple(range(width + 1, width + extra + 1))
    forward = bind_register_program(
        graph, before, clean=clean, allow_workspace_outputs=True
    )
    inverse = sign_extension_program(dtype, widened, adjoint=True)
    back = bind_register_program(
        inverse, forward.outputs, input_ownership=tuple((w, False) for w in clean)
    )
    assert forward.outputs == ((*clean, *before[0]),) and back.outputs == before
    for program, original, bound, inputs, outputs in (
        (graph, source, forward, {"x": before[0]}, {"y": forward.outputs[0]}),
        (inverse, source.adjoint(), back, {"x": forward.outputs[0]}, {"y": before[0]}),
    ):
        compare_layers(
            program, original, {f.name for f in program.functions if f.name != "main"}
        )
        reference = reference_arithmetic_gates(
            original,
            inputs,
            clean if program is graph else (),
            {},
            expected_outputs=outputs,
        )
        assert_source(bound.circuit.instructions, reference)
        assert Counter(g.name for g in bound.circuit.instructions) == {
            "cx": 2 * extra - 1
        }
        model = get_cost_value(original, QECGatesCost())
        assert model.total_t_count() == model.measurement == 0
        # MultiTargetCNOT remains one model Clifford, independently of emitted CX.
        assert model.clifford == 1
    if width > 3:
        return
    logical = torch.arange(1 << (width + 1), dtype=torch.int64)
    coefficients = logical.to(torch.float64)
    amplitudes = torch.complex(coefficients + 1, 7 - 3 * coefficients)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (width + extra + 1), dtype=torch.complex128)
    initial[logical << extra] = amplitudes
    expected = torch.zeros_like(initial)
    sign = logical >> width
    expected[(logical << extra) | (sign * ((1 << extra) - 1))] = amplitudes
    actual = forced_branch(initial, forward.circuit.instructions, ())
    torch.testing.assert_close(actual, expected, rtol=0, atol=3e-12)
    torch.testing.assert_close(
        forced_branch(actual, back.circuit.instructions, ()),
        initial,
        rtol=0,
        atol=3e-12,
    )


def test_sign_extension_rejects_invalid_widths_and_encodings() -> None:
    for left, right in (
        (signed_register_type(2), signed_register_type(2)),
        (RegisterType("uint", 2), RegisterType("uint", 3, parameters=(("bound", 4),))),
        (signed_register_type(2), RegisterType("ones_complement", 3)),
        (RegisterType("ones_complement", 1), RegisterType("ones_complement", 2)),
    ):
        with pytest.raises(ValueError, match="increasing matching dtype|bitsize must"):
            sign_extension_program(left, right)
