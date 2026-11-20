"""Approved wide fixed-point storage preserves raw views and shared arithmetic."""

from collections import Counter
from itertools import product
from math import prod

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.controlled_programs import controlled_x_program
from flagquantum.compiler.registers.arithmetic.addition import add_adjoint_ir, add_ir
from flagquantum.compiler.registers.arithmetic.addition_component import (
    addition_component_program,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.layout.component_cast import cast_word_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.register_types import fixed_point_register_type
from flagquantum.core.ir.program.register_verifier import executable_register_width
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_weighted_columns_component import normalized_feedback

np = pytest.importorskip("numpy")

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", (1, 2, 3, 4, 8))
@pytest.mark.parametrize("fraction", (-10000, -7, -2, -1, 0, 1))
def test_concrete_unsigned_width_and_fraction_preserve_descriptor(
    width: int, fraction: int
) -> None:
    dtype = fixed_point_register_type(width, fraction)
    assert executable_register_width(dtype) == width
    assert dict(dtype.parameters) == {"fractional_width": fraction, "signed": False}
    assert executable_register_width(fixed_point_register_type(width, width)) == width


@pytest.mark.parametrize("width,fraction", ((2, 2), (3, 1), (4, -2), (8, 8)))
@pytest.mark.parametrize("shape", ((), (2,), (1, 2), (0, 2)))
def test_source_cast_identity_raw_bits_and_row_major_binding(
    width: int, fraction: int, shape: tuple[int, ...]
) -> None:
    from qualtran._infra.data_types import QFxp, QUInt
    from qualtran.bloqs.bookkeeping.cast import Cast
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    fixed = fixed_point_register_type(width, fraction)
    unsigned = RegisterType("uint", width)
    leaf = cast_word_function("as_uint", fixed, unsigned, shape)
    reverse = cast_word_function("as_fixed", unsigned, fixed, shape)
    builder = ComponentProgramBuilder(
        Signature((Register("a", fixed, shape),)), (leaf, reverse)
    )
    converted = builder.call_dict("as_uint", {"reg": builder.inputs["a"]})
    restored = builder.call_dict("as_fixed", {"reg": converted["reg"]})
    record = builder.finalize({"a": restored["reg"]})
    words = tuple(tuple(range(i * width, (i + 1) * width)) for i in range(prod(shape)))
    bound = bind_register_program(record, words, clean=(0,) if not words else ())
    assert bound.outputs == words
    assert bound.circuit.instructions == ()
    assert bound.events == ()
    source_dtype = QFxp(width, fraction)
    for value in range(1 << width):
        bits = [value >> bit & 1 for bit in reversed(range(width))]
        assert source_dtype.from_bits(bits) == value
        assert source_dtype.to_bits(value) == bits
    source = Cast(source_dtype, QUInt(width), shape)
    views = np.arange(prod(shape) * width).reshape((*shape, width))
    operation, outputs = source.as_cirq_op(None, reg=views)
    assert operation is None and outputs["reg"] is views
    cost = get_cost_value(source, QECGatesCost())
    assert cost.total_t_count() == cost.measurement == cost.clifford == 0


def fixed_add(width: int, fraction: int, *, adjoint: bool = False) -> ProgramRecord:
    fixed = fixed_point_register_type(width, fraction)
    unsigned = RegisterType("uint", width)
    builder = ComponentProgramBuilder(
        Signature((Register("a", fixed), Register("b", fixed))),
        (
            cast_word_function("as_uint", fixed, unsigned),
            cast_word_function("as_fixed", unsigned, fixed),
        ),
    )
    inputs = {
        name: builder.call_dict("as_uint", {"reg": builder.inputs[name]})["reg"]
        for name in ("a", "b")
    }
    result = builder.call_program(
        addition_component_program(width, adjoint=adjoint), inputs
    )
    return builder.finalize(
        {
            name: builder.call_dict("as_fixed", {"reg": result[name]})["reg"]
            for name in ("a", "b")
        }
    )


@pytest.mark.parametrize("width,fraction", ((1, -7), (2, -2), (2, 2), (3, 1)))
def test_shared_add_complex_states_every_small_history_and_exact_inverse(
    width: int, fraction: int
) -> None:
    words = (tuple(range(width)), tuple(range(width, 2 * width)))
    work = tuple(range(2 * width, 3 * width - 1))
    forward = bind_register_program(fixed_add(width, fraction), words, clean=work)
    inverse = bind_register_program(
        fixed_add(width, fraction, adjoint=True), words, clean=work
    )
    assert forward.outputs == inverse.outputs == words
    slots = tuple(range(width - 1))
    assert_source(
        normalized_feedback(forward.circuit.instructions),
        normalized_feedback(add_ir(*words, work, slots)),
    )
    assert_source(
        normalized_feedback(inverse.circuit.instructions),
        normalized_feedback(add_adjoint_ir(*words, work, slots)),
    )
    counts = Counter(gate.name for gate in forward.circuit.instructions)
    assert counts["t"] + counts["tdg"] == 4 * (width - 1)
    assert counts["measure"] == counts["reset"] == width - 1
    size = 3 * width
    before = torch.zeros(1 << size, dtype=torch.complex128)
    after = torch.zeros_like(before)
    for a, b, spectator in product(range(1 << width), range(1 << width), (0, 1)):
        value = torch.tensor(complex(a + 1, b - 2 * spectator), dtype=torch.complex128)
        before[(a << (2 * width)) | (b << width) | spectator] = value
        after[(a << (2 * width)) | (((a + b) % (1 << width)) << width) | spectator] = (
            value
        )
    norm = before.norm()
    before /= norm
    after /= norm
    histories = tuple(product((0, 1), repeat=width - 1))
    for first, second in product(histories, repeat=2):
        state = forced_branch(before, forward.circuit.instructions, first)
        torch.testing.assert_close(
            state, after / 2 ** ((width - 1) / 2), atol=3e-12, rtol=0
        )
        restored = forced_branch(state, inverse.circuit.instructions, second)
        torch.testing.assert_close(
            restored, before / 2 ** (width - 1), atol=3e-12, rtol=0
        )
    if width == 3:
        circuit = DynamicCircuit(size, inputs=before, dtype=torch.complex128)
        circuit.append(FieldOperation(size, forward.circuit.instructions))
        circuit.append(FieldOperation(size, inverse.circuit.instructions))
        actual = run_dynamic(circuit, shots=4, seed=41, strategy="trajectory")
        torch.testing.assert_close(
            actual.final_states.reshape(4, -1), before.expand(4, -1), atol=3e-12, rtol=0
        )


@pytest.mark.parametrize("width,fraction", ((2, 2), (3, 1), (4, -2)))
def test_owned_output_is_created_and_matching_left_image_consumed(
    width: int, fraction: int
) -> None:
    dtype = fixed_point_register_type(width, fraction)
    build = ComponentProgramBuilder(
        Signature((Register("result", dtype, side=RegisterSide.RIGHT),))
    )
    allocated = build.allocate(dtype=dtype)
    forward = bind_register_program(
        build.finalize({"result": allocated}),
        (),
        clean=tuple(range(width)),
        allow_workspace_outputs=True,
    )
    assert forward.outputs == (tuple(range(width)),)
    consume = ComponentProgramBuilder(
        Signature((Register("result", dtype, side=RegisterSide.LEFT),))
    )
    word = consume.inputs["result"]
    assert isinstance(word, RegisterBinding)
    consume.free(consume.join(consume.split(word), dtype))
    inverse = bind_register_program(
        consume.finalize({}),
        forward.outputs,
        input_ownership=tuple((wire, False) for wire in range(width)),
    )
    assert inverse.outputs == ()
    assert forward.circuit.instructions == inverse.circuit.instructions == ()
    assert [event.kind for event in forward.events] == ["allocate"]
    assert [event.kind for event in inverse.events] == ["free"]


@pytest.mark.parametrize("width,fraction", ((2, 1), (3, 3), (4, -2)))
@pytest.mark.parametrize("activation", (0, 1))
def test_wider_control_retains_original_refusal(
    width: int, fraction: int, activation: int
) -> None:
    from qualtran._infra.controlled import CtrlSpec
    from qualtran._infra.data_types import QFxp
    from qualtran.bloqs.mcmt.ctrl_spec_and import CtrlSpecAnd

    with pytest.raises(NotImplementedError, match="only supports integer types"):
        CtrlSpecAnd(CtrlSpec(QFxp(width, fraction), activation))
    with pytest.raises(ValueError, match="single-bit"):
        controlled_x_program(
            (Register("ctrl", fixed_point_register_type(width, fraction)),),
            (ControlRegister(width, (activation,)),),
        )
