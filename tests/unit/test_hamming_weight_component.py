"""Stored Hamming calls preserve full streams, output ownership and erasure."""

from collections import Counter
from itertools import product

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler._component_trace.program_call_graph import (
    program_component_call_graph,
)
from flagquantum.compiler.registers.bits.hamming_weight import (
    hamming_weight_ir,
    hamming_weight_requirements,
)
from flagquantum.compiler.registers.bits.hamming_weight_component import (
    hamming_weight_program,
)
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import verify_program_record
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", (1, 2, 3, 4, 5, 7, 8, 9, 16))
@pytest.mark.parametrize("adjoint", (False, True))
def test_stored_source_order_operands_feedback_aliases_and_actual_resources(
    width: int, adjoint: bool
) -> None:
    from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    junk_width, out_width = hamming_weight_requirements(width)
    total = width + junk_width + out_width
    x = tuple(reversed(range(width)))
    junk = tuple(range(width, width + junk_width))
    out = tuple(range(width + junk_width, total))
    record = hamming_weight_program(width, adjoint=adjoint)
    assert verify_program_record(record) == ()
    inputs = (x, junk, out) if adjoint else (x,)
    clean = () if adjoint else tuple(reversed((*junk, *out)))
    ownership = tuple((wire, False) for wire in (*junk, *out)) if adjoint else ()
    bound = bind_register_program(
        record,
        inputs,
        clean=clean,
        allow_workspace_outputs=not adjoint,
        input_ownership=ownership,
    )
    words = (x, junk, out) if adjoint else bound.outputs
    slots = {
        gate.wires[0]: gate.metadata["classical_bit"]
        for gate in bound.circuit.instructions
        if gate.name == "measure"
    }
    expected = hamming_weight_ir(
        *words,
        adjoint=adjoint,
        classical_bits=tuple(slots[wire] for wire in words[1]) if adjoint else (),
    )
    assert_source(bound.circuit.instructions, expected)
    assert bound.outputs == ((x,) if adjoint else words)
    source = HammingWeightCompute(width)
    original = reference_arithmetic_gates(
        source,
        dict(zip(("x", "junk", "out"), words, strict=True)),
        (),
        {},
        cirq_style=True,
    )
    if not adjoint:
        assert_source(bound.circuit.instructions, original)
    elif width == 1:
        with pytest.raises(KeyError):
            source.adjoint().decompose_bloq()
        assert bound.circuit.instructions == original
    else:
        original_inverse = reference_arithmetic_gates(
            source.adjoint(),
            dict(zip(("x", "junk", "out"), words, strict=True)),
            (),
            slots,
            expected_outputs={"x": x},
        )
        assert_source(bound.circuit.instructions, original_inverse)
    flat, _ = expand_component_calls(record)
    assert_source(
        bind_register_program(
            flat,
            inputs,
            clean=clean,
            allow_workspace_outputs=not adjoint,
            input_ownership=ownership,
        ).circuit.instructions,
        bound.circuit.instructions,
    )
    assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit
    calls = program_component_call_graph(
        record, keep={"hamming_cx", "hamming_and"}.__contains__
    )
    assert calls.leaf_counts.get("hamming_and", 0) == junk_width
    assert calls.leaf_counts["hamming_cx"] == 5 * junk_width + width.bit_count()
    counts = Counter(gate.name for gate in bound.circuit.instructions)
    model = get_cost_value(source.adjoint() if adjoint else source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == (0 if adjoint else 4 * junk_width)
    assert counts["measure"] == counts["reset"] == (junk_width if adjoint else 0)
    assert model.measurement == counts["measure"]
    assert model.total_t_count() == counts["t"] + counts["tdg"]
    assert "any" not in counts


@pytest.mark.parametrize("width", (1, 2, 3, 4))
def test_all_complex_branches_and_normal_append_restore_every_output(
    width: int,
) -> None:
    junk_width, out_width = hamming_weight_requirements(width)
    total = width + junk_width + out_width + 1
    forward = hamming_weight_program(width)
    work = tuple(range(width + 1, total))
    bound = bind_register_program(
        forward, (tuple(range(width)),), clean=work, allow_workspace_outputs=True
    )
    backward = bind_register_program(
        hamming_weight_program(width, adjoint=True),
        bound.outputs,
        input_ownership=tuple(
            (wire, False) for word in bound.outputs[1:] for wire in word
        ),
    )
    raw = torch.arange(1, (1 << (width + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 3 - 7 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    initial[:: 1 << len(work)] = amplitudes
    circuit = Circuit(total, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(total, bound.circuit.instructions))
    computed = circuit.state().flatten()
    reference = forced_branch(initial, hamming_weight_ir(*bound.outputs), ())
    torch.testing.assert_close(computed, reference, rtol=0, atol=3e-12)
    for history in product((0, 1), repeat=junk_width):
        restored = forced_branch(computed, backward.circuit.instructions, history)
        torch.testing.assert_close(
            restored, initial / 2 ** (junk_width / 2), rtol=0, atol=3e-12
        )
        assert torch.count_nonzero(restored.reshape(-1, 1 << len(work))[:, 1:]) == 0
    assert backward.outputs == (tuple(range(width)),)
    assert CircuitIR.from_json(circuit.to_ir().to_json()) == circuit.to_ir()


@pytest.mark.parametrize("width", (0, -1, True, 1.5))
def test_invalid_widths_retain_existing_validation(width: int) -> None:
    with pytest.raises(ValueError, match="positive concrete width"):
        hamming_weight_program(width)


@pytest.mark.parametrize("adjoint", (1,))
def test_direction_requires_a_boolean(adjoint: bool) -> None:
    with pytest.raises(ValueError, match="adjoint must be boolean"):
        hamming_weight_program(3, adjoint=adjoint)
