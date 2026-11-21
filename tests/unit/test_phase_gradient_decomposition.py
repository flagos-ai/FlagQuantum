"""Complete phase-gradient catalog graphs and their actual append path."""

from dataclasses import replace
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms import rotations_ui_export as native
from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.phase_gradient_decomposition import gradient_child
from flagquantum import Circuit
from flagquantum.compiler.registers.component_graph import copy_component_program
from flagquantum.compiler.registers.phase.phase_addition import (
    phase_add_ir,
    phase_add_workspace_size,
)
from flagquantum.compiler.registers.phase.phase_gradient import (
    phase_gradient_state_ir,
    phase_gradient_unitary_ir,
)
from flagquantum.compiler.registers.phase.phase_gradient_component import (
    phase_gradient_unitary_program,
)
from flagquantum.compiler.registers.phase.scaled_phase_addition import (
    scaled_phase_add_ir,
)
from flagquantum.compiler.registers.phase.scaled_phase_component import (
    phase_add_component_program,
    scaled_phase_add_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_field_product_decomposition import source_walk
from tests.unit.test_register_phase_gradient import source_gradient_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_rotations_ui_export import originals as originals
from tests.unit.test_scaled_phase_addition import ScaledAddArguments

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "identifier",
    ("phase_gradient_unitary", "phase_gradient_state", "add_scaled_val_into_phase_reg"),
)
def test_complete_catalog_source_graphs_and_fresh_drawings(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    rows = source_walk(identifier, originals[1][identifier], catalog=native)
    assert sum(row["disposition"] == "exact_graph" for row in rows) == (
        3 if identifier == "phase_gradient_state" else 1
    )


@pytest.mark.parametrize(
    "identifier", ("phase_gradient_unitary_symbolic", "add_into_phase_grad")
)
def test_original_decomposition_refusals(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    from qualtran import DecomposeNotImplementedError, DecomposeTypeError

    with pytest.raises((DecomposeTypeError, DecomposeNotImplementedError)):
        originals[1][identifier].decompose_bloq()
    assert native.decomposition(identifier) is None


@pytest.mark.parametrize(
    "identifier", ("phase_gradient_unitary", "phase_gradient_state")
)
def test_actual_catalog_append_source_stream_full_complex_spectator_and_output_ownership(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    record = native.decomposition(identifier)
    assert record is not None and copy_component_program(record) == record
    preparing = identifier == "phase_gradient_state"
    wires = (3, 1, 4, 0)
    bound = bind_register_program(
        record,
        () if preparing else (wires,),
        clean=wires if preparing else (),
        allow_workspace_outputs=preparing,
    )
    assert bound.outputs == (wires,)
    source = originals[1][identifier]
    gates, _ = source_gradient_gates(
        source, {} if preparing else {"phase_grad": wires}, {"phase_grad": wires}
    )
    assert gates == bound.circuit.instructions
    assert gates == (
        phase_gradient_state_ir(wires, exponent=-1)
        if preparing
        else phase_gradient_unitary_ir(wires)
    )
    initial = torch.zeros(32, dtype=torch.complex128)
    if preparing:
        initial[0] = 1
        initial[4] = torch.tensor(2j, dtype=initial.dtype)
    else:
        values = torch.arange(1, 33, dtype=torch.float64)
        initial = torch.complex(values, values.flip(0))
    initial /= initial.norm()
    order = (*wires, 2)
    inverse = tuple(order.index(i) for i in range(5))
    matrix = torch.as_tensor(source.tensor_contract(), dtype=initial.dtype)
    expected = (
        (
            matrix.reshape(16, 1)
            @ initial.reshape((2,) * 5).permute(order).reshape(16, 2)[:1]
        )
        if preparing
        else (matrix @ initial.reshape((2,) * 5).permute(order).reshape(16, 2))
    )
    expected = expected.reshape((2,) * 5).permute(inverse).flatten()
    actual = (
        Circuit(5, inputs=initial, dtype=initial.dtype)
        .append(FieldOperation(5, gates))
        .state()
        .flatten()
    )
    torch.testing.assert_close(actual, expected, atol=4e-12, rtol=0)
    reverse = (
        phase_gradient_state_ir(wires, exponent=-1, inverse=True)
        if preparing
        else bind_register_program(
            phase_gradient_unitary_program(4, -1), (wires,)
        ).circuit.instructions
    )
    recovered = forced_branch(actual, reverse, ())
    torch.testing.assert_close(recovered, initial, atol=4e-12, rtol=0)
    assert tuple(e.kind for e in bound.events) == (("allocate",) if preparing else ())
    assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit


@pytest.mark.parametrize("coefficient", (2, -2, 0.5, -0.5))
def test_scaled_component_all_measured_histories_full_complex_work_zero_and_recovery(
    coefficient: float,
) -> None:
    from qualtran import QFxp
    from qualtran.bloqs.rotations.phase_gradient import AddScaledValIntoPhaseReg

    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    args: ScaledAddArguments = dict(
        input_fraction_bits=2,
        coefficient=coefficient,
        coefficient_width=2,
        coefficient_fraction_bits=2,
    )
    record = scaled_phase_add_program(2, 2, 2, coefficient, 2, 2)
    assert copy_component_program(record) == record
    # Noncontiguous source/target, an entangled spectator and separate clean carry.
    x, phase = (3, 0), (2, 4)
    bound = bind_register_program(record, (x, phase), clean=(5,))
    assert bound.outputs == (x, phase)
    gates = bound.circuit.instructions
    assert gates == scaled_phase_add_ir(x, phase, (5,), (0,), **args)
    inverse = bind_register_program(
        scaled_phase_add_program(2, 2, 2, -coefficient, 2, 2), (x, phase), clean=(5,)
    ).circuit.instructions
    source = AddScaledValIntoPhaseReg(QFxp(2, 2), 2, coefficient, QFxp(2, 2))
    values = torch.arange(1, 33, dtype=torch.float64)
    amplitude = torch.complex(values, values.flip(0))
    amplitude /= amplitude.norm()
    initial = torch.zeros(64, dtype=torch.complex128)
    initial[::2] = amplitude
    expected = torch.zeros_like(initial)
    for index in range(64):
        bits = [(index >> (5 - i)) & 1 for i in range(6)]
        a, b = 2 * bits[x[0]] + bits[x[1]], 2 * bits[phase[0]] + bits[phase[1]]
        result = int(source.on_classical_vals(x=a, phase_grad=b)["phase_grad"])
        bits[phase[0]], bits[phase[1]] = result >> 1, result & 1
        target = sum(bit << (5 - i) for i, bit in enumerate(bits))
        expected[target] += initial[index]
    measurements = sum(g.name == "measure" for g in gates)
    inverse_measurements = sum(g.name == "measure" for g in inverse)
    for history in product((0, 1), repeat=measurements):
        branch = forced_branch(initial, gates, history)
        scale = 2 ** (measurements / 2)
        torch.testing.assert_close(branch, expected / scale, atol=5e-12, rtol=0)
        assert torch.count_nonzero(branch[1::2]) == 0
        for second in product((0, 1), repeat=inverse_measurements):
            recovered = forced_branch(branch, inverse, second)
            torch.testing.assert_close(
                recovered,
                initial / (scale * 2 ** (inverse_measurements / 2)),
                atol=5e-12,
                rtol=0,
            )
    dynamic = DynamicCircuit(6, inputs=initial, dtype=initial.dtype)
    dynamic.append(FieldOperation(6, gates))
    samples = run_dynamic(dynamic, shots=8, seed=31, strategy="trajectory")
    torch.testing.assert_close(
        samples.final_states.reshape(8, -1), expected.expand(8, -1), atol=5e-12, rtol=0
    )
    assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit


@pytest.mark.parametrize("shift,sign", tuple(product((0, 1, 4), (-1, 1))))
def test_atomic_native_addition_reuses_existing_exact_emitter(
    shift: int, sign: int
) -> None:
    size = phase_add_workspace_size(2, 3, right_shift=shift)
    work = tuple(range(5, 5 + size))
    bound = bind_register_program(
        phase_add_component_program(2, 3, right_shift=shift, sign=sign),
        ((1, 0), (4, 2, 3)),
        clean=work,
    )
    slots = {
        g.wires[0]: g.metadata["classical_bit"]
        for g in bound.circuit.instructions
        if g.name == "measure"
    }
    assert_source(
        bound.circuit.instructions,
        phase_add_ir(
            (1, 0),
            (4, 2, 3),
            work,
            tuple(slots[w] for w in work),
            right_shift=shift,
            sign=sign,
        ),
    )


def test_source_child_labels_reject_changed_primitive_body() -> None:
    record = native.decomposition("phase_gradient_state")
    assert record is not None
    changed = replace(
        record,
        functions=tuple(
            replace(
                f,
                blocks=tuple(
                    replace(
                        b,
                        operations=tuple(
                            (
                                replace(op, opcode="x")
                                if op.opcode == "h"
                                else (
                                    replace(op, literal=0.7)
                                    if op.opcode == "constant"
                                    else op
                                )
                            )
                            for op in b.operations
                        ),
                    )
                    for b in f.blocks
                ),
            )
            for f in record.functions
        ),
    )
    matches = 0
    for f in record.functions:
        if gradient_child(record, f.name, "phase_gradient_state") is not None:
            assert gradient_child(changed, f.name, "phase_gradient_state") is None
            matches += 1
    assert matches == 2
