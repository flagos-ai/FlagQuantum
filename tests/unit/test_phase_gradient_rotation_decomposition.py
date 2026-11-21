"""Complete gradient rotation Programs preserve source calls, phase and ownership."""

from collections import Counter
from dataclasses import replace
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms import rotations_ui_export as native
from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.phase_gradient_rotation_decomposition import (
    phase_rotation_child,
)
from flagquantum.compiler.registers.component_graph import copy_component_program
from flagquantum.compiler.registers.phase.rz_phase_gradient import (
    rz_via_phase_gradient_ir,
)
from flagquantum.compiler.registers.phase.rz_phase_gradient_component import (
    rz_via_phase_gradient_program,
)
from flagquantum.compiler.registers.phase.zpow_phase_gradient import (
    zpow_via_phase_gradient_ir,
)
from flagquantum.compiler.registers.phase.zpow_phase_gradient_component import (
    zpow_via_phase_gradient_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.register_types import fixed_point_register_type
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_field_product_decomposition import source_walk
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_rotations_ui_export import originals as originals

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "identifier,graphs,refusals",
    (
        ("rz_via_phase_gradient", 5, 1),
        ("zpow_const_via_phase_grad", 2, 1),
        ("zpow_const_via_phase_grad_symb_prec", 1, 1),
    ),
)
def test_complete_recursive_source_graphs_and_original_drawing_refusals(
    identifier: str, graphs: int, refusals: int, originals: tuple[Any, dict[str, Any]]
) -> None:
    rows = source_walk(identifier, originals[1][identifier], catalog=native)
    assert sum(r["disposition"] == "exact_graph" for r in rows) == graphs
    assert (
        sum(r.get("drawing") == "original_controlled_title_refusal" for r in rows)
        == refusals
    )
    assert all(r["disposition"] != "original_ignored_adjoint_defect" for r in rows)


def test_symbolic_precision_graph_and_symbolic_angle_refusal(
    originals: tuple[Any, dict[str, Any]],
) -> None:
    from qualtran import DecomposeTypeError

    record = native.decomposition("zpow_const_via_phase_grad_symb_prec")
    assert record is not None and copy_component_program(record) == record
    assert len(record.functions[0].blocks[0].operations) == 5
    leaves = [f for f in record.functions if not f.blocks]
    assert {tuple(r.name for r in f.registers or ()) for f in leaves} == {
        ("ctrl", "x"),
        ("x", "phase_grad"),
    }
    with pytest.raises(ValueError):
        bind_register_program(
            record, ((0,), tuple(range(1, 41))), clean=tuple(range(41, 120))
        )
    with pytest.raises(DecomposeTypeError):
        originals[1]["zpow_const_via_phase_grad_symb_angle"].decompose_bloq()
    assert native.decomposition("zpow_const_via_phase_grad_symb_angle") is None


@pytest.mark.parametrize("constant,width", tuple(product((False, True), (2, 3))))
def test_source_full_complex_tensor_all_histories_append_work_zero_and_inverse(
    constant: bool, width: int
) -> None:
    from qualtran import QFxp
    from qualtran.bloqs.rotations.rz_via_phase_gradient import RzViaPhaseGradient
    from qualtran.bloqs.rotations.zpow_via_phase_gradient import (
        ZPowConstViaPhaseGradient,
    )

    dtype = fixed_point_register_type(width, width)
    record = (
        zpow_via_phase_gradient_program(0.75, width)
        if constant
        else rz_via_phase_gradient_program(dtype, dtype)
    )
    inverse_record = (
        zpow_via_phase_gradient_program(0.75, width, adjoint=True)
        if constant
        else rz_via_phase_gradient_program(dtype, dtype, adjoint=True)
    )
    logical = 1 + width if constant else 1 + 2 * width
    work_size = 2 * width - 1 if constant else width - 1
    total = logical + work_size + 1
    wires = tuple(reversed(range(logical)))
    inputs: tuple[tuple[int, ...], ...] = (
        ((wires[0],), wires[1:])
        if constant
        else ((wires[0],), wires[1 : width + 1], wires[width + 1 :])
    )
    clean = tuple(range(logical + 1, total))
    bound = bind_register_program(record, inputs, clean=clean)
    assert bound.outputs == inputs
    assert copy_component_program(record) == record
    gates = bound.circuit.instructions
    inverse = bind_register_program(
        inverse_record, inputs, clean=clean
    ).circuit.instructions
    slots = {
        g.wires[0]: g.metadata["classical_bit"] for g in gates if g.name == "measure"
    }
    if constant:
        gradient, angle, carry = inputs[1], clean[:width], clean[width:]
        reference = zpow_via_phase_gradient_ir(
            wires[0],
            gradient,
            angle,
            carry,
            tuple(slots[w] for w in carry),
            exponent=0.75,
        )
        source = ZPowConstViaPhaseGradient(0.75, width)
    else:
        carry = clean
        reference = rz_via_phase_gradient_ir(
            wires[0], inputs[1], inputs[2], carry, tuple(slots[w] for w in carry)
        )
        source = RzViaPhaseGradient(QFxp(width, width), QFxp(width, width))
    assert_source(gates, reference)
    values = torch.arange(1, 2 ** (logical + 1) + 1, dtype=torch.float64)
    amplitudes = torch.complex(values, values.flip(0))
    amplitudes /= amplitudes.norm()
    order = (*wires, logical, *clean)
    reverse = tuple(order.index(w) for w in range(total))
    initial_view = torch.zeros((2**logical, 2, 2**work_size), dtype=torch.complex128)
    initial_view[:, :, 0] = amplitudes.reshape(2**logical, 2)
    initial = initial_view.reshape((2,) * total).permute(reverse).flatten()
    matrix = torch.as_tensor(source.tensor_contract(), dtype=initial.dtype)
    expected_view = torch.zeros_like(initial_view)
    expected_view[:, :, 0] = matrix @ amplitudes.reshape(2**logical, 2)
    expected = expected_view.reshape((2,) * total).permute(reverse).flatten()
    measurements = sum(g.name == "measure" for g in gates)
    inverse_measurements = sum(g.name == "measure" for g in inverse)
    assert measurements == inverse_measurements == width - 1
    for history in product((0, 1), repeat=measurements):
        branch = forced_branch(initial, gates, history)
        scale = 2 ** (measurements / 2)
        torch.testing.assert_close(branch, expected / scale, atol=6e-12, rtol=0)
        work = (
            branch.reshape((2,) * total).permute(order).reshape(-1, 2**work_size)[:, 1:]
        )
        torch.testing.assert_close(work, torch.zeros_like(work), atol=6e-12, rtol=0)
        for second in product((0, 1), repeat=inverse_measurements):
            recovered = forced_branch(branch, inverse, second)
            torch.testing.assert_close(
                recovered,
                initial / (scale * 2 ** (inverse_measurements / 2)),
                atol=6e-12,
                rtol=0,
            )
    circuit = DynamicCircuit(total, inputs=initial, dtype=initial.dtype)
    circuit.append(FieldOperation(total, gates))
    result = run_dynamic(circuit, shots=8, seed=51, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(8, -1), expected.expand(8, -1), atol=6e-12, rtol=0
    )
    assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit


@pytest.mark.parametrize(
    "identifier", ("rz_via_phase_gradient", "zpow_const_via_phase_grad")
)
def test_qualified_children_reject_changed_operands(identifier: str) -> None:
    record = native.decomposition(identifier)
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
                                replace(op, operands=op.operands[::-1])
                                if op.opcode == "cx"
                                else op
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
    rejected = 0
    for f in record.functions:
        if (
            phase_rotation_child(record, f.name, native.attributes(identifier))
            is not None
        ):
            assert (
                phase_rotation_child(changed, f.name, native.attributes(identifier))
                is None
            )
            rejected += 1
    assert rejected > 0


def test_original_forty_bit_catalog_body_has_shared_exact_primitives(
    originals: tuple[Any, dict[str, Any]],
) -> None:
    from examples.algorithms.zpow_phase_gradient_catalog import GradientZPowExample
    from tests.unit.test_zpow_phase_gradient_file import assert_bound_stream

    record = native.decomposition("zpow_const_via_phase_grad")
    assert record is not None
    bound = bind_register_program(
        record, ((0,), tuple(range(1, 41))), clean=tuple(range(41, 120))
    )
    reference = assert_bound_stream(
        GradientZPowExample(3 / 8, 40), originals[1]["zpow_const_via_phase_grad"]
    )
    slots = {
        g.wires[0]: g.metadata["classical_bit"]
        for g in bound.circuit.instructions
        if g.name == "measure"
    }
    mapping = {index: slots[wire] for index, wire in enumerate(range(81, 120))}
    remapped: list[Instruction] = []
    for gate in reference:
        metadata = dict(gate.metadata)
        if "classical_bit" in metadata:
            metadata["classical_bit"] = mapping[metadata["classical_bit"]]
        if "conditions" in metadata:
            metadata["conditions"] = tuple(
                (mapping[bit], value) for bit, value in metadata["conditions"]
            )
        remapped.append(replace(gate, metadata=metadata))
    assert_source(bound.circuit.instructions, tuple(remapped))
    counts = Counter(g.name for g in bound.circuit.instructions)
    assert counts["t"] + counts["tdg"] == 156
    assert (counts["cx"], counts["measure"]) == (469, 39)
    assert bound.outputs == ((0,), tuple(range(1, 41)))
    assert sum(e.kind == "allocate" for e in bound.events) == 40
    assert sum(e.kind == "free" for e in bound.events) == 40
