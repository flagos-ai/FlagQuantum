"""Complete Hamming phase calls retain source graphs, amplitudes and cleanup."""

from collections import Counter
from dataclasses import replace
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms import rotations_ui_export as native
from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.hamming_phasing_catalog import HammingPhasingExample
from examples.algorithms.hamming_phasing_decomposition import hamming_phase_child
from flagquantum.compiler.registers.component_graph import copy_component_program
from flagquantum.compiler.registers.phase.hamming_phasing_component import (
    hamming_phasing_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_field_product_decomposition import source_walk
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_phase_gradient_source_binding import through_operators
from tests.unit.test_rotations_ui_export import originals as originals

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
FACTORIES = ("hamming_weight_phasing", "hamming_weight_phasing_via_phase_gradient")


@pytest.mark.parametrize("identifier,exact", tuple(zip(FACTORIES, (3, 4), strict=True)))
def test_complete_file_recursive_calls_drawings_and_original_dtype_refusal(
    identifier: str, exact: int, originals: tuple[Any, dict[str, Any]]
) -> None:
    rows = source_walk(identifier, originals[1][identifier], catalog=native)
    assert sum(row["disposition"] == "exact_graph" for row in rows) == exact
    assert all(row.get("drawing", "source_exact") == "source_exact" for row in rows)
    assert sum(row["disposition"] == "source_dtype_refusal" for row in rows) == int(
        identifier == FACTORIES[1]
    )


@pytest.mark.parametrize("width,gradient", tuple(product((2, 3), (False, True))))
def test_full_original_operators_all_histories_and_noncontiguous_spectator(
    width: int, gradient: bool
) -> None:
    from qualtran.bloqs.rotations.hamming_weight_phasing import (
        HammingWeightPhasing,
        HammingWeightPhasingViaPhaseGradient,
    )

    exponent, epsilon = (1.0 if gradient else 0.375), 0.2
    source = (
        HammingWeightPhasingViaPhaseGradient if gradient else HammingWeightPhasing
    )(width, exponent, epsilon)
    expected, expected_inverse = through_operators(import_register_emitter(source))
    example = HammingPhasingExample(width, exponent, epsilon, via_gradient=gradient)
    logical = width + example.gradient_width
    total = example.qubit_count + 1
    work_size = total - logical - 1
    record = hamming_phasing_program(width, exponent, epsilon, via_gradient=gradient)
    inverse = hamming_phasing_program(width, -exponent, epsilon, via_gradient=gradient)
    contiguous = tuple(range(logical))
    inputs = (contiguous[:width], contiguous[width:]) if gradient else (contiguous,)
    clean = tuple(range(logical, total - 1))
    forward = bind_register_program(record, inputs, clean=clean)
    backward = bind_register_program(inverse, inputs, clean=clean)
    for bound, matrix in ((forward, expected), (backward, expected_inverse)):
        torch.testing.assert_close(
            logical_operator(bound.circuit, logical), matrix, atol=1e-11, rtol=0
        )
    wires = tuple(reversed(range(logical)))
    inputs = (wires[:width], wires[width:]) if gradient else (wires,)
    clean = tuple(range(logical + 1, total))
    forward = bind_register_program(record, inputs, clean=clean)
    backward = bind_register_program(inverse, inputs, clean=clean)
    assert forward.outputs == backward.outputs == inputs
    assert copy_component_program(record) == record
    raw = torch.arange(1, 2 ** (logical + 1) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, raw.flip(0))
    amplitudes /= amplitudes.norm()
    order = (*wires, logical, *clean)
    restore = tuple(order.index(wire) for wire in range(total))
    initial_view = torch.zeros((2**logical, 2, 2**work_size), dtype=torch.complex128)
    initial_view[:, :, 0] = amplitudes.reshape(2**logical, 2)
    initial = initial_view.reshape((2,) * total).permute(restore).flatten()
    expected_view = torch.zeros_like(initial_view)
    expected_view[:, :, 0] = expected @ amplitudes.reshape(2**logical, 2)
    target = expected_view.reshape((2,) * total).permute(restore).flatten()
    gates, reverse = forward.circuit.instructions, backward.circuit.instructions
    count = sum(g.name == "measure" for g in gates)
    reverse_count = sum(g.name == "measure" for g in reverse)
    for history in product((0, 1), repeat=count):
        branch = forced_branch(initial, gates, history)
        torch.testing.assert_close(
            branch, target / 2 ** (count / 2), atol=1e-11, rtol=0
        )
        work = (
            branch.reshape((2,) * total).permute(order).reshape(-1, 2**work_size)[:, 1:]
        )
        torch.testing.assert_close(work, torch.zeros_like(work), atol=1e-11, rtol=0)
        for second in product((0, 1), repeat=reverse_count):
            recovered = forced_branch(branch, reverse, second)
            torch.testing.assert_close(
                recovered,
                initial / 2 ** ((count + reverse_count) / 2),
                atol=1e-11,
                rtol=0,
            )
    circuit = DynamicCircuit(total, inputs=initial, dtype=initial.dtype)
    circuit.append(FieldOperation(total, gates))
    result = run_dynamic(circuit, shots=4, seed=71, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(4, -1), target.expand(4, -1), atol=1e-11, rtol=0
    )
    assert CircuitIR.from_json(forward.circuit.to_json()) == forward.circuit


def test_original_four_bit_program_primitive_inventories() -> None:
    for identifier, expected in zip(
        FACTORIES, ((12, 50, 3, 3), (1108, 2592, 0, 277)), strict=True
    ):
        record = native.decomposition(identifier)
        assert record is not None
        example = HammingPhasingExample(via_gradient=identifier == FACTORIES[1])
        logical = 4 + example.gradient_width
        inputs = (
            (tuple(range(4)), tuple(range(4, logical)))
            if example.via_gradient
            else (tuple(range(4)),)
        )
        bound = bind_register_program(
            record, inputs, clean=tuple(range(logical, example.qubit_count))
        )
        counts = Counter(g.name for g in bound.circuit.instructions)
        assert (
            counts["t"] + counts["tdg"],
            counts["cx"],
            counts["phase"],
            counts["measure"],
        ) == expected


@pytest.mark.parametrize("identifier", FACTORIES)
def test_qualified_count_and_qvr_children_reject_changed_definitions(
    identifier: str,
) -> None:
    record = native.decomposition(identifier)
    assert record is not None
    functions = []
    for function in record.functions:
        blocks = []
        for block in function.blocks:
            operations = []
            for operation in block.operations:
                if operation.opcode == "cx":
                    operation = replace(operation, operands=operation.operands[::-1])
                elif operation.opcode == "s":
                    operation = replace(operation, opcode="z")
                operations.append(operation)
            blocks.append(replace(block, operations=tuple(operations)))
        functions.append(replace(function, blocks=tuple(blocks)))
    changed = replace(record, functions=tuple(functions))
    checked = 0
    for function in record.functions:
        presentation = hamming_phase_child(
            record, function.name, native.attributes(identifier)
        )
        if presentation is not None:
            assert (
                hamming_phase_child(
                    changed, function.name, native.attributes(identifier)
                )
                is None
            )
            checked += 1
    assert checked >= 2


def test_direct_symbolic_phasing_binds_to_accepted_numeric_body() -> None:
    from flagquantum import Circuit
    from flagquantum.core.parameters import Parameter

    symbolic = hamming_phasing_program(3, Parameter("theta"), 0.2)
    bound = bind_register_program(symbolic, ((0, 1, 2),), clean=(3, 4, 5))
    actual = (
        Circuit(6)
        .append(FieldOperation(6, bound.circuit.instructions))
        .bind_parameters({"theta": 0.375})
        .to_ir()
    )
    expected = bind_register_program(
        hamming_phasing_program(3, 0.375, 0.2), ((0, 1, 2),), clean=(3, 4, 5)
    )
    assert actual.instructions == expected.circuit.instructions
