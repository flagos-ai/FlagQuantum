"""Complete bookkeeping catalog wrappers preserve original calls and atomic leaves."""

from collections import Counter
from dataclasses import replace
from typing import Any

import pytest

from examples.algorithms import bookkeeping_ui_export as native
from examples.algorithms.bookkeeping_decomposition import bookkeeping_child
from tests.unit.test_bookkeeping_ui_export import originals as originals
from tests.unit.test_field_product_decomposition import source_walk

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
CONCRETE = ("auto_partition", "auto_partition_unused", "always_and")


@pytest.mark.parametrize("identifier", CONCRETE)
def test_complete_wrapper_calls_and_fresh_source_drawings(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    rows = source_walk(identifier, originals[1][identifier], catalog=native)
    counts = Counter(row["disposition"] for row in rows)
    assert counts["exact_graph"] > 0
    assert counts["original_ignored_adjoint_defect"] == 0


@pytest.mark.parametrize(
    "identifier", tuple(v for v in native.INSTANCES if v not in CONCRETE)
)
def test_original_atomic_catalog_refusal(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    from qualtran import DecomposeTypeError

    with pytest.raises(DecomposeTypeError):
        originals[1][identifier].decompose_bloq()
    assert native.decomposition(identifier) is None


def test_controlled_child_label_rejects_changed_primitive_operands() -> None:
    record = native.decomposition("auto_partition")
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
        if bookkeeping_child(record, f.name, "auto_partition") is not None:
            assert bookkeeping_child(changed, f.name, "auto_partition") is None
            rejected += 1
    assert rejected > 0


@pytest.mark.parametrize("identifier", CONCRETE)
def test_selected_catalog_program_normal_append_full_complex_isometry(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    from math import prod

    import torch

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum import Circuit
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.core.ir.program.registers import RegisterSide

    record = native.decomposition(identifier)
    assert record is not None
    root = record.functions[0]
    inputs = []
    position = 0
    for port in root.registers or ():
        if port.side & RegisterSide.LEFT:
            assert type(port.bitsize) is int
            for _ in range(prod(port.shape)):
                inputs.append(tuple(range(position, position + port.bitsize)))
                position += port.bitsize
    bound = bind_register_program(
        record,
        tuple(inputs),
        entry=root.name,
        clean=(position,) if identifier == "always_and" else (),
        allow_workspace_outputs=True,
    )
    width = bound.circuit.n_wires
    amplitudes = torch.arange(1, (1 << position) + 1, dtype=torch.float64)
    amplitudes = torch.complex(amplitudes, amplitudes.flip(0))
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << width, dtype=torch.complex128)
    initial[:: 1 << (width - position)] = amplitudes
    expected = torch.zeros_like(initial)
    source = originals[1][identifier]
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    logical = matrix @ amplitudes
    output_wires = tuple(wire for port in bound.outputs for wire in port)
    for basis, amplitude in enumerate(logical):
        destination = 0
        for bit, wire in enumerate(output_wires):
            destination |= ((basis >> (len(output_wires) - bit - 1)) & 1) << (
                width - wire - 1
            )
        expected[destination] = amplitude
    actual = (
        Circuit(width, inputs=initial, dtype=initial.dtype)
        .append(FieldOperation(width, bound.circuit.instructions))
        .state()
        .flatten()
    )
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=0)
