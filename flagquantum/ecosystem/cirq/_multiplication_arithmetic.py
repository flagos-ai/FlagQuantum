"""Bind concrete external multiplication declarations to owned stored Programs."""

from collections.abc import Iterator
from typing import Any

from ...compiler.registers.arithmetic.integer_product_component import (
    integer_product_program,
)
from ...compiler.registers.arithmetic.product_accumulation_component import (
    product_accumulation_program,
)
from ...compiler.registers.arithmetic.sum_of_squares_component import (
    sum_of_squares_program,
)
from ...compiler.registers.fixed_point.fixed_point_product_component import (
    fixed_point_product_program,
)
from ...compiler.registers.fixed_point.fixed_point_scaling_component import (
    scale_integer_by_real_program,
)
from ...compiler.registers.fixed_point.fixed_point_square_component import (
    fixed_point_square_program,
)
from ...compiler.registers.fixed_point.reciprocal_component import reciprocal_program
from ...compiler.registers.primitives.circuit_binding import restore_output_order_ir
from ...compiler.registers.program_calls import _expand
from ...compiler.registers.register_binding import bind_register_program
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program import ProgramRecord
from ...core.ir.program.register_verifier import executable_register_width
from ...core.ir.program.registers import RegisterSide
from ...core.ir.program.signatures import Signature
from ._register_export import export_register_cirq
from .conversion import _cirq


def multiplication_program(source: Any, *, adjoint: bool) -> ProgramRecord:
    """Translate only a frozen declaration's parameters at the SDK boundary.

    The root supplies a signature and a model. Execution belongs to the native
    paper construction, whose costs and approximation limits are independent.
    """
    from qualtran.bloqs.arithmetic.multiplication import (
        InvertRealNumber,
        MultiplyTwoReals,
        PlusEqualProduct,
        Product,
        ScaleIntByReal,
        SquareRealNumber,
        SumOfSquares,
    )

    if type(source) is PlusEqualProduct:
        return product_accumulation_program(
            source.a_bitsize,
            source.b_bitsize,
            source.result_bitsize,
            adjoint=source.is_adjoint != adjoint,
        )
    if type(source) is Product:
        return integer_product_program(
            source.a_bitsize, source.b_bitsize, adjoint=adjoint
        )
    if type(source) is SumOfSquares:
        return sum_of_squares_program(source.bitsize, source.k, adjoint=adjoint)
    if type(source) is ScaleIntByReal:
        return scale_integer_by_real_program(
            source.r_bitsize, source.i_bitsize, adjoint=adjoint
        )
    if type(source) is MultiplyTwoReals:
        return fixed_point_product_program(source.bitsize, adjoint=adjoint)
    if type(source) is SquareRealNumber:
        return fixed_point_square_program(source.bitsize, adjoint=adjoint)
    if type(source) is InvertRealNumber:
        return reciprocal_program(source.bitsize, source.num_frac)
    raise ValueError("Unsupported multiplication declaration")


def multiplication_operations(
    record: ProgramRecord,
    operands: tuple[Any, ...],
    manager: Any,
    key_prefix: str,
    tag: object,
) -> Iterator[Any]:
    """Export owned arithmetic, restoring actual output aliases before return."""
    functions = {f.name: f for f in record.functions}
    entry = functions["main"]
    assert entry.registers is not None
    signature = Signature(entry.registers)
    logical = signature.n_qubits()
    if type(logical) is not int or logical != len(operands):
        raise ValueError("Multiplication operands do not match concrete signature")
    operations, _ = _expand(entry.blocks[0], functions)
    # Sum allocation lifetimes only to size the synthetic binding pool. The SDK
    # receives exactly the scratch wires actually used after physical binding.
    capacity = sum(
        executable_register_width(op.register_type)
        for op in operations
        if op.opcode == "allocate_register" and op.register_type is not None
    )
    inputs: list[tuple[int, ...]] = []
    output_slots: list[int] = []
    consumed: list[int] = []
    layouts: dict[str, tuple[int, ...]] = {}
    offset = 0
    for reg in signature:
        width = reg.dtype.quantum_bits
        assert type(width) is int
        wires = tuple(range(offset, offset + reg.total_bits()))
        layouts[reg.name] = wires
        if reg.side & RegisterSide.LEFT:
            inputs.extend(wires[i : i + width] for i in range(0, len(wires), width))
        if reg.side == RegisterSide.RIGHT:
            output_slots.extend(wires)
        elif reg.side == RegisterSide.LEFT:
            consumed.extend(wires)
        offset += len(wires)
    bound = bind_register_program(
        record,
        tuple(inputs),
        clean=(*output_slots, *range(logical, logical + capacity)),
        allow_workspace_outputs=True,
        input_ownership=tuple((q, False) for q in consumed),
    )
    returned = iter(bound.outputs)
    output_map: dict[int, int] = {}
    for reg in signature.rights():
        actual = tuple(q for _ in reg.all_idxs() for q in next(returned))
        output_map.update(zip(layouts[reg.name], actual, strict=True))
    scratch_indices = sorted(
        {q for event in bound.events for q in event.wires if q >= logical}
    )
    physical = (*range(logical), *scratch_indices)
    unreturned = iter(q for q in physical if q not in output_map.values())
    canonical = tuple(
        output_map[q] if q in output_map else next(unreturned) for q in range(logical)
    )
    order = (*canonical, *(q for q in physical if q not in canonical))
    swaps = restore_output_order_ir(order)
    count = len(scratch_indices)
    scratch = tuple(manager.qalloc(count))
    try:
        # First-fit clean binding uses a contiguous synthetic scratch range.
        assert scratch_indices == list(range(logical, logical + count))
        gates = (*bound.circuit.instructions, *swaps)
        used = {q for g in gates for q in g.wires}
        padding = tuple(Instruction("i", (q,)) for q in physical if q not in used)
        physical_signature = Signature.build(logical=logical, work=count)
        physical_ports = {"logical": tuple(range(logical))}
        physical_arrays = {"logical": operands}
        if count:
            physical_ports["work"] = tuple(range(logical, logical + count))
            physical_arrays["work"] = scratch
        exported, _ = export_register_cirq(
            CircuitIR(logical + count, (*gates, *padding)),
            physical_signature,
            physical_ports,
            input_quregs=physical_arrays,
            feedback_key_prefix=key_prefix,
        )
        cirq = _cirq()
        yield from (
            op.with_tags(tag)
            for op in exported.circuit.all_operations()
            if op.gate != cirq.I
        )
    finally:
        manager.qfree(scratch)
