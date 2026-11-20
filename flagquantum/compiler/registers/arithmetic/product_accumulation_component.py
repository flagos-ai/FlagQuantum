# SPDX-License-Identifier: Apache-2.0
# Qualtran v0.7.0 PlusEqualProduct supplies signature, value and model contracts.
# Source roots have no decomposition. Execution retains product_accumulation.py's
# live-column compute/add/erase construction.

"""Typed occupied-accumulator products over live compressed column storage."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.registers import RegisterType
from ....core.ir.program.signatures import Signature
from ..bits.bit_compression import weighted_column_components
from ..bits.column_steps_component import append_column_steps, column_bit_functions
from ..component_builder import ComponentProgramBuilder
from .addition_component import addition_component_program
from .integer_product import _product_terms, integer_product_requirements


def product_accumulation_program(
    left_width: int, right_width: int, result_width: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Retain a/b and add or subtract their product into occupied result.

    All words are unsigned and MSB first. Result retains wraparound modulo its
    width and must cover the full input product. The shared column network owns
    a fixed clean product/carry block. Its live parity word is used directly by
    the shared Add or exact Add inverse before column erasure releases that block.
    """
    _, work_width = integer_product_requirements(left_width, right_width)
    product_width = left_width + right_width
    if type(result_width) is not int or result_width < product_width:
        raise ValueError("Product accumulator must cover the full input product")
    if type(adjoint) is not bool:
        raise ValueError("Product accumulation direction requires a boolean")
    a_type, b_type = RegisterType("uint", left_width), RegisterType("uint", right_width)
    result_type = RegisterType("uint", result_width)
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(a=a_type, b=b_type, result=result_type),
        column_bit_functions(),
    )
    a, b, result = (builder.inputs[name] for name in ("a", "b", "result"))
    assert (
        isinstance(a, RegisterBinding)
        and isinstance(b, RegisterBinding)
        and isinstance(result, RegisterBinding)
    )
    bits = (*builder.split(a), *builder.split(b))
    work = builder.allocate(work_width)
    work_bits = builder.split(work)
    addresses = tuple(range(product_width, product_width + work_width))
    wires = dict(enumerate((*bits, *work_bits)))
    compute, word, erase = weighted_column_components(
        product_width,
        addresses,
        _product_terms(
            tuple(range(left_width)), tuple(range(left_width, product_width))
        ),
    )
    append_column_steps(builder, wires, compute)
    live_word = tuple(index for index in word if index is not None)
    # Product columns are contiguous with at most one leading zero. Retain the
    # original live-word interpretation before passing it to shared unsigned Add.
    assert word[-len(live_word) :] == live_word
    added = builder.call_program(
        addition_component_program(len(live_word), result_width, adjoint=adjoint),
        {
            "a": builder.join(
                [wires[i] for i in live_word], RegisterType("uint", len(live_word))
            ),
            "b": result,
        },
    )
    product_word, result = added["a"], added["b"]
    assert isinstance(product_word, RegisterBinding) and isinstance(
        result, RegisterBinding
    )
    wires.update(zip(live_word, builder.split(product_word), strict=True))
    append_column_steps(builder, wires, erase, inverse=True)
    builder.free(
        builder.join([wires[i] for i in addresses], RegisterType("opaque", work_width))
    )
    return builder.finalize(
        {
            "a": builder.join([wires[i] for i in range(left_width)], a_type),
            "b": builder.join(
                [wires[i] for i in range(left_width, product_width)], b_type
            ),
            "result": result,
        }
    )
