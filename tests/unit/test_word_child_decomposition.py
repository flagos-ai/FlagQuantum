"""Complete bitwise source dependencies retain exact fresh source drawings."""

from dataclasses import replace
from typing import Any, Literal

import pytest

from examples.algorithms import arithmetic_ui_export as native
from examples.algorithms.word_child_decomposition import word_child_decomposition
from flagquantum.compiler.registers.bits.bitwise_component import bitwise_word_program
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    signed_register_type,
    unsigned_register_type,
)
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_score_data import register_score_data
from tests.unit.test_arithmetic_ui_export import originals as originals
from tests.unit.test_basic_gate_child_decomposition import _label, _source_score
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_register_gate_ui_export import _canonical_score

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("operation", ("xor", "xor_constant", "bitwise_not"))
@pytest.mark.parametrize("width", (1, 2, 3, 8))
@pytest.mark.parametrize("encoding", ("opaque", "uint", "signed"))
def test_full_bitwise_file_recursive_graph_and_fresh_drawings(
    operation: Literal["xor", "xor_constant", "bitwise_not"],
    width: int,
    encoding: str,
) -> None:
    from qualtran import QAny, QInt, QUInt
    from qualtran.bloqs.arithmetic.bitwise import BitwiseNot, Xor, XorK

    from flagquantum.core.ir.program.signatures import Signature

    dtype = {
        "opaque": opaque_register_type,
        "uint": unsigned_register_type,
        "signed": signed_register_type,
    }[encoding](width)
    source_dtype = {"opaque": QAny, "uint": QUInt, "signed": QInt}[encoding](width)
    constant = -1 if encoding == "signed" else (1 << width) - 1
    source = {
        "xor": lambda: Xor(source_dtype),
        "xor_constant": lambda: XorK(source_dtype, constant),
        "bitwise_not": lambda: BitwiseNot(source_dtype),
    }[operation]()
    record = bitwise_word_program(
        dtype,
        operation,
        pattern=constant % (1 << width) if operation == "xor_constant" else None,
    )
    functions = {function.name: function for function in record.functions}
    pending = [(record.functions[0].name, source)]
    visited: dict[str, Any] = {}
    while pending:
        name, original = pending.pop(0)
        if name in visited:
            assert visited[name] == original
            continue
        visited[name] = original
        function = functions[name]
        try:
            composite = original.decompose_bloq()
        except Exception as error:
            assert type(error).__name__ in {
                "DecomposeTypeError",
                "DecomposeNotImplementedError",
            }
            continue
        selected = word_child_decomposition(record, name, {"k": constant})
        assert selected is not None, (operation, encoding, width, name)
        label, template, symbols = selected
        assert label == _label(original)
        assert _native_trace(record, name) == _source_trace(
            _source(Signature(function.registers or ())), composite
        )
        actual = register_score_data(
            component_register_score(template, template.functions[0].name),
            register_wire_symbol=symbols,
        )
        assert _canonical_score(dict(actual)) == _canonical_score(
            _source_score(composite)
        )
        children = {
            instance.i: instance.bloq for instance, _, _ in composite.iter_bloqsoqs()
        }
        pending.extend(
            (operation.callee or "", children[index])
            for index, operation in enumerate(function.blocks[0].operations)
        )


def test_changed_repeated_gate_does_not_reuse_complement_or_on_each() -> None:
    record = bitwise_word_program(unsigned_register_type(3), "bitwise_not")
    changed = replace(
        record,
        functions=tuple(
            replace(
                function,
                blocks=tuple(
                    replace(
                        block,
                        operations=tuple(
                            (
                                replace(operation, opcode="z")
                                if operation.opcode == "x"
                                else operation
                            )
                            for operation in block.operations
                        ),
                    )
                    for block in function.blocks
                ),
            )
            for function in record.functions
        ),
    )
    for function in record.functions:
        if (
            tuple(r.name for r in function.registers or ()) in {("x",), ("q",)}
            and function.blocks[0].operations[0].opcode == "call"
        ):
            assert word_child_decomposition(record, function.name, {}) is not None
            assert word_child_decomposition(changed, function.name, {}) is None


def test_symbolic_complement_keeps_its_original_refused_child(
    originals: tuple[Any, dict[str, Any]],
) -> None:
    from qualtran import DecomposeTypeError

    record = native.decomposition("bitwise_not_symb")
    assert record is not None
    child = record.functions[1]
    assert not child.blocks
    assert word_child_decomposition(record, child.name, {}) is None
    source = originals[1]["bitwise_not_symb"].decompose_bloq()
    (instance,) = tuple(source.bloq_instances)
    with pytest.raises(DecomposeTypeError):
        instance.bloq.decompose_bloq()
