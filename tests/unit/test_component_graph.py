"""Pinned named traversal and rebuilding through the existing Program graph."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from flagquantum.compiler.registers.component_builder import (
    BindingCollection,
    ComponentProgramBuilder,
)
from flagquantum.compiler.registers.component_graph import (
    copy_component_program,
    iter_component_calls,
)
from flagquantum.compiler.registers.register_programs import lower_register_program
from flagquantum.core.ir.program import (
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_register_rebinding_contract import SourceRebindings

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _source(signature: Signature) -> Any:
    factory: Any = SourceRebindings
    return factory(signature)


def _native_trace(record: ProgramRecord) -> list[object]:
    block = record.functions[0].blocks[0]
    producers = dict.fromkeys(block.arguments, "entry")
    producers.update(
        (value, str(i)) for i, op in enumerate(block.operations) for value in op.results
    )

    def collection(value: BindingCollection) -> object:
        array = np.asarray(value)
        return array.shape, tuple(
            (producers[b.value], b.register.name, b.index) for b in array.flat
        )

    return [
        (
            str(position),
            tuple((name, collection(value)) for name, value in inputs.items()),
            tuple(map(collection, outputs)),
        )
        for position, inputs, outputs in iter_component_calls(record)
    ]


def _source_trace(source: Any, composite: Any) -> list[object]:
    def collection(value: Any) -> object:
        array = np.asarray(value)
        identities = []
        for binding in array.flat:
            producer, name, index = source.identity(binding)
            identities.append((str(producer), name, index))
        return array.shape, tuple(identities)

    return [
        (
            str(instance.i),
            tuple((name, collection(value)) for name, value in inputs.items()),
            tuple(map(collection, outputs)),
        )
        for instance, inputs, outputs in composite.iter_bloqsoqs()
    ]


@pytest.mark.parametrize("width", [0, 1, 3])
@pytest.mark.parametrize("shape", [(), (2,), (2, 2), (0, 2)])
def test_named_traversal_and_copy_preserve_source_predecessor_metadata(
    width: int, shape: tuple[int, ...]
) -> None:
    signature = Signature([Register("word", RegisterType("uint", width), shape)])
    native = ComponentProgramBuilder(signature)
    source = _source(signature)
    results = []
    for builder in (native, source):
        words = np.asarray(builder.inputs["word"])
        returned = np.empty(shape, dtype=object)
        for index in signature.get_left("word").all_idxs():
            bits = builder.split(words[index])
            returned[index] = builder.join(bits[::-1], signature.get_left("word").dtype)
        results.append(returned if shape else returned[()])
    record = native.finalize({"word": results[0]})
    composite = source.builder.finalize(word=results[1])
    assert _native_trace(record) == _source_trace(source, composite)
    copied = copy_component_program(record)
    assert copied is not record and verify_program_record(copied) == ()
    assert lower_register_program(copied) == lower_register_program(record)
    assert all(
        a is b for a, b in zip(record.functions[1:], copied.functions[1:], strict=True)
    )
    if shape == (0, 2):
        # Source final_soqs drops the empty output key and copy fails finalizing.
        with pytest.raises(ValueError, match="word"):
            composite.copy()
    else:
        expected = composite.copy()
        assert _native_trace(copied) == _source_trace(source, expected)
        if width == 0:
            for graph in (expected, composite):
                with pytest.raises(ValueError, match="not enough values"):
                    graph.tensor_contract()
        elif width * np.asarray(results[0]).size <= 4:
            assert np.array_equal(
                expected.tensor_contract(), composite.tensor_contract()
            )


def test_copy_uses_visitation_order_and_preserves_module_and_entry_metadata() -> None:
    signature = Signature([Register("word", RegisterType("uint", 2))])
    builder = ComponentProgramBuilder(signature)
    word = builder.inputs["word"]
    assert isinstance(word, RegisterBinding)
    bits = builder.split(word)
    returned = builder.join(bits)
    record = builder.finalize({"word": returned})
    function = record.functions[0]
    block = function.blocks[0]
    # SSA identity is independent of the spelling of the input block name.
    values = {
        v: replace(v, scope="body")
        for v in (*block.arguments, *(v for op in block.operations for v in op.results))
    }
    renamed = replace(
        block,
        name="body",
        arguments=tuple(values[v] for v in block.arguments),
        operations=tuple(
            replace(
                op,
                operands=tuple(values[v] for v in op.operands),
                results=tuple(values[v] for v in op.results),
            )
            for op in block.operations
        ),
        return_values=tuple(values[v] for v in block.return_values or ()),
    )
    record = replace(
        record,
        functions=(
            replace(function, blocks=(renamed,), entry="body"),
            *record.functions[1:],
        ),
    )
    record = replace(record, canonical_payload={"witness": "retained"})
    copied = copy_component_program(record)
    assert copied.canonical_payload is record.canonical_payload
    assert copied.functions[0].entry == "body"
    assert copied.functions[0].blocks[0].name == "body"
    assert verify_program_record(copied) == ()


@pytest.mark.parametrize("failure", ["unknown", "invalid", "gate", "primitive"])
def test_traversal_and_copy_reject_outside_profile_before_results(failure: str) -> None:
    builder = ComponentProgramBuilder(
        Signature([Register("word", RegisterType("uint", 1))])
    )
    record = builder.finalize(builder.inputs)
    entry = "absent" if failure == "unknown" else "main"
    if failure == "invalid":
        record = replace(record, functions=(*record.functions, record.functions[0]))
    elif failure == "gate":
        from flagquantum.core.ir import CircuitIR
        from tests.unit.test_program_calls import wrap_circuit

        record = wrap_circuit(CircuitIR(1, ()))
    elif failure == "primitive":
        from flagquantum.compiler.registers.component_rebindings import (
            split_word_function,
        )

        record = ProgramRecord((split_word_function("main", RegisterType("uint", 1)),))
    for operation in (
        lambda: list(iter_component_calls(record, entry)),
        lambda: copy_component_program(record, entry),
    ):
        with pytest.raises(ValueError):
            operation()


@pytest.mark.parametrize("width", [0, 1, 3])
def test_copy_keeps_directional_lifetimes_and_source_visitation(width: int) -> None:
    from qualtran import QUInt

    from flagquantum.compiler.registers.register_binding import bind_register_program

    signature = Signature([Register("word", RegisterType("uint", width))])
    native = ComponentProgramBuilder(signature)
    source = _source(signature)
    fresh = native.allocate(dtype=signature.get_left("word").dtype)
    word = native.inputs["word"]
    assert isinstance(word, RegisterBinding)
    native.free(word)
    record = native.finalize({"word": fresh})
    source_fresh = source.builder.allocate(dtype=QUInt(width))
    source.builder.free(source.inputs["word"])
    composite = source.builder.finalize(word=source_fresh)
    assert _native_trace(record) == _source_trace(source, composite)
    copied = copy_component_program(record)
    assert _native_trace(copied) == _source_trace(source, composite.copy())
    kwargs: Any = {
        "input_ownership": tuple((i, False) for i in range(width)),
        "allow_workspace_outputs": True,
        "clean": (0,) if width == 0 else (),
    }
    inputs = (tuple(range(width)),)
    assert bind_register_program(copied, inputs, **kwargs) == bind_register_program(
        record, inputs, **kwargs
    )


@pytest.mark.parametrize("width", [1, 2, 3, 8])
def test_erasure_traversal_and_copy_match_original_owned_classical_graph(
    width: int,
) -> None:
    from qualtran import QGF
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import GF2MulMBUC

    from flagquantum.compiler.registers.finite_field.product_erasure_program import (
        product_erasure_program,
    )
    from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding

    source = _source(Signature([]))
    composite = GF2MulMBUC(QGF(2, width)).decompose_bloq()
    record = product_erasure_program(FiniteFieldEncoding(2, width).register_type())
    assert _native_trace(record) == _source_trace(source, composite)
    assert _native_trace(copy_component_program(record)) == _source_trace(
        source, composite.copy()
    )
