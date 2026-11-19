"""Generation-delimited diagnostics against the untouched source graph."""

import re
from typing import Any

import numpy as np
import pytest

from examples.algorithms.register_graphs import parallel_program
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.component_debug_text import (
    format_component_debug_text,
    format_component_summary,
)
from tests.unit.test_register_rebinding_contract import SourceRebindings

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
pytest.importorskip("numpy")
pytest.importorskip("qualtran")


def normalized_source_text(record: ProgramRecord, source: object) -> str:
    from qualtran import CompositeBloq

    assert isinstance(source, CompositeBloq)
    operations = record.functions[0].blocks[0].operations
    names = {
        str(instance): f"{operations[instance.i].callee}#{instance.i}"
        for instance in source.bloq_instances
    }
    text = source.debug_text()
    pattern = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
    return re.sub(pattern, lambda match: names[match[0]], text) if names else text


def test_parallel_calls_match_source_generations_and_every_endpoint() -> None:
    from qualtran.bloqs.for_testing.with_decomposition import TestParallelCombo

    record = parallel_program()
    source = TestParallelCombo().decompose_bloq()
    text = format_component_debug_text(record)
    assert text == normalized_source_text(record, source)
    assert text.count("--------------------") == 2
    assert text.count(" -> ") == 14
    assert format_component_summary(record) == str(source).replace(
        "CompositeBloq", "ComponentProgram"
    ).replace("subbloqs", "components")


def test_summary_counts_immediate_calls_without_expanding_shared_definitions() -> None:
    from qualtran import BloqBuilder
    from qualtran.bloqs.for_testing.with_decomposition import TestParallelCombo

    record = parallel_program()
    builder = ComponentProgramBuilder(Signature(record.functions[0].registers or ()))
    outputs = builder.inputs
    source, source_inputs = BloqBuilder.from_signature(TestParallelCombo().signature)
    for _ in range(2):
        outputs = builder.call_program(record, outputs)
        source_inputs = source.add_d(TestParallelCombo(), **source_inputs)
    native = builder.finalize(outputs)
    reference = source.finalize(**source_inputs)
    assert format_component_summary(native) == str(reference).replace(
        "CompositeBloq", "ComponentProgram"
    ).replace("subbloqs", "components")
    assert format_component_summary(native) == "ComponentProgram([2 components...])"
    with pytest.raises(ValueError, match="unknown component entry"):
        format_component_summary(native, "missing")


@pytest.mark.parametrize("width", [0, 1, 3])
@pytest.mark.parametrize("shape", [(), (2,), (2, 2), (0, 2)])
def test_shaped_rebindings_and_return_reordering_match_source(
    width: int, shape: tuple[int, ...]
) -> None:
    signature = Signature((Register("word", RegisterType("uint", width), shape),))
    native = ComponentProgramBuilder(signature)
    source_factory: Any = SourceRebindings
    source = source_factory(signature)
    results: list[object] = []
    for builder in (native, source):
        words = np.asarray(builder.inputs["word"], dtype=object)
        outputs = np.empty(shape, dtype=object)
        for index in signature.get_left("word").all_idxs():
            bits = builder.split(words[index])
            outputs[index] = builder.join(bits[::-1], signature.get_left("word").dtype)
        results.append(outputs if shape else outputs[()])
    record = native.finalize({"word": results[0]})
    expected = source.builder.finalize(word=results[1])
    assert format_component_debug_text(record) == normalized_source_text(
        record, expected
    )


def test_empty_identity_has_no_call_text_and_invalid_entry_is_refused() -> None:
    builder = ComponentProgramBuilder(Signature.build(q=1))
    record = builder.finalize(builder.inputs)
    assert format_component_debug_text(record) == ""
    assert format_component_summary(record) == "ComponentProgram([0 components...])"
    with pytest.raises(ValueError, match="unknown component entry"):
        format_component_debug_text(record, "missing")


def test_expansion_occurrence_labels_are_preserved() -> None:
    from flagquantum.compiler.registers.component_expansion import (
        expand_component_calls_once,
    )

    record = parallel_program()
    signature = Signature(record.functions[0].registers or ())
    builder = ComponentProgramBuilder(signature)
    wrapped = builder.finalize(builder.call_program(record, builder.inputs))
    expanded, labels = expand_component_calls_once(wrapped, instance_ids=(100,))
    text = format_component_debug_text(expanded, instance_ids=labels)
    assert all(f"#{label}" in text for label in labels)
    assert "#100" not in text
    with pytest.raises(ValueError, match="instance_ids"):
        format_component_debug_text(expanded, instance_ids=(1,))


def test_text_projection_preserves_program_and_appended_circuit() -> None:
    import flagquantum as fq
    from examples.algorithms.component_construction_tutorial import (
        BoundComponentExample,
    )
    from flagquantum.compiler.registers.register_binding import bind_register_program

    signature = Signature.build(q=1)
    leaf = bit_gate_function(
        "flip", tuple(signature), CircuitIR(1, (Instruction("x", (0,)),))
    )
    builder = ComponentProgramBuilder(signature, (leaf,))
    q = builder.inputs["q"]
    assert isinstance(q, RegisterBinding)
    record = builder.finalize(builder.call_dict("flip", {"q": q}))
    before = repr(record)
    assert format_component_debug_text(record).splitlines() == [
        "flip#0",
        "  LeftDangle.q -> q",
        "  q -> RightDangle.q",
    ]
    assert repr(record) == before
    bound = bind_register_program(record, ((0,),))
    circuit = fq.Circuit(2)
    circuit.append(BoundComponentExample(record, ((0,),)))
    assert (
        circuit.to_ir().instructions
        == bound.circuit.instructions
        == (Instruction("x", (0,)),)
    )
