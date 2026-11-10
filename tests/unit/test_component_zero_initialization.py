"""Zero-input component wrappers preserve typed ownership and source call order."""

from collections import Counter
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_zero_initialization import (
    initialize_component_from_zero,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    BlockRecord,
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_component_expansion import _hierarchy

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def phase_fixture(width: int, shape: tuple[int, ...]) -> tuple[ProgramRecord, Any]:
    from qualtran import BloqBuilder, QAny
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import Hadamard, SGate

    base, _ = _hierarchy(1, 1)
    h = base.functions[1]
    s = replace(
        h,
        name="s",
        blocks=tuple(
            replace(
                block,
                operations=tuple(
                    replace(op, opcode="s") if op.opcode == "h" else op
                    for op in block.operations
                ),
            )
            for block in h.blocks
        ),
    )
    sig = Signature(
        [
            Register("q", quantum_bit_type(), (2,)),
            Register("idle", RegisterType("uint", width), shape),
        ]
    )
    builder = ComponentProgramBuilder(sig, (h, s))
    q = np.asarray(builder.inputs["q"]).copy()
    source_sig = SourceSignature(
        [
            SourceRegister("q", QAny(1), shape=(2,)),
            SourceRegister("idle", QAny(width), shape=shape),
        ]
    )
    bb, source_data = BloqBuilder.from_signature(source_sig)
    for name, index in (("h", 0), ("s", 0), ("h", 1), ("s", 1), ("h", 0)):
        q[index] = builder.call_dict(name, {"q": q[index]})["q"]
        source_data["q"][index] = bb.add(
            Hadamard() if name == "h" else SGate(), q=source_data["q"][index]
        )
    return builder.finalize({"q": q, "idle": builder.inputs["idle"]}), bb.finalize(
        **source_data
    )


@pytest.mark.parametrize("width", [0, 1, 2])
@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,)])
def test_original_initialization_calls_shapes_and_owned_output_binding(
    width: int, shape: tuple[int, ...]
) -> None:
    from qualtran.bloqs.basic_gates import IntState
    from qualtran.simulation.tensor._quimb import initialize_from_zero

    record, source = phase_fixture(width, shape)
    initialized = initialize_component_from_zero(record)
    assert verify_program_record(initialized) == ()
    root = initialized.functions[0]
    assert root.name == "from_zero" and root.blocks[0].arguments == ()
    assert all(r.side == RegisterSide.RIGHT for r in root.registers or ())
    assert all(
        a is b
        for a, b in zip(
            record.functions,
            initialized.functions[1 : 1 + len(record.functions)],
            strict=True,
        )
    )
    operations = root.blocks[0].operations
    definitions = {f.name: f for f in initialized.functions}
    widths = [
        int((definitions[op.callee or ""].registers or ())[0].bitsize)
        for op in operations[:-1]
    ]
    if shape == (0,):
        from qualtran import BloqBuilder
        from qualtran.simulation.tensor._quimb import _add_classical_kets

        with pytest.raises(IndexError, match="out of bounds"):
            initialize_from_zero(source)
        bb = BloqBuilder()
        values = _add_classical_kets(bb, source.signature.lefts())
        bb.add_d(source, **values)
        original_calls = sorted(bb._binsts, key=lambda i: i.i)
    else:
        original = initialize_from_zero(source)
        original_calls = sorted(original.bloq_instances, key=lambda i: i.i)
    assert widths == [
        i.bloq.bitsize for i in original_calls[:-1] if isinstance(i.bloq, IntState)
    ]
    assert operations[-1].callee == "main" and original_calls[-1].bloq == source
    assert len(operations) == len(original_calls)
    size = 2 + width * int(np.prod(shape))
    bound = bind_register_program(
        initialized,
        (),
        entry="from_zero",
        clean=tuple(range(size + 1)),
        allow_workspace_outputs=True,
    )
    assert Counter(g.name for g in bound.circuit.instructions) == {"h": 3, "s": 2}
    if width > 0 and shape != (0,):
        from tests.unit.test_register_constant_addition import (
            reference_arithmetic_gates,
        )
        from tests.unit.test_register_subtraction import assert_source

        output_words = iter(bound.outputs)
        outputs = {
            r.name: tuple(wire for _ in r.all_idxs() for wire in next(output_words))
            for r in root.registers or ()
        }
        expected = reference_arithmetic_gates(
            original, {}, tuple(range(size + 1)), {}, expected_outputs=outputs
        )
        assert_source(bound.circuit.instructions, expected)
    assert sum(map(len, bound.outputs)) == size
    assert len({wire for word in bound.outputs for wire in word}) == size
    assert all(not event.dirty for event in bound.events)
    with pytest.raises(ValueError, match="output|live|workspace"):
        bind_register_program(
            initialized, (), entry="from_zero", clean=tuple(range(size + 1))
        )


def test_metadata_namespace_collisions_and_empty_input_entry() -> None:
    record, _ = phase_fixture(1, ())
    collision = replace(record.functions[1], name="__fq_zero_0_reserved")
    record = replace(
        record,
        functions=(*record.functions, collision),
        canonical_payload={"stale": True},
    )
    initialized = initialize_component_from_zero(record, name="prepared")
    assert initialized.canonical_payload is None
    assert any(f.name.startswith("__fq_zero_1_") for f in initialized.functions)
    with pytest.raises(ValueError, match="conflicts"):
        initialize_component_from_zero(initialized, name="prepared")
    empty = ComponentProgramBuilder(Signature([])).finalize({})
    result = initialize_component_from_zero(empty)
    assert len(result.functions[0].blocks[0].operations) == 1
    assert result.functions[0].registers == ()


def test_invalid_unknown_untyped_and_classical_entries() -> None:
    record, _ = phase_fixture(1, ())
    with pytest.raises(ValueError, match="unknown"):
        initialize_component_from_zero(record, entry="missing")
    with pytest.raises(ValueError, match="invalid zero"):
        initialize_component_from_zero(
            replace(record, functions=record.functions + (record.functions[0],))
        )
    untyped = ProgramRecord(
        (FunctionRecord("main", (BlockRecord("entry", return_values=()),)),)
    )
    with pytest.raises(ValueError, match="typed logical"):
        initialize_component_from_zero(untyped)
    builder = ComponentProgramBuilder(Signature([Register("c", classical_bit_type())]))
    with pytest.raises(ValueError, match="classical"):
        initialize_component_from_zero(builder.finalize(builder.inputs))
