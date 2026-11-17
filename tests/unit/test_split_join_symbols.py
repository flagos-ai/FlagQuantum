"""Verify Split and Join glyphs and rebinding on the real typed score path."""

from hashlib import sha1
from pathlib import Path

import pytest

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_rebindings import (
    join_word_function,
    split_word_function,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.partition_symbols import split_join_wire_symbol
from flagquantum.drawer.register_score_data import register_score_data
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from tests.unit.test_drawer_component_score import assert_source_geometry

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_bookkeeping():
    from qualtran.bloqs.bookkeeping import join, split

    for source, digest in (
        (join, "b194add90de87b8aeeb05dd662747701394f400a"),
        (split, "ab02e2e6f41402c732a9f30635e83227a09d2c4a"),
    ):
        raw = Path(source.__file__).read_bytes()
        assert sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == digest


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("width", [0, 1, 2, 5])
def test_typed_split_join_scores_bindings_and_costs_match_source(reverse, width):
    import cirq
    import numpy as np
    from qualtran import QGF, BloqBuilder, QAny, QBit, QUInt
    from qualtran.bloqs.bookkeeping import Split
    from qualtran.drawing import get_musical_score_data
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    dtype, foreign = {
        0: (opaque_register_type(0), QAny(0)),
        1: (quantum_bit_type(), QBit()),
        2: (FiniteFieldEncoding(2, 2).register_type(), QGF(2, 2)),
        5: (unsigned_register_type(5), QUInt(5)),
    }[width]
    original = Split(foreign)
    if reverse:
        original = original.adjoint()
    assert get_cost_value(original, QECGatesCost()).asdict() == {}
    assert get_cost_value(original, QubitCount()) == width

    make_function = join_word_function if reverse else split_word_function
    child = make_function("rebind", dtype)
    assert child.registers is not None
    signature = Signature(child.registers)
    builder = ComponentProgramBuilder(signature, (child,))
    record = builder.finalize(builder.call_dict(child.name, builder.inputs))

    wires = tuple(reversed(range(width)))
    physical = np.asarray([cirq.LineQubit(i) for i in wires], dtype=object)
    values = physical.reshape(width, 1) if reverse else physical
    bits = tuple((wire,) for wire in wires)
    inputs = bits if reverse else (wires,)
    op, expected_wires = original.as_cirq_op(cirq.SimpleQubitManager(), reg=values)
    if not width:
        with pytest.raises(ValueError, match="at least one physical wire"):
            bind_register_program(record, inputs)
    actual = bind_register_program(record, inputs, borrowed=() if width else (0,))
    assert op is None
    assert tuple(qubit.x for qubit in expected_wires["reg"].flat) == wires
    assert actual.outputs == ((wires,) if reverse else bits)
    assert actual.circuit.instructions == actual.events == ()

    source, source_inputs = BloqBuilder.from_signature(original.signature)
    original_graph = source.finalize(**source.add_d(original, **source_inputs))
    score = register_score_data(
        component_register_score(record), register_wire_symbol=split_join_wire_symbol
    )
    if width == 0 and not reverse:
        # The frozen renderer expects a RightDangle node even with no output elements.
        with pytest.raises(KeyError, match="RightDangle"):
            get_musical_score_data(original_graph)
        (word_input,) = original.signature.lefts()
        glyph = original.wire_symbol(word_input)
        (port,) = (p for p in score["soqs"] if p["seq_x"] == 0)
        assert port["symb_cls"] == type(glyph).__name__
        assert port["symb_attributes"] == {"text": glyph.text}
    else:
        assert_source_geometry(
            score, get_musical_score_data(original_graph), ordered_ports=True
        )


@pytest.mark.parametrize("reverse", [False, True])
def test_symbolic_width_is_refused_before_rebinding(reverse):
    from qualtran import QAny
    from qualtran.bloqs.bookkeeping import Join, Split
    from sympy import Symbol

    width = Symbol("n", integer=True, nonnegative=True)
    source, native = (
        (Join, join_word_function) if reverse else (Split, split_word_function)
    )
    with pytest.raises(ValueError, match="symbolic data type"):
        source(QAny(width))
    with pytest.raises(ValueError, match="unsupported executable register descriptor"):
        native("symbolic", opaque_register_type(width))
