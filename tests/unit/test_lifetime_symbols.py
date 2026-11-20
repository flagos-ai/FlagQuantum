"""Compare default owned-word lifecycle drawing with frozen Allocate and Free."""

from hashlib import sha1
from pathlib import Path
from typing import Any

import pytest

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.register_types import (
    quantum_bit_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_score_data import register_score_data
from tests.unit.register_source_workspace import SourceWorkspace
from tests.unit.test_drawer_component_score import assert_source_geometry

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_sources() -> None:
    from qualtran.bloqs.bookkeeping import allocate, free

    for module, expected in (
        (allocate, "e459095285de98c1468b661afdbcbe58238ff1ad"),
        (free, "8a0c73db71e1d8ffa31b2c7a15d246f02f8f4f63"),
    ):
        raw = Path(module.__file__).read_bytes()
        assert sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == expected


@pytest.mark.parametrize("dirty", (False, True))
@pytest.mark.parametrize("width", (1, 3))
def test_native_owned_lifetime_callbacks_and_pool_match_locked_source(
    width: int, dirty: bool
) -> None:
    from qualtran import BloqBuilder, QBit, QUInt
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.bookkeeping import Allocate, Free
    from qualtran.drawing import get_musical_score_data
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    dtype = quantum_bit_type() if width == 1 else unsigned_register_type(width)
    builder = ComponentProgramBuilder(
        Signature((Register("spectator", quantum_bit_type()),))
    )
    word = builder.allocate(dtype=dtype, dirty=dirty)
    builder.free(word, dirty=dirty)
    record = builder.finalize(builder.inputs)

    source_dtype = QBit() if width == 1 else QUInt(width)
    source, inputs = BloqBuilder.from_signature(SourceSignature.build(spectator=1))
    allocate = Allocate(source_dtype, dirty=dirty)
    free = Free(source_dtype, dirty=dirty)
    allocated = source.add(allocate)
    source.add(free, reg=allocated)
    reference = source.finalize(**inputs)
    actual = register_score_data(component_register_score(record))
    assert_source_geometry(
        actual, get_musical_score_data(reference), ordered_ports=True
    )

    pool = tuple(reversed(range(1, width + 1)))
    bound = bind_register_program(
        record, ((0,),), clean=() if dirty else pool, borrowed=pool if dirty else ()
    )
    assert bound.outputs == ((0,),)
    assert bound.circuit.instructions == ()
    source_pool = SourceWorkspace(() if dirty else pool, pool if dirty else ())
    operation, binding = allocate.as_cirq_op(source_pool)
    assert operation is None
    assert free.as_cirq_op(source_pool, **binding) == (None, {})
    assert not source_pool.live
    assert (
        [(event.kind, event.wires, event.dirty) for event in bound.events]
        == source_pool.events
        == [
            ("allocate", pool, dirty),
            ("free", pool, dirty),
        ]
    )
    assert bound.circuit.n_wires == width + 1
    # The source model evaluates the disconnected spectator independently.
    assert get_cost_value(reference, QubitCount()) == width
    assert get_cost_value(reference, QECGatesCost()).asdict() == {}


@pytest.mark.parametrize("dirty", (False, True))
@pytest.mark.parametrize("expression", ("n", "2*n+1", "3"))
def test_symbolic_lifetime_widths_remain_display_metadata(
    expression: str, dirty: bool
) -> None:
    sympy = pytest.importorskip("sympy")
    from qualtran import BloqBuilder, QUInt
    from qualtran.bloqs.bookkeeping import Allocate, Free
    from qualtran.drawing import get_musical_score_data

    from flagquantum.drawer.register_graph import (
        export_register_graph_dot,
        register_connections,
    )
    from flagquantum.drawer.register_score_data import export_register_score_json
    from flagquantum.drawer.register_score_html import export_register_score_html

    n = sympy.Symbol("n")
    width = {"n": n, "2*n+1": 2 * n + 1, "3": sympy.Integer(3)}[expression]
    dtype = unsigned_register_type(width)
    builder = ComponentProgramBuilder(Signature(()))
    word = builder.allocate(dtype=dtype, dirty=dirty)
    builder.free(word, dirty=dirty)
    record = builder.finalize({})
    source = BloqBuilder()
    allocated = source.add(Allocate(QUInt(width), dirty=dirty))
    source.add(Free(QUInt(width), dirty=dirty), reg=allocated)
    score = component_register_score(record)
    assert len(score.lines) == 1 and score.lines[0].width is width
    assert_source_geometry(
        register_score_data(score),
        get_musical_score_data(source.finalize()),
        ordered_ports=True,
    )
    (connection,) = register_connections(score)
    assert connection.width is width
    assert f'label="{width}"' in export_register_graph_dot(score)
    data = register_score_data(score)
    assert "alloc" in export_register_score_json(data)
    assert "free" in export_register_score_html(data)
    # A displayed width never supplies a concrete physical allocation size.
    with pytest.raises(ValueError, match="unsupported executable register descriptor"):
        bind_register_program(record, (), clean=(0, 1, 2), borrowed=(3, 4, 5))


@pytest.mark.parametrize("width", (-1, True, 1.5, "n"))
def test_plain_score_retains_invalid_concrete_width_refusal(width: Any) -> None:
    from flagquantum.drawer.register_score import layout_register_score

    with pytest.raises(
        ValueError, match="Register widths must be nonnegative integers"
    ):
        layout_register_score({"reg": width}, ("reg",), ())


@pytest.mark.parametrize("allocate", (False, True))
def test_actual_symbolic_factory_open_boundary_geometry_and_width_label(
    allocate: bool,
) -> None:
    from qualtran import BloqBuilder
    from qualtran.bloqs.bookkeeping.allocate import _alloc
    from qualtran.bloqs.bookkeeping.free import _free
    from qualtran.drawing import get_musical_score_data

    from flagquantum.core.ir.program.register_bindings import RegisterBinding
    from tests.unit.test_subtraction_foreign_file import native_signature

    source = (_alloc if allocate else _free).make()
    signature = native_signature(source.signature)
    builder = ComponentProgramBuilder(signature)
    if allocate:
        word = builder.allocate(dtype=signature[0].dtype)
        outputs = {"reg": word}
    else:
        incoming = builder.inputs["reg"]
        assert isinstance(incoming, RegisterBinding)
        builder.free(incoming)
        outputs = {}
    record = builder.finalize(outputs)
    original, inputs = BloqBuilder.from_signature(source.signature)
    expected = original.finalize(**original.add_d(source, **inputs))
    data = register_score_data(component_register_score(record))
    assert_source_geometry(data, get_musical_score_data(expected), ordered_ports=True)
    boundary = [p for p in data["soqs"] if p["ident"].startswith("dang,")]
    assert len(boundary) == 1
    assert "/QAny(n)" in str(boundary[0]["symb_attributes"]["text"])
