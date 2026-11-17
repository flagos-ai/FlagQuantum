"""Compare default owned-word lifecycle drawing with frozen Allocate and Free."""

from hashlib import sha1
from pathlib import Path

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
def frozen_sources():
    from qualtran.bloqs.bookkeeping import allocate, free

    for module, expected in (
        (allocate, "e459095285de98c1468b661afdbcbe58238ff1ad"),
        (free, "8a0c73db71e1d8ffa31b2c7a15d246f02f8f4f63"),
    ):
        raw = Path(module.__file__).read_bytes()
        assert sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == expected


@pytest.mark.parametrize("dirty", (False, True))
@pytest.mark.parametrize("width", (1, 3))
def test_native_owned_lifetime_callbacks_and_pool_match_locked_source(width, dirty):
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
