"""Compare Partition and its two-word specializations with the frozen drawing."""

from hashlib import sha1
from pathlib import Path

import pytest

from flagquantum.compiler.registers.layout.component_partition import (
    partition_signature,
)
from flagquantum.core.ir.program.register_types import opaque_register_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.drawer.partition_symbols import partition_wire_symbol
from flagquantum.drawer.register_score import RegisterStep, layout_register_score
from flagquantum.drawer.register_score_data import register_score_data
from tests.unit.test_drawer_component_score import assert_source_geometry

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_partition():
    from qualtran.bloqs.bookkeeping import partition

    raw = Path(partition.__file__).read_bytes()
    assert sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == (
        "8cebbb58095d172f3db564cdfc231537e6214957"
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("shape", [(2, 2), (2,), (0, 2)])
def test_shaped_partition_score_matches_source(reverse, shape):
    from qualtran import BloqBuilder, QAny
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.bookkeeping.partition import Partition
    from qualtran.drawing import get_musical_score_data

    parts = (
        Register("sites", opaque_register_type(2), shape),
        Register("tag", opaque_register_type(1)),
    )
    signature = partition_signature(parts, reverse=reverse)
    source = Partition(
        signature.n_qubits(),
        (SourceRegister("sites", QAny(2), shape), SourceRegister("tag", QAny(1))),
        partition=not reverse,
    )
    builder, values = BloqBuilder.from_signature(source.signature)
    original = get_musical_score_data(
        builder.finalize(**builder.add_d(source, **values))
    )

    left = tuple(signature.lefts())
    right = tuple(signature.rights())
    identities = {
        reg.name: tuple(f"{reg.name}:{idx}" for idx in reg.all_idxs())
        for reg in signature
    }
    inputs = tuple(identity for reg in left for identity in identities[reg.name])
    outputs = tuple(identity for reg in right for identity in identities[reg.name])
    step = RegisterStep(
        "partition", inputs, outputs, input_registers=left, output_registers=right
    )
    score = layout_register_score(
        {
            identity: int(reg.bitsize)
            for reg in signature
            for identity in identities[reg.name]
        },
        inputs,
        (step,),
        input_registers=left,
        output_registers=right,
        outputs=outputs,
    )
    native = register_score_data(score, register_wire_symbol=partition_wire_symbol)
    assert_source_geometry(native, original, ordered_ports=True)


@pytest.mark.parametrize("reverse", [False, True])
def test_two_word_ports_and_through_glyph_match_source(reverse):
    from qualtran import QAny, Side
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.bookkeeping.partition import Split2

    source = Split2(2, 3)
    if reverse:
        source = source.adjoint()
    step = RegisterStep("split2" if not reverse else "join2")
    signature = partition_signature(
        (
            Register("y1", opaque_register_type(2)),
            Register("y2", opaque_register_type(3)),
        ),
        reverse=reverse,
    )
    for native, foreign in zip(signature, source.signature, strict=True):
        expected = source.wire_symbol(foreign)
        assert partition_wire_symbol(step, native, ()) == (
            type(expected).__name__,
            {"text": expected.text},
        )
    through = Register("label", opaque_register_type(1), side=RegisterSide.THRU)
    expected = source.wire_symbol(SourceRegister("label", QAny(1), side=Side.THRU))
    assert partition_wire_symbol(step, through, None) == (
        type(expected).__name__,
        {"text": expected.text},
    )
    empty = source.wire_symbol(None)
    assert partition_wire_symbol(step, None, None) == (
        type(empty).__name__,
        {"text": empty.text, "fontsize": empty.fontsize},
    )
