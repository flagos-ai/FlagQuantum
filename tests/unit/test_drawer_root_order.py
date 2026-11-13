"""Independent allocation roots follow first connection encounter order."""

import csv
from hashlib import sha1
from itertools import permutations
from pathlib import Path
from typing import Any

import pytest

from flagquantum.compiler.registers.component_basis_vectors import (
    integer_basis_function,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_score_data import register_score_data
from tests.unit.test_drawer_component_score import assert_source_geometry
from tests.unit.test_program_classical_registers import identity

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ORDERS = [(0, 1), (1, 0), *permutations(range(3)), (3, 1, 0, 2), (3, 2, 1, 0)]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    root = Path(__file__).resolve().parents[3]
    with (root / "FlagQuantum-Qualtran-Kickmix-移植进度.csv").open(
        encoding="utf-8-sig"
    ) as stream:
        blobs = {row["源文件"]: row["源对象"] for row in csv.DictReader(stream)}
    for name in (
        "qualtran/_infra/composite_bloq.py",
        "qualtran/drawing/musical_score.py",
    ):
        raw = (root / "ftqc-reference-env" / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blobs[name]


def root_fixture(
    order: tuple[int, ...], width: int, idle: bool, consume_roots: bool = True
) -> tuple[ProgramRecord, Any]:
    from attrs import frozen
    from qualtran import Bloq, BloqBuilder, QAny, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature

    dtype = RegisterType("uint", width)
    ports = tuple(Register(f"q{i}", dtype) for i in range(len(order)))
    signature = Signature(
        (
            *(Register(reg.name, dtype, side=RegisterSide.RIGHT) for reg in ports),
            *((Register("idle", dtype),) if idle else ()),
        )
    )
    prepare = integer_basis_function("Prepare", dtype, 0)
    consume = identity("Consume", ports)
    builder = ComponentProgramBuilder(signature, (prepare, consume))
    roots = [builder.call_dict("Prepare", {})["val"] for _ in order]
    values = {port.name: roots[index] for port, index in zip(ports, order, strict=True)}
    if consume_roots:
        values = builder.call_dict("Consume", values)
    if idle:
        values["idle"] = builder.inputs["idle"]
    record = builder.finalize(values)
    source_dtype = QAny(width)
    source_ports = tuple(SourceRegister(port.name, source_dtype) for port in ports)
    root_signature = SourceSignature(
        (
            *(
                SourceRegister(port.name, source_dtype, side=Side.RIGHT)
                for port in ports
            ),
            *((SourceRegister("idle", source_dtype),) if idle else ()),
        )
    )

    def prepare_signature(self: object) -> Any:
        return SourceSignature((SourceRegister("val", source_dtype, side=Side.RIGHT),))

    def consume_signature(self: object) -> Any:
        return SourceSignature(source_ports)

    source_prepare = frozen(
        type("Prepare", (Bloq,), {"signature": property(prepare_signature)})
    )()
    source_consume = frozen(
        type("Consume", (Bloq,), {"signature": property(consume_signature)})
    )()
    bb, inputs = BloqBuilder.from_signature(root_signature)
    source_roots = [bb.add(source_prepare) for _ in order]
    outputs = {
        port.name: source_roots[index] for port, index in zip(ports, order, strict=True)
    }
    if consume_roots:
        outputs = bb.add_d(source_consume, **outputs)
    if idle:
        outputs["idle"] = inputs["idle"]
    return record, bb.finalize(**outputs)


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("idle", [False, True])
@pytest.mark.parametrize("consume_roots", [False, True])
def test_root_positions_lifetimes_and_default_records_match_source(
    order: tuple[int, ...], width: int, idle: bool, consume_roots: bool
) -> None:
    from qualtran.drawing import get_musical_score_data

    record, source = root_fixture(order, width, idle, consume_roots)
    before = hash(record)
    score = component_register_score(record)
    expected_order = (*order, len(order)) if consume_roots else order
    assert tuple(column.ordinal for column in score.columns) == expected_order
    assert_source_geometry(
        register_score_data(score), get_musical_score_data(source), ordered_ports=True
    )
    assert hash(record) == before


@pytest.mark.parametrize("order", [(1, 0), (2, 0, 1), (3, 1, 0, 2)])
@pytest.mark.parametrize("idle", [False, True])
def test_root_typed_callbacks_preserve_source_events_and_stateful_glyphs(
    monkeypatch: pytest.MonkeyPatch, order: tuple[int, ...], idle: bool
) -> None:
    from qualtran.drawing import get_musical_score_data

    from tests.unit.test_drawer_register_symbols import (
        Event,
        native_symbols,
        source_symbols,
    )

    record, source = root_fixture(order, 1, idle)
    actual_events: list[Event] = []
    source_events: list[Event] = []
    source_symbols(monkeypatch, source, source_events, numbered=True)
    actual = register_score_data(
        component_register_score(record),
        register_wire_symbol=native_symbols(actual_events, numbered=True),
    )
    expected = get_musical_score_data(source)
    assert actual_events == source_events
    assert_source_geometry(actual, expected, ordered_ports=True)


@pytest.mark.parametrize("consume_roots", [False, True])
@pytest.mark.parametrize("idle", [False, True])
def test_root_pixels_retain_source_record_order_with_boundary_text_adaptation(
    consume_roots: bool, idle: bool
) -> None:
    import attrs
    from matplotlib import pyplot as plt
    from qualtran.drawing.musical_score import (
        Text,
        draw_musical_score,
        get_musical_score_data,
    )

    from flagquantum.drawer.register_score_mpl import draw_register_score
    from tests.unit.test_drawer_custom_symbols import assert_pixels

    record, source = root_fixture((2, 0, 1), 1, idle, consume_roots)
    actual = register_score_data(component_register_score(record))
    expected = get_musical_score_data(source)
    ports = {(port["seq_x"], port["y"]): port for port in actual["soqs"]}
    expected.soqs = [
        (
            attrs.evolve(
                port,
                symb=Text(
                    ports[(port.rpos.seq_x, port.rpos.y)]["symb_attributes"]["text"],
                    fontsize=8,
                ),
            )
            if port.ident.startswith("dang,")
            else port
        )
        for port in expected.soqs
    ]
    assert_source_geometry(actual, expected, ordered_ports=True)
    with plt.rc_context({"figure.dpi": 100, "font.family": "DejaVu Sans"}):
        native, _ = draw_register_score(actual)
        reference, _ = draw_musical_score(expected)
        try:
            assert_pixels(native, reference)
        finally:
            plt.close(native)
            plt.close(reference)
