"""Compare register score geometry and lifetimes with the locked layout."""

import pytest

from examples.synthesis_local_registers import local_composition
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.drawer.register_score import (
    RegisterLine,
    RegisterStep,
    layout_register_score,
    register_program_score,
)
from tests.register_program_cases import Registers, product

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
def test_score_coordinates_match_locked_layout() -> None:
    pytest.importorskip("qualtran")
    from attrs import frozen
    from qualtran import (
        Bloq,
        BloqBuilder,
        BloqInstance,
        QAny,
        QBit,
        Register,
        Side,
        Signature,
    )
    from qualtran.bloqs.basic_gates import Hadamard, XGate
    from qualtran.drawing.musical_score import _cbloq_musical_score

    @frozen
    class WorkPort(Bloq):
        side: Side

        @property
        def signature(self) -> Signature:
            return Signature(
                [Register("x", QBit()), Register("work", QAny(3), side=self.side)]
            )

    signature = Signature.build(x=1, y=1)
    builder, entry = BloqBuilder.from_signature(signature)
    x = builder.add(Hadamard(), q=entry["x"])
    y = builder.add(XGate(), q=entry["y"])
    x, first = builder.add(WorkPort(Side.RIGHT), x=x)
    x = builder.add(WorkPort(Side.LEFT), x=x, work=first)
    x, second = builder.add(WorkPort(Side.RIGHT), x=x)
    x = builder.add(WorkPort(Side.LEFT), x=x, work=second)
    composite = builder.finalize(x=x, y=y)
    _, positions, manager = _cbloq_musical_score(signature, composite._binst_graph)

    score = layout_register_score(
        {"x": 1, "y": 1, "first": 3, "second": 3},
        ("x", "y"),
        (
            RegisterStep("0", ("x",), ("x",)),
            RegisterStep("1", ("y",), ("y",)),
            RegisterStep("2", ("x",), ("x", "first")),
            RegisterStep("3", ("x", "first"), ("x",)),
            RegisterStep("4", ("x",), ("x", "second")),
            RegisterStep("5", ("x", "second"), ("x",)),
        ),
    )
    source_columns: dict[str, tuple[int, int, set[int]]] = {}
    for soquet, position in positions.items():
        if isinstance(soquet.binst, BloqInstance):
            _, _, lanes = source_columns.setdefault(
                str(soquet.binst.i), (position.seq_x, position.topo_gen, set())
            )
            lanes.add(position.y)
    assert {
        col.step.name: (col.sequence, col.generation, set(col.lanes))
        for col in score.columns
    } == source_columns
    assert {(line.lane, line.start, line.end) for line in score.lines} == {
        (line.y, line.seq_x_start, line.seq_x_end) for line in manager.hlines
    }
    assert score.lines[-2:] == (
        RegisterLine("first", 3, 2, 2, 3),
        RegisterLine("second", 3, 2, 4, 5),
    )
    assert score.output_generation == 6


def test_new_lane_is_assigned_before_input_only_lane_is_released() -> None:
    score = layout_register_score(
        {"a": 1, "b": 1, "c": 1},
        ("a",),
        (RegisterStep("replace", ("a",), ("b",)), RegisterStep("next", ("b",), ("c",))),
    )
    assert [column.lanes for column in score.columns] == [(0, 1), (1, 0)]
    assert [(line.register, line.lane) for line in score.lines] == [
        ("a", 0),
        ("c", 0),
        ("b", 1),
    ]


def test_native_uncompute_has_the_implicit_borrowed_inputs() -> None:
    function = capture(product, {"a": 1, "b": 1})
    registers = Registers()
    x = registers.declare("x", 1)
    result = registers.declare("result", 2, local=True)
    registers.emit("qreg.allocate", x)
    registers.emit("qreg.compute", result, x, x, callee=function.name)
    registers.emit("qreg.uncompute", result)
    registers.emit("qreg.release", result)
    score = register_program_score(registers.finish(), (function,))
    uncompute = next(col for col in score.columns if col.step.name == "qreg.uncompute")
    compute = next(col for col in score.columns if col.step.name == "qreg.compute")
    assert uncompute.step.inputs == uncompute.step.outputs == ("result", "x")
    assert uncompute.lanes == compute.lanes
    assert uncompute.generation == compute.generation + 1
    assert score.lines[0].register == "x"
    assert score.lines[0].end == len(score.columns)


@pytest.mark.parametrize("helpers", [False, True])
def test_score_expands_native_register_helpers(helpers: bool) -> None:
    program, functions = local_composition(overlapping=False, helpers=helpers)
    score = register_program_score(program, functions)
    names = {col.step.name for col in score.columns}
    assert "qreg.call" not in names
    assert {"qreg.local", "qreg.compute", "qreg.uncompute", "qreg.release"} <= names
    assert all(line.start < line.end for line in score.lines)


def test_score_rejects_unverified_native_lifetimes() -> None:
    registers = Registers()
    x = registers.declare("x", 1)
    registers.declare("unreleased", 1, local=True)
    registers.emit("qreg.allocate", x)
    with pytest.raises(HybridVerificationError, match="register.local"):
        register_program_score(registers.finish())
