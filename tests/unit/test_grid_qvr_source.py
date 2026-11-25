"""Complete source QVR metadata, symbol identity, constructors and glyphs."""

import json
from hashlib import sha1
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest
import sympy

from flagquantum.compiler._component_trace.call_graph import SymbolAllocator
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_grid_qvr_factories import grid_qvr_factories
from flagquantum.compiler.registers.phase.grid_qvr_declaration import (
    GridVariableRotationDeclaration,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord, verify_program_record
from flagquantum.core.ir.program.register_types import unsigned_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.drawer.chemistry_symbols import grid_qvr_wire_symbol
from flagquantum.drawer.register_score import RegisterStep, layout_register_score
from flagquantum.drawer.register_score_data import (
    export_register_score_json,
    register_score_data,
)
from flagquantum.errors import SerializationError
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_subtraction_foreign_file import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_KEY = "qualtran.bloqs.chemistry.trotter.grid_ham.qvr.QuantumVariableRotation"
_WIDTHS = (0, 1, 12, sympy.Symbol("n"))


@pytest.mark.parametrize("width", _WIDTHS)
def test_source_calls_keep_angle_identity_precision_and_query_allocator(
    width: Any,
) -> None:
    from qualtran.bloqs.chemistry.trotter.grid_ham.qvr import QuantumVariableRotation
    from qualtran.resource_counting import SympySymbolAllocator

    source, native = QuantumVariableRotation(width), GridVariableRotationDeclaration(
        width
    )
    assert native.signature == native_signature(source.signature)
    source_symbols, symbols = SympySymbolAllocator(), SymbolAllocator()
    assert source_symbols.new_symbol("other") == symbols.new_symbol("other")
    for i in range(3):
        ((expected, count),) = source.build_call_graph(source_symbols).items()
        ((actual, multiplicity),) = native.direct_calls(symbols).items()
        assert actual.opcode == "rz"
        assert actual.parameter == expected.angle == sympy.Symbol(f"_theta{i}")
        assert actual.precision == expected.eps == 1e-11
        assert multiplicity == count == width
    # A fresh query scope starts at theta0; repeated queries sharing an allocator
    # remain distinct exactly as they do upstream.
    assert next(iter(native.direct_calls())).parameter == sympy.Symbol("_theta0")


@pytest.mark.parametrize("width", _WIDTHS)
def test_generic_original_rotation_and_width_estimates(width: Any) -> None:
    from qualtran.bloqs.chemistry.trotter.grid_ham.qvr import QuantumVariableRotation
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = QuantumVariableRotation(width)
    formula = GridVariableRotationDeclaration(width).resource_formula()
    expected = get_cost_value(source, QECGatesCost())
    for category in (
        "t",
        "toffoli",
        "clifford",
        "and_bloq",
        "rotation",
        "measurement",
        "cswap",
    ):
        assert getattr(formula, category) == getattr(expected, category)
    assert formula.logical_qubits == get_cost_value(source, QubitCount())
    if width == 0:
        # The unchanged source counts the one-qubit child in its fallback even
        # when that child has zero multiplicity. Preserve that model anomaly.
        assert formula.rotation == 0 and formula.logical_qubits == 1


@pytest.mark.parametrize("width", _WIDTHS)
@pytest.mark.parametrize("depth", (0, 1, 2))
def test_actual_source_library_retains_bodyless_root_and_original_metadata(
    width: Any, depth: int
) -> None:
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.chemistry.trotter.grid_ham.qvr import QuantumVariableRotation

    source = QuantumVariableRotation(width)
    library = _library(source, depth)
    raw = serialize_component_library(library)
    assert serialize_component_library(deserialize_component_library(raw)) == raw
    program = resolve_component_program(library, 0, grid_qvr_factories())
    assert verify_program_record(program) == ()
    assert program.functions[0].registers == tuple(native_signature(source.signature))
    assert not program.functions[0].blocks
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
    with pytest.raises(ValueError, match="bodies"):
        bind_register_program(program, ())
    if depth:
        assert library.entries[0].counts
        # Original Rz(angle=_theta0) data remains in the actual source wire; it
        # is model data and cannot be scheduled as a root decomposition.
        assert any(entry.declaration.name.endswith(".Rz") for entry in library.entries)


def test_factory_and_docspec_and_invalid_constructor_metadata() -> None:
    from qualtran.bloqs.chemistry.trotter.grid_ham.qvr import (
        _QVR,
        QuantumVariableRotation,
        _qvr,
    )

    source = _qvr()
    assert source == QuantumVariableRotation(12)
    assert _QVR.bloq_cls is QuantumVariableRotation and _QVR.examples == (_qvr,)
    factory = grid_qvr_factories()[_KEY]
    result = factory("grid", None, {"phi_bitsize": 12})
    assert isinstance(result, FunctionRecord)
    assert result.registers == tuple(GridVariableRotationDeclaration(12).signature)
    for bad in (
        {},
        {"phi_bitsize": True},
        {"phi_bitsize": "12"},
        {"phi_bitsize": 12, "radians_per_unit": 0.5},
    ):
        with pytest.raises(SerializationError):
            factory("bad", None, bad)
    with pytest.raises(SerializationError, match="signature"):
        factory(
            "bad", (Register("phi", unsigned_register_type(12)),), {"phi_bitsize": 12}
        )


@pytest.mark.parametrize("width", (0, 1, 12))
def test_complete_original_score_geometry_json_and_rendered_labels(width: int) -> None:
    import matplotlib.pyplot as plt
    from attrs import asdict
    from qualtran import BloqBuilder
    from qualtran.bloqs.chemistry.trotter.grid_ham.qvr import QuantumVariableRotation
    from qualtran.drawing import get_musical_score_data

    from flagquantum.drawer.register_score_mpl import draw_register_score
    from tests.unit.test_drawer_component_score import assert_source_geometry

    source = QuantumVariableRotation(width)
    (port,) = GridVariableRotationDeclaration(width).signature
    step = RegisterStep(
        "qvr", ("phi",), ("phi",), input_registers=(port,), output_registers=(port,)
    )
    for native_reg, source_reg, idx in (
        (None, None, None),
        (port, next(iter(source.signature)), ()),
    ):
        symbol = source.wire_symbol(source_reg, idx or ())
        assert grid_qvr_wire_symbol(step, native_reg, idx) == (
            type(symbol).__name__,
            asdict(symbol),
        )
    builder, incoming = BloqBuilder.from_signature(source.signature)
    graph = builder.finalize(**builder.add_d(source, **incoming))
    score = layout_register_score(
        {"phi": width},
        ("phi",),
        (step,),
        input_registers=(port,),
        output_registers=(port,),
        outputs=("phi",),
    )
    data = register_score_data(score, register_wire_symbol=grid_qvr_wire_symbol)
    assert_source_geometry(data, get_musical_score_data(graph), ordered_ports=True)
    assert json.loads(export_register_score_json(data)) == data
    figure, axes = draw_register_score(data)
    try:
        figure.canvas.draw()
        assert {"e^{i*phi}", "phi"} <= {artist.get_text() for artist in axes.texts}
    finally:
        plt.close(figure)


def test_complete_frozen_source_test_assets_and_real_extension() -> None:
    import kickmix

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and callable(kickmix.Simulator)
    source = (
        Path(__file__).resolve().parents[3]
        / "ftqc-reference-env/qualtran/bloqs/chemistry/trotter/grid_ham"
    )
    for name, digest in (
        ("qvr.py", "0809fd98bc131943becaaeebbf4fe59cb49e8743"),
        ("qvr_test.py", "e918f31419e39d7adb2e386bdd65759c35c6f4b5"),
    ):
        raw = (source / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest


def test_native_declaration_and_binding_need_no_reference_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    original = builtins.__import__

    def no_source(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq"}:
            raise ImportError(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_source)
    declaration = GridVariableRotationDeclaration(12)
    assert declaration.resource_formula().rotation == 12
    assert next(iter(declaration.direct_calls())).parameter == sympy.Symbol("_theta0")
    result = grid_qvr_factories()[_KEY]("qvr", None, {"phi_bitsize": 12})
    assert isinstance(result, FunctionRecord)
    assert result.registers == tuple(declaration.signature)
