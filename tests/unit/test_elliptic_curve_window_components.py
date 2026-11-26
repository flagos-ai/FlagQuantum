"""Original point-window roots retain source calls and explicit native erasure."""

import sys
from collections import Counter
from dataclasses import replace
from hashlib import sha1
from importlib import import_module
from itertools import product
from pathlib import Path
from typing import Any
from unittest.mock import patch as mock_patch

import pytest
import sympy

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives.elliptic_curve_fixed_point import (
    fixed_point_addition_signature,
)
from flagquantum.algorithms.primitives.elliptic_curve_point import ECPoint
from flagquantum.algorithms.primitives.elliptic_curve_window_addition import (
    EllipticCurveWindowAddition,
    windowed_point_table,
)
from flagquantum.compiler.formats.legacy_component_arguments import AffinePointArgument
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_word_factories import _rename_program
from flagquantum.compiler.registers.component_definition_match import _named_definitions
from flagquantum.compiler.registers.lookup.qroam_clean_component import (
    qroam_clean_program,
)
from flagquantum.compiler.registers.lookup.qroam_erasure_component import (
    qroam_measurement_erasure_program,
)
from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem._elliptic_curve_window_library import (
    elliptic_curve_window_factories,
)
from flagquantum.errors import SerializationError
from tests.unit.ec_stage_conformance import pack
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_elliptic_curve_window_addition import layout
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_projectile_nu_library import _named_inputs
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
PREFIX = "qualtran.bloqs.cryptography.ecc.ec_add_r."


@pytest.fixture(scope="module", autouse=True)
def frozen_window_source() -> None:
    source = import_module(PREFIX.rstrip("."))
    assert source.__file__ is not None
    for name, digest in (
        ("ec_add_r.py", "195654d5b4b269d13d07e96a755424755b0196e5"),
        ("ec_add_r_test.py", "b47105fd91c4a3b15de73a25eb5a35943b40558d"),
        ("ec_point.py", "eb885c54c88d663cbec19f44cdcbbb5e9cf424ef"),
    ):
        raw = (Path(source.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest


@pytest.fixture(scope="module", params=(1, 5))
def window_graph(request: pytest.FixtureRequest) -> tuple[Any, ProgramRecord]:
    from qualtran.bloqs.cryptography.ecc.ec_add_r import ECWindowAddR
    from qualtran.bloqs.cryptography.ecc.ec_point import ECPoint as SourcePoint

    source = ECWindowAddR(3, SourcePoint(0, 1, 3, 1), int(request.param))
    library = _library(source, 0)
    with mock_patch.dict(sys.modules, {"qualtran": None, "cirq": None, "qiskit": None}):
        record = resolve_component_program(
            library, 0, elliptic_curve_window_factories()
        )
    assert not verify_program_record(record)
    return source, record


@pytest.mark.parametrize("depth", (0, 2))
def test_original_stored_libraries_five_calls_operands_and_sdk_free_resolution(
    window_graph: tuple[Any, ProgramRecord], depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, graph = window_graph
    library = _library(source, depth)
    encoded = serialize_component_library(library)
    assert (
        serialize_component_library(deserialize_component_library(encoded)) == encoded
    )
    with monkeypatch.context() as patch:
        for sdk in ("qualtran", "cirq", "qiskit"):
            patch.setitem(sys.modules, sdk, None)
        resolved = (
            resolve_component_program(library, 0, elliptic_curve_window_factories())
            if depth
            else graph
        )
        assert resolved == graph
    assert len(graph.functions[0].blocks[0].operations) == 5
    assert _named_inputs(_native_trace(graph)) == _named_inputs(
        _source_trace(SourceCalls, source.decompose_bloq())
    )
    # Existing loose source type consistency admits the five direct word links.
    definitions = {function.name: function for function in graph.functions}
    addition_name = graph.functions[0].blocks[0].operations[2].callee
    assert addition_name is not None
    assert len(definitions[addition_name].blocks[0].operations) == 6
    leaves = [function for function in graph.functions if not function.blocks]
    assert len(leaves) == 1 and "point_table_adjoint_erase" in leaves[0].name
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(
            graph, ((0,),) * int(source.add_window_size) + ((1, 2, 3), (4, 5, 6))
        )


def measured_replacement(graph: ProgramRecord, window: int) -> ProgramRecord:
    """Replace the whole explicit erase wrapper, leaving every consumer intact."""
    table = windowed_point_table(3, window, ECPoint(0, 1, 3, 1))
    target = graph.functions[0].blocks[0].operations[3].callee
    assert target is not None
    native = _rename_program(qroam_measurement_erasure_program(table, (3,) * 3), target)
    old = next(function for function in graph.functions if function.name == target)
    assert old.registers == native.functions[0].registers
    retained = tuple(
        function
        for function in graph.functions
        if function.name != target and function.blocks
    )
    assert not set(function.name for function in retained).intersection(
        function.name for function in native.functions
    )
    result = replace(graph, functions=(*retained, *native.functions))
    assert result.functions[0] == graph.functions[0]
    assert not verify_program_record(result)
    return result


def test_actual_stored_lookup_reuses_original_primitive_operands_and_feedback(
    window_graph: tuple[Any, ProgramRecord],
) -> None:
    from tests.unit.test_qroam_clean_component import _check_original_forward

    source, graph = window_graph
    window = int(source.add_window_size)
    table = windowed_point_table(3, window, ECPoint(0, 1, 3, 1))
    child = graph.functions[0].blocks[0].operations[1].callee
    assert child is not None
    expected = qroam_clean_program(
        table, (3,) * 3, selection_widths=(window,), reserve_workspace=False
    )
    assert _named_definitions(graph, child) == _named_definitions(expected, "main")
    _check_original_forward(source.qrom)


def bind_window(graph: ProgramRecord, window: int) -> tuple[BoundRegisterProgram, int]:
    logical = window + 6
    inputs = (
        *((i,) for i in range(window)),
        tuple(range(window, window + 3)),
        tuple(range(window + 3, logical)),
    )
    ownership = tuple((wire, False) for wire in range(logical))
    provisional = bind_register_program(
        graph,
        inputs,
        clean=tuple(range(logical, logical + 200)),
        allow_workspace_outputs=True,
        input_ownership=ownership,
    )
    width = max(w for event in provisional.events for w in event.wires) + 1
    return (
        bind_register_program(
            graph,
            inputs,
            clean=tuple(range(logical, width)),
            allow_workspace_outputs=True,
            input_ownership=ownership,
        ),
        width,
    )


def test_typed_window_replaces_physical_path_all_addresses_complex_histories_and_work(
    window_graph: tuple[Any, ProgramRecord], monkeypatch: pytest.MonkeyPatch
) -> None:
    source, stored = window_graph
    window = int(source.add_window_size)
    graph = measured_replacement(stored, window)
    bound, width = bind_window(graph, window)
    ports = {
        "ctrl": tuple(range(window)),
        "x": tuple(range(window, window + 3)),
        "y": tuple(range(window + 3, window + 6)),
    }
    outputs = {
        "ctrl": tuple(w[0] for w in bound.outputs[:window]),
        "x": bound.outputs[window],
        "y": bound.outputs[window + 1],
    }
    rows = [
        dict(ctrl=address, x=x, y=y)
        for address, (x, y) in product(
            range(1 << window), ((0, 0), (0, 2), (0, 1), (2, 0))
        )
    ]
    initial = [pack(row, ports, width) for row in rows]
    bits = [
        tuple(word >> (width - i - 1) & 1 for i in range(width)) for word in initial
    ]
    actual, events = certified_and_channel_map(bound.circuit.instructions, tuple(bits))
    original, old_ports, _, _ = layout(
        EllipticCurveWindowAddition(3, ECPoint(0, 1, 3, 1), window)
    )
    old_inputs = [pack(row, old_ports, original.n_wires) for row in rows]
    old_bits = [
        tuple(word >> (original.n_wires - i - 1) & 1 for i in range(original.n_wires))
        for word in old_inputs
    ]
    reference, old_events = certified_and_channel_map(
        original.instructions, tuple(old_bits)
    )
    expected = []
    for row in reference:
        values = {
            name: sum(row[w] << (len(ws) - i - 1) for i, w in enumerate(ws))
            for name, ws in old_ports.items()
        }
        expected.append(pack(values, outputs, width))
        assert sum(
            bit << (original.n_wires - i - 1) for i, bit in enumerate(row)
        ) == pack(values, old_ports, original.n_wires)
    assert [
        sum(bit << (width - i - 1) for i, bit in enumerate(row)) for row in actual
    ] == expected
    assert len(set(expected)) == len(rows)
    delta = Counter(gate.name for gate in bound.circuit.instructions)
    delta.subtract(gate.name for gate in original.instructions)
    assert delta == Counter(h=120, measure=120, x=120, z=60, cx=-60)
    assert events == old_events + 120
    weights = [complex(i + 1, 7 - i) for i in range(len(rows))]
    norm = sum(abs(value) ** 2 for value in weights) ** 0.5
    before = {word: value / norm for word, value in zip(initial, weights, strict=True)}
    after = {word: value / norm for word, value in zip(expected, weights, strict=True)}
    with monkeypatch.context() as patch:
        for sdk in ("qualtran", "cirq", "qiskit"):
            patch.setitem(sys.modules, sdk, None)
        circuit = DynamicCircuit(width).append(
            FieldOperation(width, bound.circuit.instructions)
        )
        assert circuit.to_ir().instructions == bound.circuit.instructions
        for seed in (0, 19):
            assert execute_sparse_register(
                circuit.to_ir(), before, seed=seed
            ) == pytest.approx(after, abs=4e-12)


@pytest.mark.parametrize("example", ("_ec_add_r", "_ec_add_r_small"))
def test_complete_fixed_point_examples_remain_declarations(example: str) -> None:
    from qualtran._infra.bloq import DecomposeNotImplementedError

    module = import_module(PREFIX.rstrip("."))
    source = getattr(module, example).make()
    graph = resolve_component_program(
        _library(source, 0), 0, elliptic_curve_window_factories()
    )
    assert not graph.functions[0].blocks
    assert not verify_program_record(graph)
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()


def test_symbolic_window_declaration_and_malformed_original_fields() -> None:
    factory = elliptic_curve_window_factories()[PREFIX + "ECWindowAddR"]
    n, k = sympy.symbols("n k", positive=True, integer=True)
    arguments = {"n": n, "R": AffinePointArgument(0, 1, 3, 1), "add_window_size": k}
    declaration = factory("main", None, arguments)
    assert isinstance(declaration, FunctionRecord)
    assert declaration.registers == tuple(fixed_point_addition_signature(n, k))
    assert not declaration.blocks
    for missing in arguments:
        with pytest.raises(SerializationError):
            factory(
                "main",
                None,
                {name: value for name, value in arguments.items() if name != missing},
            )
    with pytest.raises(SerializationError, match="fields"):
        factory("main", None, {**arguments, "extra": 1})
    with pytest.raises(SerializationError, match="affine"):
        factory("main", None, {**arguments, "R": 1})
    with pytest.raises(SerializationError, match="signature"):
        factory("main", (), arguments)
