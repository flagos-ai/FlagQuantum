"""Original ECC phase roots retain point powers, terminal gaps and dirty exits."""

import sys
from collections.abc import Callable
from hashlib import sha1
from importlib import import_module
from itertools import product
from pathlib import Path
from typing import Any
from unittest.mock import patch as mock_patch

import numpy as np
import pytest

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives.elliptic_curve_fixed_point import (
    fixed_point_addition_signature,
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
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem._elliptic_curve_phase_library import (
    elliptic_curve_phase_factories,
)
from flagquantum.errors import SerializationError
from tests.unit.ec_stage_conformance import pack
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_elliptic_curve_window_components import (
    bind_window,
    measured_replacement,
)
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_projectile_nu_library import _named_inputs
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
PHASE = "qualtran.bloqs.cryptography.ecc.ec_phase_estimate_r.ECPhaseEstimateR"
DOUBLE = "qualtran.bloqs.cryptography.ecc.find_ecc_private_key.FindECCPrivateKey"


@pytest.fixture(scope="module", autouse=True)
def frozen_phase_files() -> None:
    module = import_module("qualtran.bloqs.cryptography.ecc.ec_phase_estimate_r")
    assert module.__file__ is not None
    folder = Path(module.__file__).parent
    for name, expected in (
        ("ec_phase_estimate_r.py", "eb991c19aba42a307c66e6a5839331723da6f581"),
        ("ec_phase_estimate_r_test.py", "070181c947460d1d23221a7aa6826160008fa046"),
        ("find_ecc_private_key.py", "f6129af801f69d925c763a0ff7a69f03667e82a2"),
        ("find_ecc_private_key_test.py", "ffc58fd99b7060aea37643f65ceb966fa1b7497c"),
        ("../_factoring_shims.py", "29f46f638125828d4ae3cc3aa15ffd49d06e830e"),
        ("ec_point.py", "eb885c54c88d663cbec19f44cdcbbb5e9cf424ef"),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == expected


@pytest.fixture(
    scope="module", params=("_ec_pe", "_ec_pe_small", "_ecc", "window", "double")
)
def phase_graph(request: pytest.FixtureRequest) -> tuple[Any, ProgramRecord]:
    from qualtran.bloqs.cryptography.ecc.ec_phase_estimate_r import (
        ECPhaseEstimateR,
        _ec_pe,
        _ec_pe_small,
    )
    from qualtran.bloqs.cryptography.ecc.ec_point import ECPoint
    from qualtran.bloqs.cryptography.ecc.find_ecc_private_key import (
        FindECCPrivateKey,
        _ecc,
    )

    sources: dict[str, Callable[[], Any]] = {
        "_ec_pe": _ec_pe.make,
        "_ec_pe_small": _ec_pe_small.make,
        "_ecc": _ecc.make,
        "window": lambda: ECPhaseEstimateR(3, ECPoint(0, 1, 3, 1), 3),
        "double": lambda: FindECCPrivateKey(
            3, ECPoint(0, 1, 3, 1), ECPoint(0, 2, 3, 1), 3
        ),
    }
    source = sources[str(request.param)]()
    library = _library(source, 0)
    with mock_patch.dict(sys.modules, {"qualtran": None, "cirq": None, "qiskit": None}):
        record = resolve_component_program(library, 0, elliptic_curve_phase_factories())
    assert not verify_program_record(record)
    return source, record


@pytest.mark.parametrize("depth", (0, 1))
def test_original_examples_libraries_and_complete_source_call_order_without_sdks(
    phase_graph: tuple[Any, ProgramRecord], depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualtran._infra.bloq import DecomposeTypeError

    source, graph = phase_graph
    library = _library(source, depth)
    encoded = serialize_component_library(library)
    assert (
        serialize_component_library(deserialize_component_library(encoded)) == encoded
    )
    with monkeypatch.context() as patch:
        for sdk in ("qualtran", "cirq", "qiskit"):
            patch.setitem(sys.modules, sdk, None)
        resolved = (
            resolve_component_program(library, 0, elliptic_curve_phase_factories())
            if depth
            else graph
        )
        assert resolved == graph
    if not graph.functions[0].blocks:
        with pytest.raises(DecomposeTypeError, match="symbolic"):
            source.decompose_bloq()
        return
    assert _named_inputs(_native_trace(graph)) == _named_inputs(
        _source_trace(SourceCalls, source.decompose_bloq())
    )
    if type(source).__name__ == "FindECCPrivateKey":
        root = graph.functions[0]
        assert root.registers == () and len(root.blocks[0].operations) == 6
        definitions = {function.name: function for function in graph.functions}
        for operation in root.blocks[0].operations[-2:]:
            free = definitions[operation.callee or ""]
            assert free.blocks[0].operations[0].workspace_dirty is True
    else:
        assert (
            len(graph.functions[0].blocks[0].operations)
            == int(source.n) + int(source.num_windows) + 1
        )


def test_original_terminal_depth_failure_and_opaque_scalar_are_not_repaired() -> None:
    from qualtran._infra.composite_bloq import BloqError
    from qualtran.bloqs.cryptography.ecc.ec_phase_estimate_r import _ec_pe_small

    source = _ec_pe_small.make()
    with pytest.raises(BloqError, match="During finalization"):
        _library(source, 2)
    graph = resolve_component_program(
        _library(source, 0), 0, elliptic_curve_phase_factories()
    )
    assert sum(not function.blocks for function in graph.functions) == 4
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(graph, ((0, 1, 2), (3, 4, 5)), clean=tuple(range(6, 40)))


def test_original_point_powers_both_control_shapes_and_ignored_scalar_mul_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualtran.bloqs.cryptography.ecc.ec_phase_estimate_r import ECPhaseEstimateR
    from qualtran.bloqs.cryptography.ecc.ec_point import ECPoint

    from flagquantum.ecosystem import _elliptic_curve_phase_library as owner

    observed: list[dict[str, Any]] = []

    def child(name: str, registers: Any, arguments: Any) -> FunctionRecord:
        observed.append(dict(arguments))
        return FunctionRecord(
            name,
            (),
            registers=tuple(
                fixed_point_addition_signature(
                    arguments["n"], arguments.get("add_window_size")
                )
            ),
        )

    prefix = "qualtran.bloqs.cryptography.ecc.ec_add_r."
    monkeypatch.setattr(
        owner,
        "elliptic_curve_window_factories",
        lambda: {prefix + "ECAddR": child, prefix + "ECWindowAddR": child},
    )
    for window in (1, 2):
        observed.clear()
        source = ECPhaseEstimateR(4, ECPoint(0, 1, 3, 2), window, 0)
        graph = resolve_component_program(
            _library(source, 0), 0, owner.elliptic_curve_phase_factories()
        )
        assert _named_inputs(_native_trace(graph)) == _named_inputs(
            _source_trace(SourceCalls, source.decompose_bloq())
        )
        expected = [
            node.bloq.R
            for node, _, _ in sorted(
                source.decompose_bloq().iter_bloqsoqs(), key=lambda row: row[0].i
            )
            if hasattr(node.bloq, "R")
        ]
        assert [row["R"] for row in observed] == [
            AffinePointArgument(p.x, p.y, p.mod, p.curve_a) for p in expected
        ]
        assert all(("mul_window_size" in row) == (window != 1) for row in observed)


def test_selected_stored_window_child_complex_columns_all_histories_and_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualtran.bloqs.cryptography.ecc.ec_phase_estimate_r import ECPhaseEstimateR
    from qualtran.bloqs.cryptography.ecc.ec_point import ECPoint

    source = ECPhaseEstimateR(3, ECPoint(0, 1, 3, 1), 3)
    graph = resolve_component_program(
        _library(source, 0), 0, elliptic_curve_phase_factories()
    )
    target = graph.functions[0].blocks[0].operations[3].callee
    assert target is not None
    child = _rename_program(ProgramRecord(_named_definitions(graph, target)), "main")
    executable = measured_replacement(child, 3)
    bound, width = bind_window(executable, 3)
    incoming = {"ctrl": (0, 1, 2), "x": (3, 4, 5), "y": (6, 7, 8)}
    outgoing = {
        "ctrl": tuple(word[0] for word in bound.outputs[:3]),
        "x": bound.outputs[3],
        "y": bound.outputs[4],
    }
    original: Any = next(
        node.bloq
        for node, _, _ in source.decompose_bloq().iter_bloqsoqs()
        if type(node.bloq).__name__ == "ECWindowAddR"
    )
    rows = [
        dict(ctrl=c, x=x, y=y)
        for c, (x, y) in product(range(8), ((0, 0), (0, 2), (0, 1), (2, 0)))
    ]
    before_words, after_words = [], []
    for row in rows:
        result = original.on_classical_vals(
            ctrl=np.array([row["ctrl"] >> i & 1 for i in (2, 1, 0)]),
            x=row["x"],
            y=row["y"],
        )
        before_words.append(pack(row, incoming, width))
        after_words.append(
            pack(
                dict(ctrl=row["ctrl"], x=int(result["x"]), y=int(result["y"])),
                outgoing,
                width,
            )
        )
    columns = tuple(
        tuple(word >> (width - i - 1) & 1 for i in range(width))
        for word in before_words
    )
    outputs, measurements = certified_and_channel_map(
        bound.circuit.instructions, columns
    )
    assert [
        sum(bit << (width - i - 1) for i, bit in enumerate(row)) for row in outputs
    ] == after_words
    assert measurements == sum(
        gate.name == "measure" for gate in bound.circuit.instructions
    )
    weights = [complex(i + 1, 13 - i) for i in range(len(rows))]
    norm = sum(abs(value) ** 2 for value in weights) ** 0.5
    before = {
        word: value / norm for word, value in zip(before_words, weights, strict=True)
    }
    after = {
        word: value / norm for word, value in zip(after_words, weights, strict=True)
    }
    assert len(after) == len(rows)
    with monkeypatch.context() as patch:
        for sdk in ("qualtran", "cirq", "qiskit"):
            patch.setitem(sys.modules, sdk, None)
        circuit = DynamicCircuit(width).append(
            FieldOperation(width, bound.circuit.instructions)
        )
        for seed in (0, 19):
            assert execute_sparse_register(
                circuit.to_ir(), before, seed=seed
            ) == pytest.approx(after, abs=4e-12)


def test_nondividing_window_and_invalid_metadata_remain_rejected() -> None:
    from qualtran.bloqs.cryptography.ecc.ec_phase_estimate_r import ECPhaseEstimateR
    from qualtran.bloqs.cryptography.ecc.ec_point import ECPoint

    source = ECPhaseEstimateR(3, ECPoint(0, 1, 3, 1), 2)
    with pytest.raises(IndexError):
        source.decompose_bloq()
    factory = elliptic_curve_phase_factories()[PHASE]
    fields = {"n": 3, "point": AffinePointArgument(0, 1, 3, 1), "add_window_size": 2}
    with pytest.raises(ValueError, match="dividing"):
        factory("main", None, fields)
    for key in ("n", "point"):
        with pytest.raises(SerializationError):
            factory(
                "main",
                None,
                {name: value for name, value in fields.items() if name != key},
            )
    with pytest.raises(SerializationError, match="signature"):
        factory("main", (), fields)
    with pytest.raises(SerializationError, match="affine"):
        factory("main", None, {**fields, "point": 1})


def test_source_double_graph_does_not_apply_model_only_consistency_guard() -> None:
    from qualtran.bloqs.cryptography.ecc.ec_point import ECPoint
    from qualtran.bloqs.cryptography.ecc.find_ecc_private_key import FindECCPrivateKey

    source = FindECCPrivateKey(3, ECPoint(0, 1, 3, 1), ECPoint(0, 1, 5, 1))
    with pytest.raises(ValueError, match="moduli"):
        _ = source.mod
    graph = elliptic_curve_phase_factories()[DOUBLE](
        "main",
        None,
        {
            "n": 3,
            "base_point": AffinePointArgument(0, 1, 3, 1),
            "public_key": AffinePointArgument(0, 1, 5, 1),
        },
    )
    assert isinstance(graph, ProgramRecord)
    assert _named_inputs(_native_trace(graph)) == _named_inputs(
        _source_trace(SourceCalls, source.decompose_bloq())
    )
