"""Replay complete source QREF fixtures through owned metadata conversion."""

import builtins
import os
from collections.abc import Callable
from hashlib import sha1
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from flagquantum.compiler.formats.qref_resources import QrefComponent, qref_component
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.ecosystem._qref_source import qref_from_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def source() -> Any:
    pytest.importorskip("qref")
    pytest.importorskip("qualtran")
    import cirq
    import kickmix
    import qualtran.qref_interop._bloq_to_qref as source

    assert version("qualtran") == "0.7.0" and version("qref") == "0.11.0"
    assert cirq.__version__.startswith(
        os.environ.get("FQ_FIXED_GATE_SOURCE_CIRQ", "1.7") + "."
    )
    assert getattr(kickmix, "__version__", None) == "0.1" and hasattr(
        kickmix, "Simulator"
    )
    parent = Path(source.__file__).parent
    for name, expected in (
        ("_bloq_to_qref.py", "aab3422ba170c0c909ee152d46537f78335829b8"),
        ("_bloq_to_qref_test.py", "e7092a50ce00daf63488a419b0bd8609e937fc38"),
    ):
        data = (parent / name).read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == expected
    return source


@pytest.mark.parametrize(
    "example",
    (
        "add_oop_large",
        "black_box_lcu_block",
        "lcu_block",
        "df_one_body",
        "df_block_encoding",
        "qrom_symb",
        "rsa_pe",
    ),
)
@pytest.mark.parametrize("mode", ("leaf", "calls", "composite"))
def test_complete_large_source_fixtures_keep_schema_topology_and_resources(
    source: Any, example: str, mode: str
) -> None:
    fixtures = import_module("qualtran.qref_interop._bloq_to_qref_test")
    original = getattr(fixtures, "_" + example).make()
    from qref.verification import verify_topology
    from qualtran._infra.bloq import DecomposeTypeError

    if mode == "composite":
        if example in {"qrom_symb", "rsa_pe"}:
            with pytest.raises(DecomposeTypeError, match="symbolic|parameterized"):
                original.decompose_bloq()
            assert (
                qref_from_source(original, decomposition_rules=True).model_dump_json()
                == source.bloq_to_qref(
                    original, decomposition_rules=True
                ).model_dump_json()
            )
            return
        original = original.decompose_bloq()
    counted = mode == "calls"
    expected = source.bloq_to_qref(original, from_callgraph=counted)
    actual = qref_from_source(original, from_callgraph=counted)
    assert actual.model_dump_json() == expected.model_dump_json()
    assert verify_topology(actual)


@pytest.mark.parametrize("size", (0, 1, 2, 12))
def test_declared_array_port_order_is_the_source_lexicographic_order(
    source: Any, size: int
) -> None:
    from qualtran._infra.data_types import QBit
    from qualtran._infra.registers import Register as SourceRegister
    from qualtran.bloqs.basic_gates.on_each import OnEach
    from qualtran.bloqs.basic_gates.x_basis import XGate

    original = OnEach(size, XGate())
    expected = source.bloq_to_qref(original)
    native = qref_component(
        QrefComponent("OnEach", "OnEach", {"t": 0, "clifford": size, "rotations": 0}),
        (Register("q", quantum_bit_type(), (size,)),),
    )
    source_ports = source._ports_from_register(
        SourceRegister("q", QBit(), shape=(size,))
    )
    assert [port.model_dump() for port in native.program.ports] == [
        port.model_dump() for port in source_ports
    ]
    assert qref_from_source(original).model_dump() == expected.model_dump()


@pytest.mark.parametrize("rules", (False, True, "match", "miss"))
def test_selective_expansion_retains_disconnected_scalar_and_source_instance_ids(
    source: Any, rules: bool | str
) -> None:
    from qualtran._infra.quantum_graph import BloqInstance
    from qualtran.bloqs.basic_gates.cnot import CNOT
    from qualtran.bloqs.basic_gates.su2_rotation import SU2RotationGate

    original = SU2RotationGate(0.1, 0.2, 0.0)
    selected = (
        rules
        if isinstance(rules, bool)
        else (SU2RotationGate if rules == "match" else CNOT,)
    )
    for obj in (original, BloqInstance(original, 17)):
        actual = qref_from_source(obj, decomposition_rules=selected)
        expected = source.bloq_to_qref(obj, decomposition_rules=selected)
        assert actual.model_dump() == expected.model_dump()
        if rules is True or rules == "match":
            assert any(child.type == "GlobalPhase" for child in actual.program.children)
    with pytest.raises(ValueError, match="BloqInstance cannot"):
        qref_from_source(BloqInstance(original, 17), from_callgraph=True)


@pytest.mark.parametrize("kind", ("float", "literal", "none", "integer", "symbol"))
def test_resource_scalar_projection_matches_original_helper(
    source: Any, kind: str
) -> None:
    from sympy import Integer, Symbol

    values = {
        "float": 0.5,
        "literal": "count",
        "none": None,
        "integer": Integer(3),
        "symbol": Symbol("n"),
    }
    value = values[kind]
    actual = qref_component(
        QrefComponent("Value", "Value", {"t": value, "clifford": 0, "rotations": 0})
    )
    assert next(
        resource.value for resource in actual.program.resources if resource.name == "t"
    ) == source._ensure_primitive_type(value)


def test_classical_measurement_and_cirq_leaf_labels_have_source_metadata(
    source: Any,
) -> None:
    cirq = pytest.importorskip("cirq")
    from qualtran._infra.quantum_graph import BloqInstance
    from qualtran.bloqs.basic_gates.z_basis import MeasureZ
    from qualtran.cirq_interop._cirq_to_bloq import CirqGateAsBloq

    for original in (
        MeasureZ(),
        CirqGateAsBloq(cirq.X),
        BloqInstance(CirqGateAsBloq(cirq.X), 23),
    ):
        assert (
            qref_from_source(original).model_dump()
            == source.bloq_to_qref(original).model_dump()
        )


def test_explicit_empty_and_disconnected_occurrences_are_validated(source: Any) -> None:
    from flagquantum.compiler.formats.legacy_component_library import (
        ComponentDeclaration,
        ComponentLibrary,
        ComponentLibraryEntry,
        ComponentOccurrence,
    )
    from flagquantum.compiler.formats.qref_resources import qref_component_graph
    from flagquantum.errors import SerializationError

    library = ComponentLibrary(
        entries=(ComponentLibraryEntry(0, ComponentDeclaration("Empty", registers=())),)
    )
    metadata = {
        0: QrefComponent("Empty", "Empty", {"t": 0, "clifford": 0, "rotations": 0})
    }
    actual = qref_component_graph(
        library, 0, metadata, decomposition_rules=True, decompositions={0: ()}
    )
    assert actual.program.type == "CompositeBloq" and not actual.program.children
    with pytest.raises(SerializationError, match="occurrences differ"):
        qref_component_graph(
            library,
            0,
            metadata,
            decomposition_rules=True,
            decompositions={0: (ComponentOccurrence(1, 0), ComponentOccurrence(1, 0))},
        )


def test_captured_large_schema_roundtrip_and_consumer_need_no_source_sdk(
    source: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qref.schema_v1 import SchemaV1
    from qref.verification import verify_topology

    fixtures = import_module("qualtran.qref_interop._bloq_to_qref_test")
    original = fixtures._df_block_encoding.make()
    schema = qref_from_source(original, from_callgraph=True)
    payload = schema.model_dump_json()
    imported: Callable[..., ModuleType] = builtins.__import__

    def native_only(name: str, *args: Any, **kwargs: Any) -> ModuleType:
        if name.split(".")[0] == "qualtran":
            raise AssertionError(f"QREF consumer imported source SDK {name}")
        return imported(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", native_only)
    restored = SchemaV1.model_validate_json(payload)
    assert restored.model_dump() == schema.model_dump()
    assert verify_topology(restored)


def test_expanded_resources_use_actual_body_instead_of_analytic_parent_override(
    source: Any,
) -> None:
    pytest.importorskip("numpy")
    from numpy import object_
    from qualtran._infra.bloq import Bloq
    from qualtran._infra.composite_bloq import BloqBuilder, SoquetT
    from qualtran._infra.registers import Signature
    from qualtran.bloqs.basic_gates.x_basis import XGate
    from qualtran.cirq_interop.t_complexity_protocol import TComplexity

    class AnalyticPair(Bloq):
        @property
        def signature(self) -> Signature:
            return Signature.build(q=1)

        def t_complexity(self) -> TComplexity:
            return TComplexity(clifford=17)

        def build_composite_bloq(
            self, bb: BloqBuilder, **soqs: SoquetT[object_]
        ) -> dict[str, SoquetT[object_]]:
            values = bb.add_d(XGate(), q=soqs["q"])
            return bb.add_d(XGate(), **values)

    original = AnalyticPair()
    for expanded, cliffords in ((False, 17), (True, 2)):
        actual = qref_from_source(original, decomposition_rules=expanded)
        expected = source.bloq_to_qref(original, decomposition_rules=expanded)
        assert actual.model_dump_json() == expected.model_dump_json()
        assert (
            next(r.value for r in actual.program.resources if r.name == "clifford")
            == cliffords
        )


def test_named_symbolic_constants_are_not_added_as_free_port_parameters(
    source: Any,
) -> None:
    from qualtran.bloqs.basic_gates.on_each import OnEach
    from qualtran.bloqs.basic_gates.x_basis import XGate
    from sympy import pi

    original = OnEach(pi, XGate())
    actual = qref_from_source(original)
    expected = source.bloq_to_qref(original)
    assert actual.model_dump_json() == expected.model_dump_json()
    assert actual.program.input_params == []
