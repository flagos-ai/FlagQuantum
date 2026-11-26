"""Bind the complete Hubbard SELECT source file through native constructors."""

import sys
from hashlib import sha1
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest

from flagquantum.algorithms.primitives.hubbard_majorana import HubbardMajorana
from flagquantum.algorithms.primitives.hubbard_majorana_component import (
    hubbard_majorana_program,
)
from flagquantum.algorithms.primitives.hubbard_select import SelectHubbard
from flagquantum.algorithms.primitives.hubbard_select_component import (
    hubbard_select_program,
)
from flagquantum.algorithms.primitives.hubbard_spin_up import HubbardSpinUpZ
from flagquantum.algorithms.primitives.hubbard_spin_up_component import (
    hubbard_spin_up_program,
)
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_hubbard_selection_factories import (
    hubbard_selection_factories,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.errors import SerializationError
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multi_control_component import atomic_leaves

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def source_module() -> Any:
    import kickmix
    from qualtran.bloqs.chemistry.hubbard_model.qubitization import select_hubbard

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(select_hubbard.__file__).parent
    for filename, digest in (
        ("select_hubbard.py", "6d912f2ae1f63730cc296cfc35f68eb9ccbabb84"),
        ("select_hubbard_test.py", "c3307b5f562b8ac01e397ed6a7b1534088e7fbd4"),
        ("select_hubbard.ipynb", "c9b80d41b44f907f81ae5bfabcac647515e4cb63"),
    ):
        raw = (folder / filename).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return select_hubbard


def terminals(record: ProgramRecord) -> set[str]:
    return atomic_leaves(record) | {
        f.name for f in record.functions if f.registers == ()
    }


@pytest.mark.parametrize(
    "kind", ["SelectHubbard", "HubbardMajorannaOperator", "HubbardSpinUpZ"]
)
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_complete_source_constructor_call_layers_and_execution_path_replacement(
    source_module: Any, kind: str, controlled: bool, depth: int
) -> None:
    source = getattr(source_module, kind)(2, 2, control_val=1 if controlled else None)
    library = _library(source, depth)
    wire = serialize_component_library(library)
    assert serialize_component_library(deserialize_component_library(wire)) == wire
    record = resolve_component_program(library, 0, hubbard_selection_factories())
    assert verify_program_record(record) == ()
    compare_layers(record, source, terminals(record))
    operation: SelectHubbard | HubbardMajorana | HubbardSpinUpZ
    if kind == "SelectHubbard":
        operation = SelectHubbard(2, controlled)
        accepted = hubbard_select_program(operation)
    elif kind == "HubbardMajorannaOperator":
        majorana = HubbardMajorana(2, controlled=controlled)
        operation = majorana
        accepted = hubbard_majorana_program(majorana)
    else:
        spin = HubbardSpinUpZ(2, controlled)
        operation = spin
        accepted = hubbard_spin_up_program(spin)
    ports = []
    cursor = 0
    for _, width in operation.layout:
        ports.append(tuple(range(cursor, cursor + width)))
        cursor += width
    clean = tuple(range(cursor, cursor + operation.requirements()[0]))
    bound = bind_register_program(record, tuple(ports), clean=clean)
    expected = bind_register_program(accepted, tuple(ports), clean=clean)
    assert bound == expected
    assert bound.outputs == tuple(ports)


@pytest.mark.parametrize("kind", ["HubbardMajorannaOperator", "HubbardSpinUpZ"])
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_exact_source_generic_adjoint_constructor(
    source_module: Any, kind: str, controlled: bool, depth: int
) -> None:
    source = getattr(source_module, kind)(
        2, 2, control_val=1 if controlled else None
    ).adjoint()
    record = resolve_component_program(
        _library(source, depth), 0, hubbard_selection_factories()
    )
    assert verify_program_record(record) == ()
    compare_layers(record, source, terminals(record))


@pytest.mark.parametrize("axis", ["X", "Y"])
@pytest.mark.parametrize("control", [None, 1])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_majorana_axis_metadata_reaches_the_exact_child(
    source_module: Any, axis: str, control: int | None, depth: int
) -> None:
    source = source_module.HubbardMajorannaOperator(2, 2, axis, control_val=control)
    record = resolve_component_program(
        _library(source, depth), 0, hubbard_selection_factories()
    )
    compare_layers(record, source, terminals(record))


@pytest.mark.parametrize(
    "kind", ["SelectHubbard", "HubbardMajorannaOperator", "HubbardSpinUpZ"]
)
def test_singleton_metadata_does_not_acquire_an_identity_body(
    source_module: Any, kind: str
) -> None:
    source = getattr(source_module, kind)(1, 1)
    library = _library(source, 0)
    record = resolve_component_program(library, 0, hubbard_selection_factories())
    assert record.functions[0].registers == library.entries[0].declaration.registers
    assert not record.functions[0].blocks


@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_original_select_specialized_inverse_codec_failure_is_not_repaired(
    source_module: Any, controlled: bool, depth: int
) -> None:
    source = source_module.SelectHubbard(
        2, 2, control_val=1 if controlled else None
    ).adjoint()
    with pytest.raises(ValueError, match="Cannot serialize SpecializeOnCtrlBit.ONE"):
        _library(source, depth)


@pytest.mark.parametrize(
    "name",
    [
        "_sel_hubb",
        "_hubb_majoranna",
        "_hubb_majoranna_small",
        "_hubb_spin_up_z",
        "_hubb_spin_up_z_small",
    ],
)
def test_all_original_factories_and_symbolic_logical_declarations(
    source_module: Any, name: str
) -> None:
    source = getattr(source_module, name).make()
    library = _library(source, 0)
    record = resolve_component_program(library, 0, hubbard_selection_factories())
    assert record.functions[0].registers == library.entries[0].declaration.registers
    assert verify_program_record(record) == ()
    if name in {"_hubb_majoranna", "_hubb_spin_up_z"}:
        assert not record.functions[0].blocks
    else:
        compare_layers(record, source, terminals(record))


def test_zero_polarity_source_boundaries_and_native_validator(
    source_module: Any,
) -> None:
    with pytest.raises(NotImplementedError, match="control_val=0 not supported"):
        source_module.SelectHubbard(2, 2, control_val=0)
    source = source_module.HubbardMajorannaOperator(2, 2, control_val=0)
    record = resolve_component_program(
        _library(source, 0), 0, hubbard_selection_factories()
    )
    assert not record.functions[0].blocks
    with pytest.raises(KeyError, match="control"):
        source.decompose_bloq()
    spin = source_module.HubbardSpinUpZ(2, 2, control_val=0)
    record = resolve_component_program(
        _library(spin, 0), 0, hubbard_selection_factories()
    )
    compare_layers(record, spin, terminals(record))
    with pytest.raises(ValueError, match="zero control executes a positive"):
        hubbard_spin_up_program(HubbardSpinUpZ(2, True, control_value=0))


def test_source_free_constructors_and_exact_recipe_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for module in ("qualtran", "cirq", "kickmix", "quimb"):
        monkeypatch.setitem(sys.modules, module, None)
    factories = hubbard_selection_factories()
    for key, factory in factories.items():
        if key.startswith(
            "qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard."
        ):
            record = factory("main", None, {"x_dim": 2, "y_dim": 2})
            assert (
                isinstance(record, ProgramRecord)
                and verify_program_record(record) == ()
            )
            ports = []
            cursor = 0
            for register in record.functions[0].registers or ():
                width = int(register.bitsize)
                ports.append(tuple(range(cursor, cursor + width)))
                cursor += width
            bound = bind_register_program(
                record, tuple(ports), clean=tuple(range(cursor, cursor + 4))
            )
            assert bound.outputs == tuple(ports)
            assert all(gate.matrix is None for gate in bound.circuit.instructions)
    inverse = hubbard_selection_factories()["qualtran._infra.adjoint.Adjoint"]
    assert isinstance(record, ProgramRecord)
    with pytest.raises(SerializationError, match="Missing exact"):
        inverse("main", None, {"subbloq": record.functions[0]})


def test_constructor_metadata_and_signature_refusals() -> None:
    factory = hubbard_selection_factories()[
        "qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard.SelectHubbard"
    ]
    with pytest.raises(NotImplementedError, match="x_dim=y_dim"):
        factory("main", None, {"x_dim": 2, "y_dim": 3})
    with pytest.raises(SerializationError, match="control must"):
        factory("main", None, {"x_dim": 2, "y_dim": 2, "control_val": 2})
    with pytest.raises(SerializationError, match="signature"):
        factory("main", (), {"x_dim": 2, "y_dim": 2})
