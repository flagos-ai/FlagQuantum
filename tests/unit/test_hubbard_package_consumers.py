"""Consume all five Hubbard package exports at the owned native boundary."""

import builtins
from dataclasses import asdict
from typing import Any

import pytest

from flagquantum.algorithms.primitives.hubbard_majorana import HubbardMajorana
from flagquantum.algorithms.primitives.hubbard_prepare import PrepareHubbard
from flagquantum.algorithms.primitives.hubbard_select import SelectHubbard
from flagquantum.algorithms.primitives.hubbard_spin_up import HubbardSpinUpZ
from flagquantum.algorithms.primitives.hubbard_walk import HubbardWalk
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.registers.layout.partition import partition_register
from flagquantum.dynamic import DynamicCircuit

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "kind,controlled",
    tuple(
        (kind, controlled)
        for kind in ("select", "majorana_x", "majorana_y", "spin")
        for controlled in (False, True)
    )
    + (("prepare", False), ("walk", False)),
)
def test_actual_package_constructor_library_and_owned_append(
    kind: str, controlled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualtran.bloqs.chemistry.hubbard_model import qubitization as source
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    cv = 1 if controlled else None
    if kind == "prepare":
        original = source.PrepareHubbard(2, 2, 1, 2)
    elif kind == "select":
        original = source.SelectHubbard(2, 2, control_val=cv)
    elif kind.startswith("majorana"):
        original = source.HubbardMajorannaOperator(
            2, 2, "X" if kind.endswith("x") else "Y", control_val=cv
        )
    elif kind == "spin":
        original = source.HubbardSpinUpZ(2, 2, control_val=cv)
    else:
        original = source.get_walk_operator_for_hubbard_model(2, 2, 1, 2)
    proto = bloqs_to_proto(original, max_depth=0)
    library = deserialize_component_library(proto.SerializeToString())
    encoded = serialize_component_library(library)
    restored = type(proto).FromString(encoded)
    assert restored == proto
    assert bloqs_from_proto(restored)[0] == original
    assert (
        serialize_component_library(deserialize_component_library(encoded)) == encoded
    )
    declaration = library.entries[0].declaration
    values = {argument.name: argument.value for argument in declaration.arguments}

    def integer(name: str) -> int:
        value = values[name]
        assert type(value) is int
        return value

    if kind == "walk":
        block = values["block_encoding"]
        # The constructor root refers to owned child declarations by integer ID.
        assert block is not None
        prepare_entry = next(
            entry
            for entry in library.entries
            if entry.declaration.name.endswith(".PrepareHubbard")
        )
        prep_values = {
            argument.name: argument.value
            for argument in prepare_entry.declaration.arguments
        }
        assert prep_values["x_dim"] == prep_values["y_dim"] == 2
        d, t, u = (prep_values[name] for name in ("x_dim", "t", "u"))
        assert type(d) is int and type(t) is int and type(u) is int
        native: Any = HubbardWalk.from_lattice(d, t, u)
        layout = native.source_layout
    else:
        assert integer("x_dim") == integer("y_dim")
        d = integer("x_dim")
        enabled = values.get("control_val") is not None
        if kind == "prepare":
            assert values["t"] == 1 and values["u"] == 2
            native = PrepareHubbard(d, 1, 2)
        elif kind == "select":
            native = SelectHubbard(d, enabled)
        elif kind.startswith("majorana"):
            pauli = values["gate"]
            assert pauli in ("X", "Y")
            native = HubbardMajorana(d, "X" if pauli == "X" else "Y", enabled)
        else:
            native = HubbardSpinUpZ(d, enabled)
        layout = native.layout
    assert declaration.registers is not None
    declared_layout = tuple(
        (register.name, register.total_bits()) for register in declaration.registers
    )
    if kind == "walk":
        # The original walk forms its signature through a set of named registers.
        # Native binding is by name, preserving each captured physical operand.
        assert dict(layout) == dict(declared_layout)
        physical_layout = (*declared_layout, ("temp", 2))
    else:
        assert layout == declared_layout
        physical_layout = native.layout
    expected = get_cost_value(original, QECGatesCost()).asdict()
    peak = get_cost_value(original, QubitCount())
    original_import = builtins.__import__

    def owned_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq", "kickmix"}:
            raise AssertionError(f"Owned Hubbard component imported {name}")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", owned_import)
        model = estimate_component_resources(native.source_resource_model())
        logical = sum(width for _, width in native.layout)
        work, _ = native.requirements()
        ports = partition_register(tuple(range(logical)), physical_layout)
        workspace = tuple(range(logical, logical + work))
        result = (
            DynamicCircuit(logical + work)
            .append(native.make_on(ports, workspace=workspace))
            .to_ir()
            .validate()
        )
        assert result.instructions
    assert {
        key: value for key, value in asdict(model.gates).items() if value
    } == expected
    assert model.peak_logical_qubits == peak
