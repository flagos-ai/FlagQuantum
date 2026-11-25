"""Full symbolic potential graphs retain exact source edges and static banks."""

from collections import Counter
from typing import Any

import numpy as np
import pytest
import sympy

from flagquantum.compiler.formats.grid_potential_graph import (
    bind_grid_pair_potential_library,
    grid_pair_potential_library,
)
from flagquantum.compiler.formats.legacy_component_library import (
    ComponentLibrary,
    ComponentOccurrence,
    LibraryBoundary,
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.core.ir.program import verify_program_record
from flagquantum.ecosystem._grid_potential_component_library import (
    grid_potential_library_factories,
)
from flagquantum.errors import SerializationError
from tests.unit.test_grid_potential_source import _children, _fingerprint, _pair
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_subtraction_foreign_file import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_N, _P, _T = sympy.symbols("n p t")


def _assert_source_graph(source: Any, library: ComponentLibrary) -> None:
    from qualtran import LeftDangle, RightDangle

    original = source.decompose_bloq()
    entries = {entry.identifier: entry for entry in library.entries}
    root = library.entries[0]
    assert root.declaration.registers == tuple(native_signature(source.signature))
    occurrences = {
        port.owner.instance_id: port.owner
        for connection in root.connections
        for port in (connection.source, connection.destination)
        if isinstance(port.owner, ComponentOccurrence)
    }
    assert len(occurrences) == len(original.bloq_instances) == 26
    for instance in original.bloq_instances:
        declaration = entries[occurrences[instance.i].definition_id].declaration
        assert (
            declaration.name
            == type(instance.bloq).__module__ + "." + type(instance.bloq).__name__
        )
        assert declaration.registers == tuple(native_signature(instance.bloq.signature))
        arguments = {
            argument.name: argument.value for argument in declaration.arguments
        }
        if type(instance.bloq).__name__ in {"Allocate", "Free"}:
            assert bool(arguments.get("dirty", False)) == instance.bloq.dirty
        if type(instance.bloq).__name__ == "QROM":
            data = arguments["data_or_shape"]
            assert isinstance(data, (tuple, list, np.ndarray))
            for expected, actual in zip(instance.bloq.data, data, strict=True):
                np.testing.assert_array_equal(expected, actual)

    def source_port(port: Any) -> tuple[Any, str, tuple[int, ...]]:
        owner = (
            "input"
            if port.binst == LeftDangle
            else "output" if port.binst == RightDangle else port.binst.i
        )
        return owner, port.reg.name, port.idx

    def native_port(port: Any) -> tuple[Any, str, tuple[int, ...]]:
        owner = (
            "input"
            if port.owner is LibraryBoundary.INPUT
            else (
                "output"
                if port.owner is LibraryBoundary.OUTPUT
                else port.owner.instance_id
            )
        )
        return owner, port.register.name, port.index

    assert Counter(
        (source_port(edge.left), source_port(edge.right))
        for edge in original.connections
    ) == Counter(
        (native_port(edge.source), native_port(edge.destination))
        for edge in root.connections
    )
    assert Counter(
        occurrence.definition_id for occurrence in occurrences.values()
    ) == dict(root.counts)


@pytest.mark.parametrize(
    "width,poly,target",
    ((_N, 3, 5), (1, _P, 5), (_N, _P, _T), (2 * _N + 1, _P + 1, _T + 2)),
)
def test_entire_symbolic_source_graph_and_dataset_without_unsupported_protocol(
    width: Any, poly: Any, target: Any
) -> None:
    from qualtran.bloqs.chemistry.trotter.grid_ham.potential import PairPotential

    data = _pair().qrom_data
    source = PairPotential(width, data, poly_bitsize=poly, inv_sqrt_bitsize=target)
    library = grid_pair_potential_library(
        {
            "bitsize": width,
            "qrom_data": data,
            "poly_bitsize": poly,
            "inv_sqrt_bitsize": target,
        }
    )
    _assert_source_graph(source, library)
    # Original and native legacy protocol refusal are explicit; neither falls
    # back to pickle nor drops symbolic arguments to emit incomplete bytes.
    with pytest.raises(ValueError, match="Object arrays cannot be saved"):
        _library(source, 1)
    with pytest.raises(SerializationError, match="Invalid array argument"):
        serialize_component_library(library)


@pytest.mark.parametrize("rows", (5, 16))
def test_partial_and_complete_binding_preserve_every_bank_then_rebuild_exact_source_program(
    rows: int,
) -> None:
    from qualtran.bloqs.chemistry.trotter.grid_ham.potential import PairPotential

    data = tuple(tuple((row + index) % 8 for index in range(rows)) for row in range(4))
    graph = grid_pair_potential_library(
        {
            "bitsize": _N,
            "poly_bitsize": _P,
            "inv_sqrt_bitsize": _T,
            "qrom_data": data,
            "label": "Coulomb",
        }
    )
    partial = bind_grid_pair_potential_library(graph, {"n": 1})
    _assert_source_graph(
        PairPotential(1, data, poly_bitsize=_P, inv_sqrt_bitsize=_T, label="Coulomb"),
        partial,
    )
    sequential = bind_grid_pair_potential_library(partial, {"p": 3, "t": 5})
    direct = bind_grid_pair_potential_library(graph, {"n": 1, "p": 3, "t": 5})
    source = PairPotential(1, data, poly_bitsize=3, inv_sqrt_bitsize=5, label="Coulomb")
    _assert_source_graph(source, sequential)
    assert serialize_component_library(direct) == serialize_component_library(
        sequential
    )
    restored = deserialize_component_library(serialize_component_library(direct))
    actual = resolve_component_program(restored, 0, grid_potential_library_factories())
    expected = resolve_component_program(_library(source, 2), 0, _children())
    assert verify_program_record(actual) == ()
    assert _fingerprint(actual) == _fingerprint(expected)
    root = {
        argument.name: argument.value
        for argument in restored.entries[0].declaration.arguments
    }
    np.testing.assert_array_equal(root["qrom_data"], data)


def test_data_snapshots_and_sdk_free_symbolic_graph_and_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    data = [list(row) for row in _pair().qrom_data]
    expected = tuple(tuple(row) for row in data)
    graph = grid_pair_potential_library(
        {"bitsize": _N, "poly_bitsize": _P, "qrom_data": data, "inv_sqrt_bitsize": 5}
    )
    data[0][1] ^= 1
    root = {
        argument.name: argument.value
        for argument in graph.entries[0].declaration.arguments
    }
    np.testing.assert_array_equal(root["qrom_data"], expected)
    original = builtins.__import__

    def no_source(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq"}:
            raise ImportError(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_source)
    bound = bind_grid_pair_potential_library(graph, {"n": 1, "p": 3})
    assert sum(count for _, count in bound.entries[0].counts) == 26
    actual = resolve_component_program(bound, 0, grid_potential_library_factories())
    assert verify_program_record(actual) == ()


def test_binding_does_not_silently_ignore_names_or_convert_symbolic_literals() -> None:
    graph = grid_pair_potential_library(
        {"bitsize": _N, "poly_bitsize": _P, "qrom_data": _pair().qrom_data}
    )
    with pytest.raises(SerializationError, match="known symbols"):
        bind_grid_pair_potential_library(graph, {"wrong": 1})
    with pytest.raises(SerializationError, match="integer dimensions"):
        bind_grid_pair_potential_library(graph, {"n": True})
    assert (
        bind_grid_pair_potential_library(graph, {}).entries[0].declaration
        == graph.entries[0].declaration
    )
    literal_graph = grid_pair_potential_library(
        {
            "bitsize": sympy.Integer(1),
            "poly_bitsize": sympy.Integer(3),
            "inv_sqrt_bitsize": 5,
            "qrom_data": _pair().qrom_data,
        }
    )
    rebound = bind_grid_pair_potential_library(literal_graph, {})
    arguments = {
        argument.name: argument.value
        for argument in rebound.entries[0].declaration.arguments
    }
    assert type(arguments["bitsize"]) is sympy.Integer(1).__class__
    assert type(arguments["poly_bitsize"]) is sympy.Integer(3).__class__
    assert rebound.entries[0].declaration == literal_graph.entries[0].declaration
