"""Actual source declarations enter owned DF models without executable bodies."""

import builtins
from dataclasses import asdict
from itertools import product
from typing import Any

import pytest

from flagquantum.algorithms.primitives.double_factorization_prepare_model import (
    source_double_factorization_indexed_data_cost,
    source_double_factorization_indexed_data_signature,
    source_double_factorization_inner_prepare_cost,
    source_double_factorization_inner_prepare_signature,
    source_double_factorization_outer_prepare_cost,
    source_double_factorization_outer_prepare_signature,
)
from flagquantum.algorithms.primitives.double_factorization_rotation_model import (
    source_double_factorization_programmed_rotation_cost,
    source_double_factorization_programmed_rotation_signature,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.ecosystem.cirq._register_graph import import_register_graph

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "kind,depth,adjoint",
    tuple(product(("inner", "outer", "indexed", "rotation"), (0, 1, 2), (False, True))),
)
@pytest.mark.parametrize(
    "aux,spin,eigenvalues,bits,alias", ((50, 10, 250, 8, 12), (31, 8, 124, 4, 6))
)
def test_actual_foreign_constructor_library_and_directional_model(
    kind: str,
    depth: int,
    adjoint: bool,
    aux: int,
    spin: int,
    eigenvalues: int,
    bits: int,
    alias: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.chemistry.df.prepare import (
        InnerPrepareDoubleFactorization,
        OuterPrepareDoubleFactorization,
        OutputIndexedData,
    )
    from qualtran.bloqs.chemistry.df.select_bloq import ProgRotGateArray
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    sources = {
        "inner": InnerPrepareDoubleFactorization(aux, spin, eigenvalues, bits, alias),
        "outer": OuterPrepareDoubleFactorization(aux, alias, bits),
        "indexed": OutputIndexedData(aux, spin, eigenvalues, bits),
        "rotation": ProgRotGateArray(aux, spin, eigenvalues, bits),
    }
    source = sources[kind]
    declaration = (
        deserialize_component_library(
            bloqs_to_proto(source, max_depth=0).SerializeToString()
        )
        .entries[0]
        .declaration
    )
    arguments = {argument.name: argument.value for argument in declaration.arguments}

    def integer(name: str) -> int:
        value = arguments[name]
        assert type(value) is int
        return value

    # Values below come from the owned constructor capture, including explicit
    # defaults and parameters absent from some source cost expressions.
    if kind == "outer":
        signature = source_double_factorization_outer_prepare_signature(
            integer("num_aux")
        )
        model = source_double_factorization_outer_prepare_cost(
            integer("num_aux"), integer("num_bits_state_prep"), adjoint=adjoint
        )
    else:
        spin = integer("num_spin_orb")
        eigenvalues = integer("num_eig")
        if kind == "inner":
            bits = integer("num_bits_rot_aa")
            signature = source_double_factorization_inner_prepare_signature(
                spin, eigenvalues, bits
            )
            model = source_double_factorization_inner_prepare_cost(
                spin, eigenvalues, bits, integer("num_bits_state_prep"), adjoint=adjoint
            )
        elif kind == "indexed":
            bits = integer("num_bits_rot_aa")
            signature = source_double_factorization_indexed_data_signature(
                integer("num_aux"), spin, eigenvalues, bits
            )
            model = source_double_factorization_indexed_data_cost(
                integer("num_aux"), spin, eigenvalues, bits, adjoint=adjoint
            )
        else:
            bits = integer("num_bits_rot")
            signature = source_double_factorization_programmed_rotation_signature(
                spin, eigenvalues, bits
            )
            model = source_double_factorization_programmed_rotation_cost(
                spin, eigenvalues, bits, adjoint=adjoint
            )
    assert tuple(signature) == declaration.registers
    direction = source.adjoint() if adjoint else source
    proto = bloqs_to_proto(direction, max_depth=depth)
    raw = proto.SerializeToString()
    library = deserialize_component_library(raw)
    encoded = serialize_component_library(library)
    assert type(proto).FromString(encoded) == proto
    assert (
        serialize_component_library(deserialize_component_library(encoded)) == encoded
    )
    assert bloqs_from_proto(proto)[0] == direction
    assert library.entries[0].declaration.registers == tuple(signature)
    assert not library.entries[0].connections
    expected = get_cost_value(direction, QECGatesCost()).asdict()
    peak = get_cost_value(direction, QubitCount())
    assert source.adjoint().adjoint() == source
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
    with pytest.raises(DecomposeNotImplementedError):
        import_register_graph(source)
    original_import = builtins.__import__

    def owned_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq", "kickmix"}:
            raise AssertionError(f"Owned DF model retained SDK dependency {name}")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", owned_import)
        result = estimate_component_resources(model)
    assert {
        name: count for name, count in asdict(result.gates).items() if count
    } == expected
    assert result.peak_logical_qubits == peak


@pytest.mark.parametrize("kind", ("inner", "rotation"))
def test_actual_negative_local_declaration_keeps_signed_counts(kind: str) -> None:
    from qualtran.bloqs.basic_gates import Toffoli
    from qualtran.bloqs.chemistry.df.prepare import InnerPrepareDoubleFactorization
    from qualtran.bloqs.chemistry.df.select_bloq import ProgRotGateArray
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    source = (
        InnerPrepareDoubleFactorization(1, 2, 7, 0, 0)
        if kind == "inner"
        else ProgRotGateArray(50, 10, 250, 1)
    )
    proto = bloqs_to_proto(source, max_depth=1)
    library = deserialize_component_library(proto.SerializeToString())
    encoded = serialize_component_library(library)
    assert type(proto).FromString(encoded) == proto
    assert bloqs_from_proto(type(proto).FromString(encoded))[0] == source
    source_count = source.bloq_counts()[Toffoli()]
    assert source_count < 0
    by_id = {entry.identifier: entry for entry in library.entries}
    counts = {
        by_id[identifier].declaration.name: count
        for identifier, count in library.entries[0].counts
    }
    assert counts["qualtran.bloqs.basic_gates.toffoli.Toffoli"] == source_count
    with pytest.raises(ValueError, match="negative gate count"):
        if kind == "inner":
            source_double_factorization_inner_prepare_cost(2, 7, 0, 0)
        else:
            source_double_factorization_programmed_rotation_cost(10, 250, 1)
