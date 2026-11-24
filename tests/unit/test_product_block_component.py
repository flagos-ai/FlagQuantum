"""Stored Product calls preserve flags, unused words and measured cleanup."""

from itertools import product
from math import sqrt
from typing import Any

import pytest
import torch

from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_unitary_block_factories import (
    unitary_block_factories,
)
from flagquantum.compiler.registers.layout.product_block_component import (
    product_block_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _children(
    source: Any,
) -> tuple[tuple[ProgramRecord, ...], tuple[ProgramRecord, ...]]:
    return tuple(
        resolve_component_program(_library(u, 0), 0, unitary_block_factories())
        for u in source.block_encodings
    ), tuple(
        resolve_component_program(
            _library(u.adjoint(), 0), 0, unitary_block_factories()
        )
        for u in source.block_encodings
    )


def _bound(record: ProgramRecord, source: Any) -> Any:
    assert not verify_program_record(record)
    compare_layers(record, source, atomic_leaves(record))
    cursor = 0
    inputs = []
    for reg in record.functions[0].registers or ():
        width = int(reg.total_bits())
        inputs.append(tuple(range(cursor, cursor + width)))
        cursor += width
    bound = bind_register_program(
        record, tuple(inputs), clean=tuple(range(cursor, cursor + 3))
    )
    torch.testing.assert_close(
        logical_operator(bound.circuit, cursor),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=2e-11,
        rtol=0,
    )
    return bound


@pytest.mark.parametrize("properties", (False, True))
@pytest.mark.parametrize("inverse", (False, True))
def test_original_product_graph_and_complete_complex_operator(
    properties: bool, inverse: bool
) -> None:
    from qualtran.bloqs.block_encoding.product import (
        _product_block_encoding,
        _product_block_encoding_properties,
    )
    from qualtran.serialization.bloq import bloqs_to_proto

    original = (
        _product_block_encoding_properties if properties else _product_block_encoding
    ).make()
    children, inverses = _children(original)
    record = product_block_component_program(
        children, inverses=inverses, adjoint=inverse
    )
    source = original.adjoint() if inverse else original
    _bound(record, source)
    with pytest.raises(ValueError, match="Object arrays"):
        bloqs_to_proto(source, max_depth=0)


@pytest.mark.parametrize(
    "ancillas", ((0,), (2,), (0, 0), (2, 1), (1, 2), (0, 2), (2, 2), (2, 3), (1, 2, 1))
)
def test_unused_shared_words_dirty_flags_and_exact_inverse_all_histories(
    ancillas: tuple[int, ...],
) -> None:
    from qualtran.bloqs.basic_gates import Hadamard, SGate, TGate
    from qualtran.bloqs.block_encoding.product import Product
    from qualtran.bloqs.block_encoding.unitary import Unitary

    gates = (TGate(), Hadamard(), SGate())
    original = Product(
        tuple(
            Unitary(gates[i], ancilla_bitsize=width, resource_bitsize=int(i % 2 == 0))
            for i, width in enumerate(ancillas)
        )
    )
    children, inverses = _children(original)
    records = tuple(
        product_block_component_program(children, inverses=inverses, adjoint=direction)
        for direction in (False, True)
    )
    bounds = tuple(
        _bound(record, source)
        for record, source in zip(records, (original, original.adjoint()), strict=True)
    )
    logical = int(original.signature.n_qubits())
    wires = bounds[0].circuit.n_wires
    seed = torch.arange(1, (1 << logical) + 1, dtype=torch.float64)
    values = torch.complex(seed, seed.remainder(5) - 2)
    values /= values.norm()
    state = torch.zeros(1 << wires, dtype=torch.complex128)
    state[:: 1 << (wires - logical)] = values
    instructions = (*bounds[0].circuit.instructions, *bounds[1].circuit.instructions)
    count = sum(g.name == "measure" for g in instructions)
    for history in product((0, 1), repeat=count):
        torch.testing.assert_close(
            forced_branch(state, instructions, history) * sqrt(2**count),
            state,
            atol=2e-11,
            rtol=0,
        )


def test_original_symbolic_product_declaration_and_physical_refusal() -> None:
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.block_encoding.product import _product_block_encoding_symb

    from flagquantum.compiler.registers.layout.product_block_component import (
        product_block_signature,
    )

    source = _product_block_encoding_symb.make()
    children, _ = _children(source)
    record = product_block_component_program(children)
    assert not verify_program_record(record) and not record.functions[0].blocks
    assert record.functions[0].registers == tuple(
        product_block_signature(
            source.system_bitsize,
            tuple(u.ancilla_bitsize for u in source.block_encodings),
            tuple(u.resource_bitsize for u in source.block_encodings),
        )
    )
    with pytest.raises(DecomposeTypeError, match="symbolic"):
        source.decompose_bloq()
    with pytest.raises(ValueError):
        bind_register_program(record, ((0,), (1, 2, 3)))


def test_unequal_nonzero_resource_failure_keeps_original_cause() -> None:
    from qualtran._infra.composite_bloq import BloqError
    from qualtran.bloqs.basic_gates import Hadamard, TGate
    from qualtran.bloqs.block_encoding.product import Product
    from qualtran.bloqs.block_encoding.unitary import Unitary

    source = Product(
        (Unitary(TGate(), resource_bitsize=2), Unitary(Hadamard(), resource_bitsize=1))
    )
    with pytest.raises(BloqError, match="register dtypes.*resource"):
        source.decompose_bloq()
    with pytest.raises(BloqError, match="register dtypes.*resource"):
        source.tensor_contract()
    children, _ = _children(source)
    with pytest.raises(ValueError, match="unequal nonzero resource"):
        product_block_component_program(children)


def test_product_native_build_bind_and_append_without_source_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from qualtran.bloqs.block_encoding.product import _product_block_encoding_properties

    from flagquantum import Circuit
    from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation

    children, inverses = _children(_product_block_encoding_properties.make())
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        patch.setitem(sys.modules, "cirq", None)
        record = product_block_component_program(
            children, inverses=inverses, adjoint=True
        )
        bound = bind_register_program(record, ((0,), (1, 2, 3), (4,)), clean=(5, 6))
        circuit = Circuit(bound.circuit.n_wires)
        circuit.append(
            _BoundBlockOperation(bound.circuit.instructions, bound.circuit.n_wires)
        )
        assert (
            circuit.to_ir().to_dict()["instructions"]
            == bound.circuit.to_dict()["instructions"]
        )
