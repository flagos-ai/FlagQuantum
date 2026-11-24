"""Packed blocks preserve source calls, auxiliary state and exact inverses."""

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
from flagquantum.compiler.registers.layout.packed_block_component import (
    tensor_product_block_component_program,
    unitary_block_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _check(record: ProgramRecord, source: Any) -> Any:
    assert not verify_program_record(record)
    compare_layers(record, source, atomic_leaves(record))
    cursor = 0
    inputs = []
    for reg in record.functions[0].registers or ():
        width = int(reg.total_bits())
        inputs.append(tuple(range(cursor, cursor + width)))
        cursor += width
    bound = bind_register_program(record, tuple(inputs))
    torch.testing.assert_close(
        logical_operator(bound.circuit, cursor),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=2e-11,
        rtol=0,
    )
    assert not bound.events
    return bound


@pytest.mark.parametrize("name", ("TGate", "SGate", "CNOT", "Hadamard"))
@pytest.mark.parametrize("auxiliary", ((0, 0), (1, 2)))
@pytest.mark.parametrize("inverse", (False, True))
def test_unitary_original_calls_and_full_auxiliary_operator(
    name: str, auxiliary: tuple[int, int], inverse: bool
) -> None:
    from qualtran.bloqs import basic_gates
    from qualtran.bloqs.block_encoding.unitary import Unitary

    child = getattr(basic_gates, name)()
    source = Unitary(child, ancilla_bitsize=auxiliary[0], resource_bitsize=auxiliary[1])
    if inverse:
        source = source.adjoint()
    record = resolve_component_program(
        _library(source, 0), 0, unitary_block_factories()
    )
    _check(record, source)


@pytest.mark.parametrize("properties", (False, True))
@pytest.mark.parametrize("inverse", (False, True))
def test_original_unitary_examples_and_deeper_wire_failure(
    properties: bool, inverse: bool
) -> None:
    from qualtran.bloqs.block_encoding.unitary import (
        _unitary_block_encoding,
        _unitary_block_encoding_properties,
    )
    from qualtran.serialization.bloq import bloqs_to_proto

    source = (
        _unitary_block_encoding_properties if properties else _unitary_block_encoding
    ).make()
    if inverse:
        source = source.adjoint()
    record = resolve_component_program(
        _library(source, 0), 0, unitary_block_factories()
    )
    _check(record, source)
    with pytest.raises(ValueError, match="inhomogeneous|sequence"):
        bloqs_to_proto(source, max_depth=1)


@pytest.mark.parametrize("properties", (False, True))
@pytest.mark.parametrize("inverse", (False, True))
def test_original_tensor_graph_and_full_complex_operator(
    properties: bool, inverse: bool
) -> None:
    from qualtran.bloqs.block_encoding.tensor_product import (
        _tensor_product_block_encoding,
        _tensor_product_block_encoding_properties,
    )
    from qualtran.serialization.bloq import bloqs_to_proto

    original = (
        _tensor_product_block_encoding_properties
        if properties
        else _tensor_product_block_encoding
    ).make()
    children = tuple(
        resolve_component_program(_library(child, 0), 0, unitary_block_factories())
        for child in original.block_encodings
    )
    inverses = tuple(
        resolve_component_program(
            _library(child.adjoint(), 0), 0, unitary_block_factories()
        )
        for child in original.block_encodings
    )
    record = tensor_product_block_component_program(
        children, inverses=inverses, adjoint=inverse
    )
    source = original.adjoint() if inverse else original
    _check(record, source)
    with pytest.raises(ValueError, match="Object arrays"):
        bloqs_to_proto(source, max_depth=0)


@pytest.mark.parametrize("tensor", (False, True))
def test_measured_child_exact_inverse_all_raw_histories_and_zero_work(
    tensor: bool,
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.block_encoding.unitary import Unitary

    from tests.unit.test_legacy_block_encoding_factories import _oracles

    _, source_select, _, _, pair = _oracles()
    positive, _ = source_select.get_ctrl_system(CtrlSpec())
    original = Unitary(positive, ancilla_bitsize=1, resource_bitsize=1)
    records = tuple(
        unitary_block_component_program(
            pair[0], ancilla=1, resource=1, inverse=pair[1], adjoint=direction
        )
        for direction in (False, True)
    )
    if tensor:
        from qualtran.bloqs.basic_gates import TGate
        from qualtran.bloqs.block_encoding.tensor_product import TensorProduct

        extra = Unitary(TGate())
        extra_pair = tuple(
            resolve_component_program(_library(child, 0), 0, unitary_block_factories())
            for child in (extra, extra.adjoint())
        )
        records = tuple(
            tensor_product_block_component_program(
                (records[0], extra_pair[0]),
                inverses=(records[1], extra_pair[1]),
                adjoint=direction,
            )
            for direction in (False, True)
        )
        original = TensorProduct((original, extra))
    bounds = []
    logical = 5 + int(tensor)
    for record, source in zip(records, (original, original.adjoint()), strict=True):
        assert not verify_program_record(record)
        compare_layers(record, source, atomic_leaves(record))
        width = logical - 2
        bound = bind_register_program(
            record,
            (tuple(range(width)), (width,), (width + 1,)),
            clean=(logical,),
        )
        torch.testing.assert_close(
            logical_operator(bound.circuit, logical),
            torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
            atol=2e-11,
            rtol=0,
        )
        bounds.append(bound)
    gates = (*bounds[0].circuit.instructions, *bounds[1].circuit.instructions)
    count = sum(g.name == "measure" for g in gates)
    assert count > 0
    seed = torch.arange(1, (1 << logical) + 1, dtype=torch.float64)
    amplitudes = torch.complex(seed, seed.remainder(3) - 1)
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << (logical + 1), dtype=torch.complex128)
    state[::2] = amplitudes
    for history in product((0, 1), repeat=count):
        torch.testing.assert_close(
            forced_branch(state, gates, history) * sqrt(2**count),
            state,
            atol=2e-11,
            rtol=0,
        )


def test_unitary_resolution_binding_and_append_without_source_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from qualtran.bloqs.block_encoding.unitary import _unitary_block_encoding_properties

    from flagquantum import Circuit
    from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation

    source = _unitary_block_encoding_properties.make().adjoint()
    library = _library(source, 0)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        patch.setitem(sys.modules, "cirq", None)
        record = resolve_component_program(library, 0, unitary_block_factories())
        bound = bind_register_program(record, ((0,), (1, 2), (3,)))
        circuit = Circuit(4)
        circuit.append(_BoundBlockOperation(bound.circuit.instructions, 4))
        assert (
            circuit.to_ir().to_dict()["instructions"]
            == bound.circuit.to_dict()["instructions"]
        )


def test_original_unitary_shaped_width_mismatch_is_not_repaired() -> None:
    from qualtran.bloqs.basic_gates import Toffoli
    from qualtran.bloqs.block_encoding.unitary import Unitary

    source = Unitary(Toffoli())
    assert source.system_bitsize == 2 and Toffoli().signature.n_qubits() == 3
    source.decompose_bloq()  # The shallow source graph admits the bad boundary.
    with pytest.raises(ValueError, match="Total bitsize"):
        source.tensor_contract()
    with pytest.raises(ValueError, match="Total bitsize"):
        source.decompose_bloq().flatten_once()
    with pytest.raises(ValueError):
        resolve_component_program(_library(source, 0), 0, unitary_block_factories())


def test_symbolic_unitary_bystanders_and_original_tensor_refusal() -> None:
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.block_encoding.tensor_product import (
        _tensor_product_block_encoding_symb,
    )

    source = _tensor_product_block_encoding_symb.make()
    children = tuple(
        resolve_component_program(_library(child, 0), 0, unitary_block_factories())
        for child in source.block_encodings
    )
    record = tensor_product_block_component_program(children)
    assert not verify_program_record(record)
    assert not record.functions[0].blocks
    from flagquantum.compiler.registers.layout.packed_block_component import (
        tensor_product_block_signature,
    )

    assert record.functions[0].registers == tuple(
        tensor_product_block_signature(
            tuple(
                (child.system_bitsize, child.ancilla_bitsize, child.resource_bitsize)
                for child in source.block_encodings
            )
        )
    )
    with pytest.raises(DecomposeTypeError, match="symbolic"):
        source.decompose_bloq()
    with pytest.raises(ValueError):
        bind_register_program(record, ((0, 1), (2, 3)))
