"""Owned snapshot values participate in source-ordered term collection."""

from collections.abc import Callable, Hashable, Mapping
from dataclasses import asdict, dataclass, replace
from itertools import product
from math import sqrt

import pytest
import torch

from flagquantum.algorithms.primitives._block_metadata import _BlockMetadata
from flagquantum.algorithms.primitives.block_encoding_rewrites import collect_like_terms
from flagquantum.algorithms.primitives.circuit_block_encoding import (
    CircuitBlockEncoding,
)
from flagquantum.algorithms.primitives.linear_combination_block_encoding import (
    LinearCombinationBlockEncoding,
)
from flagquantum.circuit import _CircuitOperation
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    GateCost,
    estimate_component_resources,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem.cirq._block_encoding import import_block_encoding
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_block_encoding_rewrites import _bind_rewritten_lcu
from tests.unit.test_linear_combination_catalog import _assert_source_flow

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

# Reuse the existing source traversal through its concrete consumer contract.
_assert_block_source_flow: Callable[
    [object, LinearCombinationBlockEncoding, DynamicCircuit], None
] = _assert_source_flow


def _binder(
    child: CircuitBlockEncoding,
) -> Callable[
    [int, Mapping[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...], bool],
    _CircuitOperation,
]:
    def bind(
        control: int,
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        inverse: bool,
    ) -> _CircuitOperation:
        return child.make_on(
            control=(control,),
            system=ports["system"],
            ancilla=ports.get("ancilla", ()),
            resource=ports.get("resource", ()),
            workspace=work,
            classical_bits=slots,
            adjoint=inverse,
        )

    return bind


@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_independent_foreign_snapshots_rewrite_and_execute_all_histories(
    adjoint: bool, nested: bool
) -> None:
    from qualtran.bloqs.basic_gates import Hadamard, XGate
    from qualtran.bloqs.block_encoding import LinearCombination, Unitary
    from qualtran.bloqs.block_encoding.rewrites import collect_like_terms as original
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    x, another_x, h = (
        import_block_encoding(Unitary(gate)) for gate in (XGate(), XGate(), Hadamard())
    )
    assert x is not another_x and x == another_x
    with pytest.raises(TypeError):
        hash(x)
    if nested:
        native = LinearCombinationBlockEncoding(
            (LinearCombinationBlockEncoding((x, h), (1.0, 3.0), 1), another_x),
            (1.0, 2.0),
            1,
        )
        source = LinearCombination(
            (
                LinearCombination(
                    (Unitary(XGate()), Unitary(Hadamard())), (1.0, 3.0), 1
                ),
                Unitary(XGate()),
            ),
            (1.0, 2.0),
            1,
        )
    else:
        native = LinearCombinationBlockEncoding((x, another_x, h), (1.0, 2.0, 3.0), 1)
        source = LinearCombination(
            (Unitary(XGate()), Unitary(XGate()), Unitary(Hadamard())),
            (1.0, 2.0, 3.0),
            1,
        )
    result = collect_like_terms(native)
    locked = original(source)
    assert isinstance(result, LinearCombinationBlockEncoding)
    assert result.block_encodings == (x, h)
    assert result.block_encodings[0] is x
    assert result.coefficients == locked._lambd == (3.0, 3.0)
    assert result.coefficient_bits == locked.lambd_bits
    circuit, work = _bind_rewritten_lcu(
        result, (_binder(x), _binder(h)), adjoint=adjoint
    )
    if adjoint:
        locked = locked.adjoint()
    _assert_block_source_flow(locked, result, circuit)
    prepare = result.coefficient_preparation.resource_model()
    select = result.select.resource_model(
        (
            ComponentCost("CX", 2, GateCost(clifford=1)),
            ComponentCost("CH", 2, GateCost(clifford=4, rotation=2)),
        )
    )
    model = estimate_component_resources(
        result.resource_model(select, prepare, prepare)
    )
    original_cost = get_cost_value(locked, QECGatesCost())
    assert asdict(model.gates) == {
        key: getattr(original_cost, key) for key in asdict(model.gates)
    }
    assert model.peak_logical_qubits == get_cost_value(locked, QubitCount())
    matrix = torch.as_tensor(locked.tensor_contract(), dtype=torch.complex128)
    data = torch.arange(matrix.shape[0], dtype=torch.float64)
    coherent = torch.complex(data + 1, data % 3 - 1)
    coherent /= coherent.norm()
    clean = torch.zeros(1 << len(work), dtype=torch.complex128)
    clean[0] = 1
    initial = torch.kron(coherent, clean)
    expected = torch.kron(matrix @ coherent, clean)
    gates = circuit.to_ir().instructions
    events = sum(gate.name == "measure" for gate in gates)
    for history in product((0, 1), repeat=events):
        branch = forced_branch(initial, gates, history)
        torch.testing.assert_close(
            branch, expected / sqrt(2**events), atol=6e-12, rtol=0
        )
        if work:
            assert branch.reshape(-1, 1 << len(work))[:, 1:].abs().max().item() < 6e-12


@pytest.mark.parametrize("field,value", [("alpha", 2.0), ("epsilon", 0.1)])
def test_snapshot_metadata_prevents_incorrect_collection(
    field: str, value: float
) -> None:
    from qualtran.bloqs.basic_gates import Hadamard, XGate
    from qualtran.bloqs.block_encoding import Unitary

    x = import_block_encoding(Unitary(XGate()))
    distinct = (
        replace(x, alpha=value) if field == "alpha" else replace(x, epsilon=value)
    )
    h = import_block_encoding(Unitary(Hadamard()))
    result = collect_like_terms(
        LinearCombinationBlockEncoding((x, distinct, h), (1.0, 2.0, 3.0), 1)
    )
    assert isinstance(result, LinearCombinationBlockEncoding)
    assert result.block_encodings == (x, distinct, h)
    assert result.coefficients == (1.0, 2.0, 3.0)


def test_snapshot_control_availability_is_part_of_term_identity() -> None:
    from qualtran.bloqs.basic_gates import Hadamard, XGate
    from qualtran.bloqs.block_encoding import Unitary

    full = import_block_encoding(Unitary(XGate()))
    ordinary = import_block_encoding(Unitary(XGate()), include_controlled=False)
    h = import_block_encoding(Unitary(Hadamard()))
    result = collect_like_terms(
        LinearCombinationBlockEncoding((full, ordinary, h), (1.0, 2.0, 3.0), 1)
    )
    assert isinstance(result, LinearCombinationBlockEncoding)
    assert result.block_encodings == (full, ordinary, h)


def test_explicit_hash_collisions_compare_complete_keys() -> None:
    from qualtran.bloqs.basic_gates import Hadamard, XGate
    from qualtran.bloqs.block_encoding import Unitary

    @dataclass(frozen=True)
    class Key:
        label: str

        def __hash__(self) -> int:
            return 0

    x, another_x, h = (
        import_block_encoding(Unitary(gate)) for gate in (XGate(), XGate(), Hadamard())
    )

    def identity(term: _BlockMetadata) -> Hashable:
        return Key("h" if term is h else "x")

    result = collect_like_terms(
        LinearCombinationBlockEncoding((x, another_x, h), (1.0, 2.0, 3.0), 1),
        term_key=identity,
    )
    assert isinstance(result, LinearCombinationBlockEncoding)
    assert result.block_encodings == (x, h)
    assert result.coefficients == (3.0, 3.0)
