"""Check literal sparse chemistry SELECT calls, complex states and clean work."""

from hashlib import sha1
from itertools import product
from pathlib import Path

import pytest

from flagquantum.compiler.registers.lookup.sparse_chemistry_select_component import (
    sparse_select_component_pair,
)
from flagquantum.core.ir.program import verify_program_record
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_default_linear_combination_oracles import bind
from tests.unit.test_hubbard_hamiltonian_gqsp import _source_sparse_branch
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_qubitization_walk_component import (
    reference_runtime as reference_runtime,
)
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_sparse_source() -> None:
    from qualtran.bloqs.chemistry.sparse import select_bloq

    for name, blob in (
        ("select_bloq.py", "1c7fd5595aa87a6c647070cffb43d941a2d7cc9f"),
        ("select_bloq_test.py", "4afa771527bfa51d37cc7a85f64b15fd7471db4c"),
    ):
        raw = (Path(select_bloq.__file__).parent / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


def test_zero_width_spatial_source_decomposition_remains_unsupported() -> None:
    from qualtran.bloqs.chemistry.sparse.select_bloq import SelectSparse

    original = SelectSparse(2)
    assert len(original.decompose_bloq().bloq_instances) == 5
    with pytest.raises(AssertionError):
        original.decompose_bloq().flatten()
    with pytest.raises(ValueError, match="zero-width spatial axis"):
        sparse_select_component_pair(2)


@pytest.mark.parametrize("size", (4, 6, 8))
@pytest.mark.parametrize("control", (None, 0, 1))
@pytest.mark.parametrize("inverse", (False, True))
def test_complete_calls_operands_and_sparse_complex_state(
    size: int, control: int | None, inverse: bool
) -> None:
    cirq = pytest.importorskip("cirq")
    from qualtran import Adjoint
    from qualtran._infra.gate_with_registers import get_named_qubits
    from qualtran.bloqs.chemistry.sparse.select_bloq import SelectSparse

    from tests.unit.measurement_certificate import certify_erasure_histories

    original = SelectSparse(size, control)
    source = Adjoint(original) if inverse else original
    record = sparse_select_component_pair(size, control_value=control)[int(inverse)]
    assert not verify_program_record(record)
    compare_layers(record, source, atomic_leaves(record))
    bound, logical = bind(record)
    named = get_named_qubits(source.signature)
    visible = tuple(q for reg in source.signature for q in named[reg.name].flat)
    operations = list(
        cirq.decompose(
            source.on_registers(**named),
            keep=lambda op: cirq.is_measurement(op)
            or isinstance(op, cirq.ClassicallyControlledOperation)
            or isinstance(op.gate, cirq.ResetChannel)
            or (len(op.qubits) <= 3 and cirq.has_unitary(op)),
        )
    )
    # Include flag, target and address coherence, plus both outer-control
    # sectors when present. No selection-only or absolute-value comparison.
    initial = {0: 0.5 + 0j, 1: 0.5j, 1 << size: -0.5 + 0j, 1 << (logical - 1): 0.5j}
    work = bound.circuit.n_wires - logical
    events = certify_erasure_histories(
        bound.circuit.instructions,
        n_wires=bound.circuit.n_wires,
        zero_wires=tuple(range(logical, bound.circuit.n_wires)),
        check_clean_output=False,
    )
    histories = (
        product((0, 1), repeat=events)
        if size == 4 and control is None
        else ((0,) * events, (1,) * events)
    )
    assert size != 4 or control is not None or events <= 10
    expected = _source_sparse_branch(operations, visible, initial, 0)
    for history in histories:
        actual = _branch(
            bound.circuit.instructions,
            bound.circuit.n_wires,
            history,
            {key << work: value for key, value in initial.items()},
            normalize_measurements=True,
        )
        assert all(key % (1 << work) == 0 for key in actual)
        decoded = {key >> work: value for key, value in actual.items()}
        for key in expected.keys() | decoded.keys():
            assert decoded.get(key, 0) == pytest.approx(expected.get(key, 0), abs=3e-10)
