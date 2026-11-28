"""Resolve original sparse SELECT constructors and both exact directions."""

import sys
from importlib import import_module
from unittest.mock import patch

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    import_module("numpy")
    import_module("cirq")
    import_module("qualtran")
    import_module("kickmix")
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

from flagquantum import Circuit
from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.core.ir.program import verify_program_record
from flagquantum.ecosystem.cirq._sparse_chemistry_library import (
    sparse_chemistry_library_factories,
)
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_default_linear_combination_oracles import bind
from tests.unit.test_hubbard_hamiltonian_gqsp import _source_sparse_branch
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_qubitization_walk_component import (
    reference_runtime as reference_runtime,
)
from tests.unit.test_sparse_chemistry_prepare import _branch
from tests.unit.test_sparse_chemistry_select_component import (
    frozen_sparse_source as frozen_sparse_source,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("direction", ("forward", "before", "after"))
@pytest.mark.parametrize("values", ((1, 0), (0, 1, 0)))
def test_grouped_control_source_bystander_and_inverse_order(
    direction: str, values: tuple[int, ...]
) -> None:
    cirq = pytest.importorskip("cirq")
    from qualtran import Adjoint, CtrlSpec
    from qualtran._infra.gate_with_registers import get_named_qubits
    from qualtran.bloqs.chemistry.sparse.select_bloq import SelectSparse

    from tests.unit.measurement_certificate import certify_erasure_histories

    original = SelectSparse(4)
    source = (Adjoint(original) if direction == "before" else original).controlled(
        CtrlSpec(cvs=values)
    )
    if direction == "after":
        source = Adjoint(source)
    if direction == "before":
        named = get_named_qubits(source.signature)
        with pytest.raises(ValueError, match="can't be decomposed.*Controlled"):
            list(
                cirq.decompose(
                    source.on_registers(**named),
                    keep=lambda op: cirq.is_measurement(op)
                    or isinstance(op, cirq.ClassicallyControlledOperation)
                    or isinstance(op.gate, cirq.ResetChannel)
                    or (len(op.qubits) <= 3 and cirq.has_unitary(op)),
                )
            )
        record = resolve_component_program(
            _library(source, 0), 0, sparse_chemistry_library_factories()
        )
        compare_layers(record, source, atomic_leaves(record))
        from tests.unit.test_sparse_chemistry_prepare_component import (
            bind_used_workspace,
        )

        with pytest.raises(ValueError, match="requires component bodies"):
            bind_used_workspace(record, allow_workspace_outputs=True)
        return
    record = resolve_component_program(
        _library(source, 0), 0, sparse_chemistry_library_factories()
    )
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
    initial = {0: 0.6 + 0j, 1 << (logical - 1): 0.8j}
    work = bound.circuit.n_wires - logical
    count = certify_erasure_histories(
        bound.circuit.instructions,
        n_wires=bound.circuit.n_wires,
        zero_wires=tuple(range(logical, bound.circuit.n_wires)),
        check_clean_output=False,
    )
    for outcome in (0, 1):
        expected = _source_sparse_branch(operations, visible, initial, outcome)
        actual = _branch(
            bound.circuit.instructions,
            bound.circuit.n_wires,
            (outcome,) * count,
            {k << work: v for k, v in initial.items()},
            normalize_measurements=True,
        )
        assert all(k % (1 << work) == 0 for k in actual)
        decoded = {k >> work: v for k, v in actual.items()}
        for key in expected.keys() | decoded.keys():
            assert decoded.get(key, 0) == pytest.approx(expected.get(key, 0), abs=3e-10)


@pytest.mark.parametrize("size", (4, 6, 8))
@pytest.mark.parametrize("control", (None, 0, 1))
@pytest.mark.parametrize("inverse", (False, True))
def test_original_constructor_depths_exact_calls_and_sdk_free_append(
    size: int, control: int | None, inverse: bool
) -> None:
    from qualtran import Adjoint
    from qualtran.bloqs.chemistry.sparse.select_bloq import SelectSparse

    original = SelectSparse(size, control)
    source = Adjoint(original) if inverse else original
    for depth in (0, 1):
        record = resolve_component_program(
            _library(source, depth), 0, sparse_chemistry_library_factories()
        )
        assert not verify_program_record(record)
        compare_layers(record, source, atomic_leaves(record))
        with patch.dict(sys.modules, {"qualtran": None, "cirq": None}):
            bound, _ = bind(record)
            circuit = Circuit(bound.circuit.n_wires).append(
                _BoundBlockOperation(bound.circuit.instructions, bound.circuit.n_wires)
            )
            assert circuit.to_ir().instructions == bound.circuit.instructions


@pytest.mark.parametrize("existing", (0, 1))
@pytest.mark.parametrize("direction", ("forward", "before", "after"))
def test_preexisting_control_polarities_and_grouped_control_orders(
    existing: int,
    direction: str,
) -> None:
    from qualtran import Adjoint, CtrlSpec
    from qualtran.bloqs.chemistry.sparse.select_bloq import SelectSparse
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd

    from tests.unit.test_sparse_chemistry_prepare_component import bind_used_workspace

    original = SelectSparse(4, existing)
    source = ControlledViaAnd(
        Adjoint(original) if direction == "before" else original,
        CtrlSpec(cvs=(1, 0)),
    )
    if direction == "after":
        source = Adjoint(source)
    record = resolve_component_program(
        _library(source, 0), 0, sparse_chemistry_library_factories()
    )
    compare_layers(record, source, atomic_leaves(record))
    if direction == "before":
        with pytest.raises(ValueError, match="requires component bodies"):
            bind_used_workspace(record, allow_workspace_outputs=True)
    else:
        bound, _ = bind(record)
        circuit = Circuit(bound.circuit.n_wires).append(
            _BoundBlockOperation(bound.circuit.instructions, bound.circuit.n_wires)
        )
        assert circuit.to_ir().instructions == bound.circuit.instructions
