"""Compare the actual original Ising walk with its retained physical environment."""

import sys
from math import sqrt
from typing import Any
from unittest.mock import patch

import pytest

from flagquantum import Circuit
from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation
from flagquantum.ecosystem._pauli_walk_component_library import (
    capture_pauli_walk_context,
)
from tests.unit.test_default_linear_combination_oracles import bind
from tests.unit.test_hubbard_hamiltonian_gqsp import _source_sparse_branch
from tests.unit.test_qubitization_walk_component import (
    reference_runtime as reference_runtime,
)
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("controlled", (False, True))
def test_original_ising_full_physical_complex_output_and_joint_inverse(
    controlled: bool,
) -> None:
    try:
        import cirq
    except ModuleNotFoundError:
        pytest.skip("The locked Cirq reference environment is required")
    from qualtran._infra.gate_with_registers import get_named_qubits
    from qualtran.bloqs.qubitization.qubitization_walk_operator import _walk_op
    from qualtran.bloqs.reflections.reflection_using_prepare import (
        ReflectionUsingPrepare,
    )

    from tests.unit.measurement_certificate import certify_erasure_histories

    original = _walk_op.make()
    context = capture_pauli_walk_context(original)
    source = original.controlled() if controlled else original
    native = context.pair(controlled=controlled)
    with patch.dict(sys.modules, {"qualtran": None, "cirq": None}):
        forward, logical = bind(native[0])
        backward, inverse_logical = bind(native[1])
        circuit = Circuit(forward.circuit.n_wires).append(
            _BoundBlockOperation(forward.circuit.instructions, forward.circuit.n_wires)
        )
        assert circuit.to_ir().instructions == forward.circuit.instructions
    assert inverse_logical == logical
    named = get_named_qubits(source.signature)
    body = list(cirq.flatten_to_ops(cirq.decompose_once(source.on_registers(**named))))
    reflection = next(op for op in body if isinstance(op.gate, ReflectionUsingPrepare))

    # The source releases signal junk without proving zero. Preserve its exact
    # allocation groups as visible data, while all other source work stays hidden.
    class RecordingManager(cirq.SimpleQubitManager):
        def __init__(self) -> None:
            super().__init__(prefix="retained_signal")
            self.groups: list[list[Any]] = []

        def qalloc(self, n: int, dim: int = 2) -> list[Any]:
            result = super().qalloc(n, dim)
            self.groups.append(result)
            return result

    manager = RecordingManager()
    reflection_body = cirq.decompose_once(
        reflection, context=cirq.DecompositionContext(manager)
    )
    signal_groups = manager.groups if controlled else manager.groups[1:]
    signal = tuple(q for group in signal_groups for q in group)
    visible = (
        *tuple(q for reg in source.signature for q in named[reg.name].flat),
        *signal,
    )
    assert len(visible) == logical
    expanded = [reflection_body if op is reflection else op for op in body]
    operations = list(
        cirq.decompose(
            expanded,
            keep=lambda op: cirq.is_measurement(op)
            or isinstance(op, cirq.ClassicallyControlledOperation)
            or isinstance(op.gate, cirq.ResetChannel)
            or (len(op.qubits) <= 3 and cirq.has_unitary(op)),
        )
    )
    target_lsb = 1 << len(signal)
    selection_lsb = target_lsb << 4
    initial = {
        0: 1 / sqrt(30),
        target_lsb: 2j / sqrt(30),
        selection_lsb | target_lsb: -3 / sqrt(30),
        1 << (logical - 1): 4j / sqrt(30),
    }
    for pattern in (0, 1):
        expected = _source_sparse_branch(operations, visible, initial, pattern)
        work = forward.circuit.n_wires - logical
        count = certify_erasure_histories(
            forward.circuit.instructions,
            n_wires=forward.circuit.n_wires,
            zero_wires=tuple(range(logical, forward.circuit.n_wires)),
            check_clean_output=False,
        )
        actual = _branch(
            forward.circuit.instructions,
            forward.circuit.n_wires,
            (pattern,) * count,
            {key << work: value for key, value in initial.items()},
            normalize_measurements=True,
        )
        assert all(key % (1 << work) == 0 for key in actual)
        decoded = {key >> work: value for key, value in actual.items()}
        for key in expected.keys() | decoded.keys():
            assert decoded.get(key, 0) == pytest.approx(expected.get(key, 0), abs=4e-10)
        inverse_work = backward.circuit.n_wires - logical
        inverse_count = certify_erasure_histories(
            backward.circuit.instructions,
            n_wires=backward.circuit.n_wires,
            zero_wires=tuple(range(logical, backward.circuit.n_wires)),
            check_clean_output=False,
        )
        restored = _branch(
            backward.circuit.instructions,
            backward.circuit.n_wires,
            (pattern,) * inverse_count,
            {key << inverse_work: value for key, value in decoded.items()},
            normalize_measurements=True,
        )
        target = {key << inverse_work: value for key, value in initial.items()}
        for key in restored.keys() | target.keys():
            assert restored.get(key, 0) == pytest.approx(target.get(key, 0), abs=4e-10)
