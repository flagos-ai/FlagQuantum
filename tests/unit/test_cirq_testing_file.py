"""Locked Cirq testing helpers and named fixture behavior."""

from hashlib import sha1
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from flagquantum.core.ir.program.register_types import opaque_register_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem.cirq._legacy_t_complexity import LegacyTComplexity
from flagquantum.ecosystem.cirq._testing import (
    CirqGateFixture,
    assert_circuit_input_output,
    assert_decomposition_cost,
    get_circuit_input_output,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def cirq() -> Any:
    return pytest.importorskip("cirq")


@pytest.fixture(scope="module")
def locked_testing() -> Any:
    import kickmix
    from qualtran.cirq_interop import testing

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    source = Path(testing.__file__)
    for path, digest in (
        (source, "203207f7591c99f0e66dfce9f90980fc996f37ac"),
        (
            source.with_name("testing_test.py"),
            "cc35f145ea9be79e1108ae0e4c76330f544a72f3",
        ),
    ):
        raw = path.read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return testing


@pytest.mark.ftqc_reference
def test_original_basis_io_and_phase(cirq: Any, locked_testing: Any) -> None:
    qubits = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(cirq.X(qubits[3]))
    inputs, outputs = [0, 1, 0, 0], [0, 1, 0, 1]
    assert get_circuit_input_output(circuit, qubits, inputs, outputs) == (
        locked_testing.get_circuit_inp_out_cirqsim(circuit, qubits, inputs, outputs)
    )
    assert_circuit_input_output(circuit, qubits, inputs, outputs)
    with pytest.raises(AssertionError, match="expected='0001'"):
        assert_circuit_input_output(circuit, qubits, inputs, [0, 0, 0, 1])

    reversed_order = tuple(reversed(qubits))
    changed_phase = cirq.Circuit(cirq.X(qubits[3]), cirq.global_phase_operation(-1j))
    args = (changed_phase, reversed_order, [0, 0, 1, 0], [1, 0, 1, 0])
    assert get_circuit_input_output(
        *args
    ) == locked_testing.get_circuit_inp_out_cirqsim(*args)
    with pytest.raises(AssertionError):
        assert_circuit_input_output(*args)


@pytest.mark.ftqc_reference
def test_original_named_multi_and_fixture(cirq: Any, locked_testing: Any) -> None:
    from qualtran.bloqs.mcmt import MultiAnd

    gate = MultiAnd(cvs=(1, 0, 1, 0))
    source = locked_testing.GateHelper(gate)
    signature = Signature(
        (
            Register("ctrl", opaque_register_type(1), shape=(4,)),
            Register(
                "junk", opaque_register_type(1), shape=(2,), side=RegisterSide.RIGHT
            ),
            Register("target", opaque_register_type(1), side=RegisterSide.RIGHT),
        )
    )
    native = CirqGateFixture(gate, signature)
    assert native.gate == source.gate
    assert [(r.name, r.bitsize, r.shape, r.side.name) for r in native.signature] == [
        (r.name, r.bitsize, r.shape, r.side.name) for r in source.r
    ]
    assert tuple(native.quregs) == ("ctrl", "junk", "target")
    for name, qubits in native.quregs.items():
        np.testing.assert_array_equal(qubits, source.quregs[name])
    assert (
        native.operation.qubits == tuple(native.all_qubits) == tuple(source.all_qubits)
    )
    assert len(native.circuit) == 1
    assert list(native.decomposed_circuit.all_operations()) == list(
        source.decomposed_circuit.all_operations()
    )
    state = np.zeros(128, dtype=np.complex128)
    state[::8] = np.exp(1j * np.arange(16) / 3) / 4
    actual = (
        cirq.Simulator(dtype=np.complex128)
        .simulate(
            native.decomposed_circuit,
            initial_state=state.copy(),
            qubit_order=native.all_qubits,
        )
        .final_state_vector
    )
    expected = (
        cirq.Simulator(dtype=np.complex128)
        .simulate(
            source.decomposed_circuit,
            initial_state=state.copy(),
            qubit_order=source.all_qubits,
        )
        .final_state_vector
    )
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)


def test_owned_gate_fixture_cleans_work_and_keeps_explicit_qid_order(cirq: Any) -> None:
    class WorkGate(cirq.testing.SingleQubitGate):  # type: ignore[misc]
        def _decompose_with_context_(
            self, qubits: tuple[Any, ...], context: Any
        ) -> Any:
            (source,) = qubits
            (work,) = context.qubit_manager.qalloc(1)
            yield cirq.CNOT(source, work)
            yield cirq.CNOT(source, work)
            context.qubit_manager.qfree([work])

    # The first offered work wire has the same name as the input.
    fixture = CirqGateFixture(
        WorkGate(),
        Signature.build(a_0=1),
        cirq.GreedyQubitManager("a", maximize_reuse=True),
    )
    data, work = cirq.NamedQubit.range(2, prefix="a_")
    assert fixture.all_qubits == [data, work]
    from flagquantum.ecosystem.cirq._decompose import decompose_cirq_once

    expected = [cirq.CNOT(data, work)] * 2
    assert list(fixture.decomposed_circuit.all_operations()) == cirq.decompose(expected)
    first_level = decompose_cirq_once(fixture.operation)
    assert first_level is not None
    assert [op.qubits[0] for op in first_level] == [data, data]
    assert first_level[0].qubits[1] == first_level[1].qubits[1]
    assert first_level[0].qubits[1] != data
    state = np.array([1, 0, 1j, 0], dtype=np.complex128) / np.sqrt(2)
    actual = (
        cirq.Simulator(dtype=np.complex128)
        .simulate(
            fixture.decomposed_circuit,
            initial_state=state.copy(),
            qubit_order=fixture.all_qubits,
        )
        .final_state_vector
    )
    np.testing.assert_allclose(actual, state, atol=2e-12, rtol=0)


@pytest.mark.ftqc_reference
def test_original_cost_assertion_and_failure_modes(
    cirq: Any, locked_testing: Any
) -> None:
    from qualtran.cirq_interop.testing_test import ConsistentDecompostion

    original = ConsistentDecompostion()
    locked_testing.assert_decompose_is_consistent_with_t_complexity(original)
    assert_decomposition_cost(original)

    class Empty(cirq.testing.SingleQubitGate):  # type: ignore[misc]
        def _t_complexity_(self) -> LegacyTComplexity:
            return LegacyTComplexity()

        def _decompose_(self, qubits: Any) -> list[Any]:
            return []

    class Missing(cirq.testing.SingleQubitGate):  # type: ignore[misc]
        def _t_complexity_(self) -> LegacyTComplexity:
            return LegacyTComplexity()

    class Mismatch(Empty):
        def _t_complexity_(self) -> LegacyTComplexity:
            return LegacyTComplexity(1)

    assert_decomposition_cost(Empty())
    with pytest.raises(AssertionError, match="no _t_complexity_"):
        assert_decomposition_cost(cirq.X)
    with pytest.raises(AssertionError, match="no decomposition"):
        assert_decomposition_cost(Missing())
    with pytest.raises(AssertionError, match="!="):
        assert_decomposition_cost(Mismatch())
