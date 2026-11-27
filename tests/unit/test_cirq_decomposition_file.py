"""Locked Cirq one-step decomposition and resource fallback regressions."""

from hashlib import sha1
from importlib.metadata import version
from pathlib import Path
from typing import Any

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)
import pytest

from flagquantum.compiler._component_trace.qec_gate_cost import (
    QECGateCost,
    query_qec_gate_cost,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    ControlledZPowerCall,
    RotationCall,
)
from flagquantum.ecosystem.cirq import CirqConversionError, export_cirq, import_cirq
from flagquantum.ecosystem.cirq._decompose import (
    decompose_cirq_once,
    import_cirq_decomposition,
)
from flagquantum.ecosystem.cirq.resource_keys import (
    cirq_gate_decomposition_resource_keys,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def cirq() -> Any:
    return pytest.importorskip("cirq")


@pytest.fixture(scope="module")
def locked_source() -> Any:
    import kickmix
    from qualtran import cirq_interop

    assert version("qualtran") == "0.7.0"
    assert getattr(kickmix, "__version__") == "0.1" and hasattr(kickmix, "Simulator")
    folder = Path(cirq_interop.__file__).parent
    for name, digest in (
        ("_cirq_to_bloq.py", "29c7a89a164836eb2e9af2527c2b78294b952e0e"),
        ("_cirq_to_bloq_test.py", "d2c49ee57cc2680b5b545c6673db63b96333184c"),
        ("decompose_protocol.py", "ee269ae02a76bb9e39f8462a8fc685a1856c5ff6"),
        ("decompose_protocol_test.py", "7ea030a32eee0281edb5bd7af3ee78eea943a166"),
    ):
        raw = (folder / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return cirq_interop


@pytest.mark.ftqc_reference
def test_original_empty_protocol_distinguishes_missing_decomposition(
    cirq: Any, locked_source: Any
) -> None:
    from qualtran.cirq_interop.decompose_protocol import (
        _decompose_once_considering_known_decomposition,
    )

    class EmptyGate(cirq.testing.SingleQubitGate):  # type: ignore[misc]
        def _decompose_(self, qubits: Any) -> list[Any]:
            return []

    class OpaqueGate(cirq.testing.SingleQubitGate):  # type: ignore[misc]
        pass

    empty = EmptyGate()
    missing = OpaqueGate()
    for gate, expected in ((empty, []), (missing, None)):
        assert _decompose_once_considering_known_decomposition(gate) == expected
        assert decompose_cirq_once(gate) == expected
        assert decompose_cirq_once(gate.on(cirq.NamedQubit("input"))) == expected

    imported = import_cirq_decomposition(empty)
    assert imported is not None
    assert imported.ir.n_wires == 1
    assert imported.ir.instructions == ()
    assert imported.report.lossless
    assert import_cirq_decomposition(missing) is None
    assert cirq_gate_decomposition_resource_keys(empty) == ()
    assert cirq_gate_decomposition_resource_keys(missing) is None


@pytest.mark.ftqc_reference
def test_decomposed_import_retains_named_operand_and_idle_extent(
    cirq: Any, locked_source: Any
) -> None:
    from qualtran.cirq_interop.decompose_protocol import (
        _decompose_once_considering_known_decomposition,
    )

    class OrderedGate(cirq.Gate):  # type: ignore[misc]
        def _num_qubits_(self) -> int:
            return 3

        def _decompose_(self, qubits: tuple[Any, ...]) -> Any:
            left, _, right = qubits
            return [cirq.H(right), (cirq.CNOT(right, left), [cirq.S(left)])]

    external = (cirq.NamedQubit("z"), cirq.LineQubit(17), cirq.NamedQubit("a"))
    gate = OrderedGate().on(*external)
    decomposition = _decompose_once_considering_known_decomposition(gate)
    assert decompose_cirq_once(gate) == decomposition
    assert decomposition is not None

    result = import_cirq_decomposition(gate)
    assert result is not None
    ir = result.ir
    assert ir.n_wires == 3
    assert ir.metadata["interop"]["qubits"] == ("z", "q(17)", "a")
    assert [(g.name, g.wires) for g in ir.instructions] == [
        ("h", (2,)),
        ("cx", (2, 0)),
        ("s", (0,)),
    ]
    # Static export reports the idle wire. The explicit order still allows a full unitary comparison.
    exported = export_cirq(ir, allow_lossy=True)
    assert {issue.code for issue in exported.report.issues} == {
        "idle_wire_extent_not_represented"
    }
    actual = exported.circuit.unitary(
        qubit_order=cirq.LineQubit.range(3),
        qubits_that_should_be_present=cirq.LineQubit.range(3),
    )
    expected = cirq.Circuit(decomposition).unitary(
        qubit_order=external, qubits_that_should_be_present=external
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-12)
    with pytest.raises(CirqConversionError, match="unsupported_operation"):
        import_cirq(cirq.Circuit(OrderedGate().on(*cirq.LineQubit.range(3))))


def test_decomposition_context_uses_fresh_work_and_import_requires_lifetimes(
    cirq: Any,
) -> None:
    class WorkGate(cirq.testing.SingleQubitGate):  # type: ignore[misc]
        def _decompose_with_context_(
            self, qubits: tuple[Any, ...], context: Any
        ) -> Any:
            (work,) = context.qubit_manager.qalloc(1)
            yield cirq.CNOT(qubits[0], work)
            yield cirq.CNOT(qubits[0], work)
            context.qubit_manager.qfree([work])

    q = cirq.NamedQubit("data")
    operation = WorkGate().on(q)
    first = decompose_cirq_once(operation)
    second = decompose_cirq_once(operation)
    assert first is not None and second is not None
    assert [op.qubits[0] for op in first + second] == [q, q, q, q]
    assert first[0].qubits[1] == first[1].qubits[1]
    assert second[0].qubits[1] != first[0].qubits[1]
    with pytest.raises(ValueError, match="lifetime binding"):
        import_cirq_decomposition(operation)


def test_decomposition_does_not_strip_tags_controls_or_unknown_leaves(
    cirq: Any,
) -> None:
    class Cycle(cirq.testing.SingleQubitGate):  # type: ignore[misc]
        def _decompose_(self, qubits: tuple[Any, ...]) -> Any:
            return [self.on(*qubits)]

    class TaggedLeaf(cirq.testing.SingleQubitGate):  # type: ignore[misc]
        def _decompose_(self, qubits: tuple[Any, ...]) -> Any:
            return [cirq.X.on(*qubits).with_tags("opaque")]

    q = cirq.NamedQubit("q")
    with pytest.raises(TypeError, match="untagged gate"):
        import_cirq_decomposition(TaggedLeaf().on(q).with_tags("external"))
    with pytest.raises(TypeError, match="untagged gate"):
        import_cirq_decomposition(TaggedLeaf().on(q).with_classical_controls("m"))
    with pytest.raises(CirqConversionError, match="unsupported_operation"):
        import_cirq_decomposition(TaggedLeaf())
    assert cirq_gate_decomposition_resource_keys(Cycle()) is None
    assert cirq_gate_decomposition_resource_keys(TaggedLeaf()) is None


@pytest.mark.ftqc_reference
def test_original_symbolic_swap_power_decomposition_resources(
    cirq: Any, locked_source: Any
) -> None:
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from sympy import Symbol

    theta = Symbol("theta", real=True)
    swap = cirq.SwapPowGate(exponent=theta)
    leaves = cirq_gate_decomposition_resource_keys(swap)
    assert leaves is not None
    assert leaves == (
        CliffordCall("cx", 2),
        RotationCall("ypow", -0.5),
        ControlledZPowerCall(theta),
        RotationCall("ypow", 0.5),
        CliffordCall("cx", 2),
    )
    cost = QECGateCost()
    for leaf in leaves:
        cost = cost.plus(query_qec_gate_cost(leaf), 1)
    source = get_cost_value(locked_source.CirqGateAsBloq(swap), QECGatesCost())
    assert cost == QECGateCost(clifford=5, rotation=1, and_bloq=1, measurement=1)
    assert source.asdict() == {
        name: count for name, count in vars(cost).items() if count
    }
