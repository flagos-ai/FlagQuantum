from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Mapping

import pytest

qml = pytest.importorskip("pennylane")
torch = pytest.importorskip("torch")

import flagquantum as fq  # noqa: E402
from flagquantum.interop import semantic_fingerprint  # noqa: E402
from flagquantum.interop.pennylane import (  # noqa: E402
    PennyLaneConversionError,
    export_pennylane,
    import_pennylane,
)

pytestmark = [pytest.mark.integration, pytest.mark.pennylane]
EXTERNAL_MODULES = {"qiskit", "qiskit_aer", "pennylane", "cudaq", "braket"}


def _external_types(value: Any, *, seen: set[int] | None = None) -> set[str]:
    seen = set() if seen is None else seen
    if id(value) in seen:
        return set()
    seen.add(id(value))
    module = type(value).__module__.split(".", 1)[0]
    found = (
        {f"{type(value).__module__}.{type(value).__qualname__}"}
        if module in EXTERNAL_MODULES
        else set()
    )
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            found.update(_external_types(getattr(value, field.name), seen=seen))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            found.update(_external_types(key, seen=seen))
            found.update(_external_types(item, seen=seen))
    elif isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            found.update(_external_types(item, seen=seen))
    return found


def test_pennylane_round_trip_preserves_complex128_and_contains_objects() -> None:
    source = qml.tape.QuantumScript(
        [qml.Hadamard(0), qml.RY(0.231, 1), qml.CNOT((1, 2))]
    )

    imported = import_pennylane(source)
    exported = export_pennylane(imported.ir)
    round_trip = import_pennylane(exported.quantum_script)

    assert isinstance(imported.ir, fq.CircuitIR)
    assert imported.ir.dtype == "complex128"
    assert _external_types(imported) == set()
    assert _external_types(exported.report) == set()
    assert type(exported.quantum_script).__module__.startswith("pennylane")
    assert semantic_fingerprint(imported.ir) == semantic_fingerprint(round_trip.ir)

    matrix = torch.as_tensor(qml.matrix(source, wire_order=[0, 1, 2])).to(
        torch.complex128
    )
    initial = torch.zeros(8, dtype=torch.complex128)
    initial[0] = 1
    expected = fq.Circuit.from_ir(imported.ir, dtype=torch.complex128).state()[0]
    torch.testing.assert_close(matrix @ initial, expected, atol=1e-10, rtol=0)


def test_pennylane_execution_policy_and_unknown_operations_fail_together() -> None:
    source = qml.tape.QuantumScript(
        [qml.Rot(0.1, 0.2, 0.3, 0)],
        measurements=[qml.expval(qml.PauliZ(0))],
        shots=10,
    )

    with pytest.raises(PennyLaneConversionError) as caught:
        import_pennylane(source)

    assert {issue.code for issue in caught.value.report.issues} == {
        "finite_shots_not_represented",
        "measurements_not_represented",
        "unsupported_operation",
    }
    lossy = import_pennylane(source, allow_lossy=True)
    assert lossy.ir.instructions == ()
    assert not lossy.report.lossless
    assert _external_types(lossy) == set()
