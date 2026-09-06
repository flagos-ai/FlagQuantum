from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Mapping

import pytest

qiskit = pytest.importorskip("qiskit")

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit import Parameter as QiskitParameter  # noqa: E402

import flagquantum as fq  # noqa: E402
from flagquantum.interop.qiskit import (  # noqa: E402
    QiskitConversionError,
    export_qiskit,
    import_qiskit,
    semantic_fingerprint,
)

pytestmark = [pytest.mark.integration, pytest.mark.qiskit]
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


def test_qiskit_round_trip_preserves_owned_semantics_and_contains_objects() -> None:
    theta = QiskitParameter("theta")
    source = QuantumCircuit(3, 2, name="boundary")
    source.global_phase = 0.125
    source.h(0)
    source.ry(theta, 2)
    source.cx(2, 1)
    source.measure(1, 0)

    imported = import_qiskit(source)
    exported = export_qiskit(imported.ir)
    round_trip = import_qiskit(exported.circuit)

    assert isinstance(imported.ir, fq.CircuitIR)
    assert _external_types(imported) == set()
    assert _external_types(exported.report) == set()
    assert exported.circuit.__class__.__module__.startswith("qiskit")
    assert semantic_fingerprint(imported.ir) == semantic_fingerprint(round_trip.ir)
    assert imported.ir.instructions[-1].metadata["classical_bit"] == 0
    assert round_trip.ir.metadata["interop"]["global_phase"] == pytest.approx(0.125)


def test_qiskit_loss_is_machine_readable_and_requires_explicit_opt_in() -> None:
    source = QuantumCircuit(1)
    source.h(0)
    source.barrier(0)

    with pytest.raises(QiskitConversionError) as caught:
        import_qiskit(source)

    assert [issue.code for issue in caught.value.report.issues] == ["barrier_dropped"]
    lossy = import_qiskit(source, allow_lossy=True)
    assert not lossy.report.lossless
    assert [instruction.name for instruction in lossy.ir.instructions] == ["h"]
    assert _external_types(lossy) == set()
