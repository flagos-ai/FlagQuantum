from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Mapping

import pytest

qiskit = pytest.importorskip("qiskit")
torch = pytest.importorskip("torch")

from qiskit import QuantumCircuit, qasm3  # noqa: E402
from qiskit.circuit import Parameter as QiskitParameter  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402

import flagquantum as fq  # noqa: E402
from flagquantum._compiler.offline_deployment import (  # noqa: E402
    OfflineStaticTarget,
    OfflineTextFormat,
    compile_offline_static,
)
from flagquantum._compiler.passes.placement_routing import (  # noqa: E402
    DirectedCouplingGraph,
)
from flagquantum.interop.qiskit import (  # noqa: E402
    QiskitConversionError,
    export_qiskit,
    import_qiskit,
    qiskit_statevector_to_flagquantum,
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


@pytest.mark.parametrize(
    ("output_format", "loader"),
    (
        (OfflineTextFormat.OPENQASM2, QuantumCircuit.from_qasm_str),
        (OfflineTextFormat.OPENQASM3, qasm3.loads),
    ),
)
def test_compiler_openqasm_artifacts_round_trip_through_qiskit(
    output_format: OfflineTextFormat,
    loader: Any,
) -> None:
    source = fq.CircuitIR(
        2,
        (
            fq.Instruction("ry", (0,), {"theta": 0.231}),
            fq.Instruction("rz", (1,), {"theta": -0.417}),
            fq.Instruction("cx", (0, 1)),
        ),
        dtype="complex128",
    )
    target = OfflineStaticTarget(
        DirectedCouplingGraph(2, ((0, 1),)),
        "e" * 64,
    )
    compiled = compile_offline_static(
        source,
        target,
        output_formats=(output_format,),
    )

    assert compiled.ok, compiled.diagnostics
    external = loader(compiled.emission(output_format).text)
    imported = import_qiskit(external)
    external_state = qiskit_statevector_to_flagquantum(
        Statevector.from_instruction(external).data,
        source.n_wires,
    ).to(dtype=fq.run(source).state.dtype)
    flagquantum_state = fq.run(source).state.reshape(-1)

    assert _external_types(imported) == set()
    torch.testing.assert_close(
        external_state,
        flagquantum_state,
        atol=1e-10,
        rtol=0,
    )
