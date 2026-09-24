from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import Circuit
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.core.parameters import Parameter
from flagquantum.ecosystem import get_adapter
from flagquantum.ecosystem.pennylane import (
    PennyLaneConversionError,
    PennyLaneLightningExecutionError,
    from_pennylane,
    import_pennylane,
    run,
    to_pennylane,
)

qml = pytest.importorskip("pennylane")
pytestmark = [pytest.mark.integration, pytest.mark.pennylane]
ROOT = Path(__file__).resolve().parents[1]

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

CONTRACT = tomllib.loads(
    (ROOT / "contracts" / "pennylane-interop-contract.toml").read_text(encoding="utf-8")
)


def _state(script):
    matrix = torch.as_tensor(qml.matrix(script, wire_order=list(script.wires)))
    initial = torch.zeros(matrix.shape[0], dtype=torch.complex128)
    initial[0] = 1
    return matrix.to(torch.complex128) @ initial


def test_pennylane_execution_import_is_lazy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from flagquantum.ecosystem.pennylane import run; "
            "assert callable(run); assert 'pennylane' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_pennylane_execution_statevector_matches_native(dtype: torch.dtype) -> None:
    circuit = Circuit(3, dtype=dtype).h(0).ry(1, 0.37).cx(0, 2).cz(2, 1)

    result = run(circuit, options=fq.ExecutionOptions(seed=13))

    assert result.state is not None
    assert result.state.dtype == dtype
    torch.testing.assert_close(
        result.state,
        circuit.state(refresh=True),
        rtol=1e-5,
        atol=1e-6,
    )
    assert result.runtime["backend"] == "pennylane_lightning_qubit"
    assert result.runtime["fallback_used"] is False
    assert result.compatibility["external_backend_explicit"] is True
    assert result.provenance["flagquantum_ir_hash"] == circuit.to_ir().content_hash
    assert result.provenance["pennylane_lightning_version"]


def test_pennylane_execution_preserves_idle_wire_extent() -> None:
    circuit = Circuit(3, dtype=torch.complex128).x(2)

    result = run(circuit)

    assert result.state is not None
    torch.testing.assert_close(result.state, circuit.state(refresh=True))
    assert result.provenance["idle_wire_extent_preserved"] is True


def test_pennylane_execution_samples_and_counts_use_requested_wire_order() -> None:
    circuit = Circuit(3).x(0).x(2)
    options = fq.ExecutionOptions(shots=8, seed=7)

    samples = run(
        circuit,
        outputs=fq.samples(qubits=(2, 1, 0)),
        options=options,
    )
    counts = run(
        circuit,
        outputs=fq.counts(qubits=(2, 1, 0)),
        options=options,
    )

    assert samples.samples is not None
    assert samples.samples.shape == (1, 8, 3)
    assert samples.samples.unique(dim=1).tolist() == [[[1, 0, 1]]]
    assert counts.counts == [{"101": 8}]

    random_circuit = Circuit(1).h(0)
    first = run(random_circuit, outputs=fq.samples(), options=options)
    second = run(random_circuit, outputs=fq.samples(), options=options)
    assert first.samples is not None
    assert torch.equal(first.samples, second.samples)


def test_pennylane_execution_unsupported_requests_fail_before_device() -> None:
    with pytest.raises(TypeError, match=r"requires fq\.samples"):
        run(Circuit(1), shots=10)
    with pytest.raises(ValueError, match="require shots"):
        run(Circuit(1), outputs=fq.counts())
    with pytest.raises(ValueError, match="must be unique"):
        run(Circuit(2), outputs=fq.samples(qubits=(0, 0)), shots=4)
    with pytest.raises(PennyLaneLightningExecutionError, match=r"supports fq\.samples"):
        run(Circuit(2), outputs=fq.probabilities())
    with pytest.raises(PennyLaneLightningExecutionError, match="dynamic circuit"):
        run(
            CircuitIR(
                1,
                (Instruction("x", (0,), metadata={"conditions": ((0, 1),)}),),
            )
        )
    with pytest.raises(
        PennyLaneLightningExecutionError, match="does not support autograd"
    ):
        run(Circuit(1).rx(0, torch.tensor(0.2, requires_grad=True)))
    with pytest.raises(PennyLaneLightningExecutionError, match="options.backend"):
        run(Circuit(1), options=fq.ExecutionOptions(backend="cirq_simulator"))


def test_quantum_script_round_trip_preserves_complex128_semantics() -> None:
    source = (
        Circuit(3, dtype=torch.complex128)
        .h(0)
        .ry(1, theta=0.231)
        .cx(0, 2)
        .rzz(1, 2, theta=-0.317)
    )
    script = to_pennylane(source.to_ir())
    round_trip = from_pennylane(script)
    assert round_trip.dtype == "complex128"
    assert [item.name for item in round_trip.instructions] == ["h", "ry", "cx", "rzz"]
    expected = Circuit.from_ir(round_trip, dtype=torch.complex128).state()[0]
    torch.testing.assert_close(_state(script), expected, atol=1e-10, rtol=1e-10)
    assert get_adapter("pennylane").name == "pennylane"


@pytest.mark.parametrize("mapping", CONTRACT["operations"])
def test_every_declared_gate_matches_at_complex128(mapping: dict[str, str]) -> None:
    opcode = mapping["flagquantum"]
    schema = OPERATOR_SCHEMAS[opcode]
    parameters = {
        name: 0.173 * (index + 1) for index, name in enumerate(schema.parameters)
    }
    preparations = tuple(
        Instruction("ry", (wire,), {"theta": 0.119 * (wire + 1)})
        for wire in range(schema.arity)
    )
    ir = CircuitIR(
        schema.arity,
        preparations + (Instruction(opcode, tuple(range(schema.arity)), parameters),),
        dtype="complex128",
    )
    script = to_pennylane(ir)
    expected = Circuit.from_ir(ir, dtype=torch.complex128).state()[0]
    torch.testing.assert_close(_state(script), expected, atol=1e-10, rtol=1e-10)


def test_noncontiguous_wires_fail_closed_and_lossy_is_explicit() -> None:
    script = qml.tape.QuantumScript([qml.Hadamard("left"), qml.CNOT(["left", "right"])])
    with pytest.raises(PennyLaneConversionError, match="wire_labels_flattened"):
        from_pennylane(script)
    result = import_pennylane(script, allow_lossy=True)
    assert result.ir.n_wires == 2
    assert result.report.issues[0].code == "wire_labels_flattened"


def test_measurements_shots_and_unsupported_ops_fail_closed() -> None:
    script = qml.tape.QuantumScript(
        [qml.Rot(0.1, 0.2, 0.3, 0)], [qml.state()], shots=10
    )
    with pytest.raises(PennyLaneConversionError) as caught:
        from_pennylane(script)
    codes = {issue.code for issue in caught.value.report.issues}
    assert {
        "finite_shots_not_represented",
        "measurements_not_represented",
        "unsupported_operation",
    } <= codes


def test_symbolic_parameters_and_idle_extent_fail_closed() -> None:
    symbolic = CircuitIR(
        1, (Instruction("rx", (0,), {"theta": Parameter("theta")}),), dtype="complex128"
    )
    with pytest.raises(
        PennyLaneConversionError, match="symbolic_parameter_not_supported"
    ):
        to_pennylane(symbolic)
    idle = CircuitIR(2, (Instruction("x", (0,)),), dtype="complex128")
    with pytest.raises(
        PennyLaneConversionError, match="idle_wire_extent_not_represented"
    ):
        to_pennylane(idle)
