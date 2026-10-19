"""Independent QCIS matrices and ordinary FlagQuantum execution workflows."""

from __future__ import annotations

import cmath
import math
import subprocess
import sys

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler.qcis import emit_qcis
from flagquantum.ecosystem.qcis import import_qcis

pytestmark = pytest.mark.integration


def test_keyword_angle_executes_asymmetric_measurements_and_gradients() -> None:
    converted = import_qcis("RY Q7 (lambda/2)\nX Q17\nM Q17 Q7")
    circuit = fq.Circuit.from_ir(converted.ir)
    assert circuit.parameter_names == ("lambda",)
    angle = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    bound = circuit.bind_parameters({"lambda": angle})
    result = fq.run(bound, outputs=fq.probabilities((1, 0)))
    expected = torch.tensor(
        [[0.0, 0.0, math.cos(0.6 / 4) ** 2, math.sin(0.6 / 4) ** 2]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(result.probabilities, expected)
    result.probabilities[0, 3].backward()
    assert angle.grad is not None
    torch.testing.assert_close(
        angle.grad, torch.tensor(math.sin(0.6 / 2) / 4, dtype=torch.float64)
    )


def _matrix(name: str) -> torch.Tensor:
    identity = torch.eye(2, dtype=torch.complex128)
    pauli = {
        "X": torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
        "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),
        "Z": torch.diag(torch.tensor([1, -1], dtype=torch.complex128)),
    }
    if name in pauli:
        return pauli[name]
    if name == "H":
        return (pauli["X"] + pauli["Z"]) / math.sqrt(2)
    if name in {"S", "SD", "T", "TD"}:
        angle = {
            "S": math.pi / 2,
            "SD": -math.pi / 2,
            "T": math.pi / 4,
            "TD": -math.pi / 4,
        }[name]
        return torch.diag(
            torch.tensor(
                [1, complex(math.cos(angle), math.sin(angle))], dtype=torch.complex128
            )
        )
    axis = name[1] if name.startswith("R") else name[0]
    angle = (
        0.713
        if name.startswith("R")
        else (-1 if name.endswith("M") else 1) * math.pi / 2
    )
    return math.cos(angle / 2) * identity - 1j * math.sin(angle / 2) * pauli[axis]


@pytest.mark.parametrize(
    "name",
    [
        "X",
        "Y",
        "Z",
        "H",
        "S",
        "SD",
        "T",
        "TD",
        "X2P",
        "X2M",
        "Y2P",
        "Y2M",
        "RX",
        "RY",
        "RZ",
    ],
)
def test_each_gate_matches_independent_matrix_columns(name: str) -> None:
    expected = _matrix(name)
    for bit in (0, 1):
        source = (
            ("X Q3\n" if bit else "")
            + f"{name} Q3"
            + (" 0.713" if name.startswith("R") else "")
        )
        actual = fq.run(import_qcis(source).ir).state.reshape(-1)
        torch.testing.assert_close(actual, expected[:, bit], atol=1e-12, rtol=1e-12)


def test_native_bell_fixture_preserves_relative_phase_and_counts() -> None:
    text = "Y2P Q4\nY2M Q9\nCZ Q4 Q9\nY2P Q9\nM Q9 Q4"
    ir = import_qcis(text).ir
    expected = torch.tensor([1, 0, 0, 1], dtype=torch.complex128) / math.sqrt(2)
    torch.testing.assert_close(
        fq.run(ir).state.reshape(-1), expected, atol=1e-12, rtol=1e-12
    )
    result = fq.run(
        fq.Circuit.from_ir(ir), outputs=fq.counts(ir.measurements[0].wires), shots=128
    )
    assert set(result.counts[0]) <= {"00", "11"}
    assert sum(result.counts[0].values()) == 128


def test_asymmetric_measurement_order_is_not_sorted_with_labels() -> None:
    ir = import_qcis("X Q100\nM Q100 Q2").ir
    assert fq.run(ir).probabilities.tolist() == [[0.0, 0.0, 1.0, 0.0]]
    assert fq.run(
        fq.Circuit.from_ir(ir), outputs=fq.counts(ir.measurements[0].wires), shots=16
    ).counts[0] == {"10": 16}


def test_cz_phase_is_observable_in_interference() -> None:
    ir = import_qcis("H Q7\nX Q103\nCZ Q7 Q103\nH Q7").ir
    assert ir.n_wires == 2
    assert ir.metadata["interop"]["qubits"] == ("Q7", "Q103")
    expected = torch.tensor([0, 0, 0, 1], dtype=torch.complex128)
    torch.testing.assert_close(
        fq.run(ir).state.reshape(-1), expected, atol=1e-12, rtol=1e-12
    )


@pytest.mark.parametrize("symbolic", [False, True])
def test_mixed_sequence_against_independent_amplitudes(symbolic: bool) -> None:
    theta, phi, beta = 0.713, -0.391, 0.827
    source = (
        "H Q7\nRX Q103 theta\nCZ Q7 Q103\nRZ Q7 phi\nRY Q103 beta"
        if symbolic
        else f"H Q7\nRX Q103 {theta}\nCZ Q7 Q103\nRZ Q7 {phi}\nRY Q103 {beta}"
    )
    ir = import_qcis(source).ir
    program = fq.Circuit.from_ir(ir)
    if symbolic:
        program = program.bind_parameters({"theta": theta, "phi": phi, "beta": beta})
    # Expand H and RX on |00>, flip the |11> phase with CZ, then apply
    # the RZ phases and RY's real two-component rotation to each pair.
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    u, v = math.cos(beta / 2), math.sin(beta / 2)
    minus, plus = cmath.exp(-0.5j * phi), cmath.exp(0.5j * phi)
    expected = torch.tensor(
        [
            minus * (u * c + 1j * v * s),
            minus * (v * c - 1j * u * s),
            plus * (u * c - 1j * v * s),
            plus * (v * c + 1j * u * s),
        ],
        dtype=torch.complex128,
    ) / math.sqrt(2)
    torch.testing.assert_close(
        fq.run(program).state.reshape(-1), expected, atol=1e-12, rtol=1e-12
    )


def test_existing_exporter_round_trip_up_to_global_phase() -> None:
    # Compatibility with the existing exporter, not an independent physics oracle.
    circuit = (
        fq.Circuit(3)
        .h(0)
        .cx(0, 2)
        .rx(1, theta=0.417)
        .ry(2, theta=-0.63)
        .rz(0, theta=0.29)
    )
    expected = fq.run(circuit).state.reshape(-1).to(torch.complex128)
    actual = fq.run(import_qcis(emit_qcis(circuit)).ir).state.reshape(-1)
    overlap = torch.vdot(expected, actual)
    # The existing exporter writes pi to six decimals; this is its precision.
    torch.testing.assert_close(
        actual, expected * overlap / overlap.abs(), atol=1e-6, rtol=1e-6
    )


def test_importer_does_not_load_execution_or_external_toolchains() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from flagquantum.ecosystem.qcis import import_qcis
import_qcis('X Q0')
for prefix in ('flagquantum.runtime', 'flagquantum.simulation', 'docker'):
    assert not any(n == prefix or n.startswith(prefix + '.') for n in sys.modules), prefix
""",
        ],
        check=True,
        timeout=30,
    )


@pytest.mark.parametrize("phi,theta", [(0.391, 0.713), (-1.113, -0.827), (2.1, 4.3)])
def test_rxy_matches_independent_matrix_columns(phi: float, theta: float) -> None:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    expected = torch.tensor(
        [
            [c, -1j * cmath.exp(-1j * phi) * s],
            [-1j * cmath.exp(1j * phi) * s, c],
        ],
        dtype=torch.complex128,
    )
    for bit in (0, 1):
        source = ("X Q7\n" if bit else "") + f"RXY Q7 {phi} {theta}"
        actual = fq.run(import_qcis(source).ir).state.reshape(-1)
        torch.testing.assert_close(actual, expected[:, bit], atol=1e-12, rtol=1e-12)


def test_parameterized_import_binds_shared_angles_and_preserves_autograd() -> None:
    from flagquantum.core.ir import CircuitIR

    imported = import_qcis(
        "RXY Q7 (phi + theta/2) (theta*theta)\nRZ Q7 (theta+phi)\nH Q7\nM Q7"
    )
    restored = CircuitIR.from_json(imported.ir.to_json())
    circuit = fq.Circuit.from_ir(restored)
    assert circuit.parameter_names == ("phi", "theta")
    theta = torch.tensor(0.713, dtype=torch.float64, requires_grad=True)
    phi = torch.tensor(-0.391, dtype=torch.float64, requires_grad=True)
    bound = circuit.bind_parameters({"theta": theta, "phi": phi})
    assert not bound.is_parameterized()
    # Circuit.from_ir carries gates only: explicitly request the source's M Q7.
    result = fq.run(bound, outputs=fq.probabilities((0,)))
    t, p = float(theta.detach()), float(phi.detach())
    delta = 1.5 * t + 2 * p
    p0 = (1 + math.sin(t * t) * math.sin(delta)) / 2
    torch.testing.assert_close(
        result.probabilities, torch.tensor([[p0, 1 - p0]], dtype=torch.float64)
    )
    result.probabilities[0, 0].backward()
    dt = t * math.cos(t * t) * math.sin(delta) + 0.75 * math.sin(t * t) * math.cos(
        delta
    )
    dp = math.sin(t * t) * math.cos(delta)
    assert theta.grad is not None and phi.grad is not None
    torch.testing.assert_close(
        theta.grad, torch.tensor(dt, dtype=torch.float64), atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        phi.grad, torch.tensor(dp, dtype=torch.float64), atol=1e-12, rtol=1e-12
    )
    # Rebinding must leave the symbolic template reusable.
    rebound = circuit.bind_parameters({"theta": -0.827, "phi": 1.113})
    other_p0 = (1 + math.sin(0.827**2) * math.sin(1.5 * (-0.827) + 2 * 1.113)) / 2
    actual = fq.run(rebound, outputs=fq.probabilities((0,))).probabilities
    torch.testing.assert_close(
        actual, torch.tensor([[other_p0, 1 - other_p0]], dtype=torch.float64)
    )
