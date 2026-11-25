"""Execute native exact, sampled, noisy, and MPS tutorial paths."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import torch

import flagquantum as fq
from examples.synthesis.tutorials.basics.quantum_programming.execution_tutorial.main import (
    main as square_plus_one,
)
from flagquantum import synth
from flagquantum.compiler.basis_conversion import convert_basis
from flagquantum.core.ir import CircuitIR
from flagquantum.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_channel,
    two_qubit_depolarizing_channel,
)
from flagquantum.services import capabilities
from flagquantum.synth import CX, H, Output, QArray, QBit, allocate, qfunc

SEED = 7
DEFAULT_SHOTS = 2048
LARGE_SHOTS = 10_000
Mode = Literal["statevector", "density_matrix", "mps"]


@dataclass(frozen=True)
class Calibration:
    device: str
    release: str
    date: str
    basis: tuple[str, ...]
    errors: dict[tuple[str, tuple[int, ...]], float]
    readout: tuple[tuple[float, float], ...]


def calibration() -> Calibration:
    data = json.loads(
        Path(__file__).with_name("pittsburgh_calibration.json").read_text()
    )
    return Calibration(
        data["device"],
        data["snapshot_release"],
        data["properties_date"],
        tuple(data["basis"]),
        {(name, tuple(wires)): error for name, wires, error in data["gates"]},
        tuple((p01, p10) for p01, p10 in data["readout"]),
    )


def build_square() -> synth.Program:
    return synth.synthesize(
        square_plus_one, max_qubits=10, max_workspace=1, memory_limit_bytes=1024**2
    )


@qfunc
def ghz(output: Output[QArray[QBit]]) -> None:
    allocate(3, output)
    H(output[0])
    CX(output[0], output[1])
    CX(output[1], output[2])


def build_ghz() -> synth.Program:
    return synth.synthesize(
        ghz, max_qubits=3, max_workspace=0, memory_limit_bytes=1024**2
    )


def square_counts(program: synth.Program, shots: int) -> dict[tuple[int, int], int]:
    x, y = program.registers["x"], program.registers["y"]
    result = fq.run(
        program.circuit,
        options=fq.ExecutionOptions(mode="statevector", shots=shots, seed=SEED),
        outputs=fq.counts((*x, *y)),
    )
    return {
        (int(str(word)[: len(x)], 2), int(str(word)[len(x) :], 2)): count
        for word, count in result.counts[0].items()
    }


def native_basis(program: synth.Program, profile: Calibration) -> CircuitIR:
    return convert_basis(program.circuit, gates=profile.basis).program


def gate_noise(circuit: CircuitIR, profile: Calibration) -> NoiseModel:
    model = NoiseModel()
    gates = {(op.name, op.wires) for op in circuit.instructions}
    for name, wires in sorted(gates):
        error = profile.errors[name, wires]
        if error:
            channel = (
                depolarizing_channel(
                    torch.tensor(1.5 * error, dtype=torch.float64),
                    dtype=torch.complex128,
                )
                if len(wires) == 1
                else two_qubit_depolarizing_channel(
                    torch.tensor(1.25 * error, dtype=torch.float64),
                    dtype=torch.complex128,
                )
            )
            model.add(name, channel, qubits=wires)
    return model


def noisy_ghz(
    circuit: CircuitIR, profile: Calibration
) -> tuple[dict[str | int, int], torch.Tensor, torch.Tensor]:
    model = gate_noise(circuit, profile)
    prepared = fq.run(
        circuit,
        options=fq.ExecutionOptions(mode="density_matrix"),
        outputs=fq.density_matrix(),
        noise_model=model,
    )
    for wire, (p01, p10) in enumerate(profile.readout):
        model.add_readout(wire, ReadoutError(((1 - p01, p01), (p10, 1 - p10))))
    observed = fq.run(
        circuit,
        options=fq.ExecutionOptions(
            mode="density_matrix", shots=DEFAULT_SHOTS, seed=SEED
        ),
        outputs=(fq.counts(), fq.probabilities()),
        noise_model=model,
    )
    return (
        observed.counts[0],
        observed.probabilities.reshape(8),
        prepared.density_matrix.reshape(8, 8),
    )


def ideal_state(program: synth.Program) -> torch.Tensor:
    result = fq.run(program.circuit, options=fq.ExecutionOptions(mode="statevector"))
    return cast(torch.Tensor, result.statevector()).reshape(-1)


def mps_counts(program: synth.Program) -> dict[str | int, int]:
    result = fq.run(
        program.circuit,
        options=fq.ExecutionOptions(mode="mps", shots=DEFAULT_SHOTS, seed=SEED),
        outputs=fq.counts(),
    )
    return cast(dict[str | int, int], result.counts[0])


def backend_names() -> tuple[str, ...]:
    return tuple(sorted(cast(dict[str, object], capabilities()["backends"])))
