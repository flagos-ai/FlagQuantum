"""Time marching full states, packed bit order and coherent counter ownership."""

import ast
import math
import re
from pathlib import Path
from typing import Annotated, TypedDict

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_differential_equations_solvers.time_marching import (
    main as native,
)
from examples.synthesis.algorithms.quantum_differential_equations_solvers.time_marching.comparisons.circuit import (
    counter_matrix,
    reference_circuit,
)
from examples.synthesis.algorithms.quantum_differential_equations_solvers.time_marching.data import (
    EVEN_PHASES,
    ODD_PHASES,
)
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    QNum,
    R,
    adjoint,
    allocate,
    control,
    inplace_prepare_amplitudes,
    phase,
    qfunc,
)

pytestmark = pytest.mark.integration
ORIGINAL = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/corpus/original/algorithms/quantum_differential_equations_solvers/time_marching/time_marching.qmod"
)


class Limits(TypedDict, total=False):
    max_qubits: int
    max_workspace: int
    max_operations: int
    max_depth: int
    memory_limit_bytes: int


def state(program: synth.Program) -> torch.Tensor:
    raw = fq.run(program).statevector()
    assert isinstance(raw, torch.Tensor)
    viewed = raw.reshape(2048, 2, -1)
    assert float(viewed[:, 1:].abs().square().sum()) < 1e-22
    assert float(viewed[:, :, 1:].abs().square().sum()) < 1e-22
    return viewed[:, 0, 0]


@pytest.mark.parametrize("duration,delta", [(2.0, 0.0), (1.3, 0.017), (2.7, -0.023)])
def test_native_original_and_independent_full_states(
    duration: float, delta: float
) -> None:
    even = tuple(v + delta * (i + 1) for i, v in enumerate(EVEN_PHASES))
    odd = tuple(v - delta * (i + 1) for i, v in enumerate(ODD_PHASES))
    expected = fq.run(reference_circuit(duration, even, odd)).statevector().flatten()
    program = native.build_program(duration, even, odd)
    torch.testing.assert_close(state(program), expected, atol=1e-10, rtol=0)
    assert len(program.registers["index"]) == len(program.registers["time"]) == 2
    assert (
        len(program.registers["counter"]) == 3 and len(program.registers["unused"]) == 1
    )
    text = ORIGINAL.read_text()
    match = re.search(r"qsvt_lcu_expanded___0\((\[.*?\]), (\[.*?\]), qbe", text, re.S)
    assert match
    assert tuple(ast.literal_eval(match[1])) == EVEN_PHASES
    assert tuple(ast.literal_eval(match[2])) == ODD_PHASES
    text = (
        text[: match.start(1)]
        + repr(list(even))
        + ", "
        + repr(list(odd))
        + text[match.end(2) :]
    )
    text = text.replace(
        "long_time_integrator_naive_expanded___0(2, 4,",
        "long_time_integrator_naive_expanded___0(" + repr(duration) + ", 4,",
    )
    original = (
        synth.synthesize(ORIGINAL)
        if delta == 0 and duration == 2
        else synth.synthesize(text, source_format="qmod")
    )
    raw = fq.run(original).statevector().reshape(2048, -1)
    assert float(raw[:, 1:].abs().square().sum()) < 1e-22
    # Frozen struct fields occupy low-to-high bits; native named ports are MSB-first.
    mapped = (
        raw[:, 0].reshape(8, 2, 2, 2, 2, 4, 4).permute(6, 5, 4, 3, 2, 1, 0).flatten()
    )
    torch.testing.assert_close(mapped, expected, atol=1e-10, rtol=0)


@qfunc
def coherent(
    phi: float,
    index: QArray[QBit],
    time: QArray[QBit],
    block: QBit,
    aux: QBit,
    selector: QBit,
    unused: QBit,
    counter: QNum,
    amplitudes: list[float],
) -> None:
    for i in range(2):
        R(0.31 * (i + 1), math.pi / 2, index[i])
        phase(index[i], phi * (i + 1))
        R(0.43 * (i + 1), math.pi / 2, time[i])
        phase(time[i], -phi * (i + 1))
    R(0.71, math.pi / 2, block)
    phase(block, phi)
    R(0.83, math.pi / 2, aux)
    phase(aux, -phi)
    R(1.17, math.pi / 2, selector)
    phase(selector, 2 * phi)
    R(0.63, math.pi / 2, unused)
    inplace_prepare_amplitudes(amplitudes, 0, counter)
    phase(counter, phi)


@qfunc
def counter_trial(
    phi: float,
    amplitudes: list[float],
    index: Output[QArray[QBit]],
    time: Output[QArray[QBit]],
    block: Output[QBit],
    aux: Output[QBit],
    selector: Output[QBit],
    unused: Output[QBit],
    counter: Output[Annotated[QNum, 3]],
    marker: Output[QBit],
) -> None:
    allocate(2, index)
    allocate(2, time)
    allocate(block)
    allocate(aux)
    allocate(selector)
    allocate(unused)
    allocate(counter)
    allocate(marker)
    coherent(phi, index, time, block, aux, selector, unused, counter, amplitudes)
    native.toggle_success(time, block, aux, selector, marker)
    control(marker, lambda: native.decrement(counter))
    native.toggle_success(time, block, aux, selector, marker)


@qfunc
def inverse_trial(
    phi: float,
    amplitudes: list[float],
    even: list[float],
    odd: list[float],
    index: Output[QArray[QBit]],
    time: Output[QArray[QBit]],
    block: Output[QBit],
    aux: Output[QBit],
    selector: Output[QBit],
    unused: Output[QBit],
    counter: Output[Annotated[QNum, 3]],
    marker: Output[QBit],
) -> None:
    allocate(2, index)
    allocate(2, time)
    allocate(block)
    allocate(aux)
    allocate(selector)
    allocate(unused)
    allocate(counter)
    allocate(marker)
    coherent(phi, index, time, block, aux, selector, unused, counter, amplitudes)
    native.step(even, odd, 0.3, 1.1, index, time, block, aux, selector)
    with adjoint():
        native.step(even, odd, 0.3, 1.1, index, time, block, aux, selector)


def coherent_circuit(phi: float, amplitudes: list[float]) -> fq.Circuit:
    circuit = fq.Circuit(11, dtype=torch.complex128)
    for i in range(2):
        circuit.ry(1 - i, theta=0.31 * (i + 1)).phase(1 - i, theta=phi * (i + 1))
        circuit.ry(3 - i, theta=0.43 * (i + 1)).phase(3 - i, theta=-phi * (i + 1))
    for wire, angle, coefficient in [
        (4, 0.71, 1),
        (5, 0.83, -1),
        (6, 1.17, 2),
        (7, 0.63, 0),
    ]:
        circuit.ry(wire, theta=angle).phase(wire, theta=coefficient * phi)
    vector = torch.tensor(amplitudes, dtype=torch.complex128)
    eye = torch.eye(8, dtype=torch.complex128)
    delta = eye[:, 0] - vector
    circuit.any(
        8, 9, 10, matrix=eye - 2 * torch.outer(delta, delta.conj()) / delta.vdot(delta)
    )
    for bit in range(3):
        circuit.phase(10 - bit, theta=phi * (1 << bit))
    return circuit


@pytest.mark.parametrize("phi", [0.37, -0.61])
@pytest.mark.parametrize("inverse", [False, True])
def test_nonzero_coherent_counter_wrap_or_borrowed_inverse(
    phi: float, inverse: bool
) -> None:
    amplitudes = [(i + 1) / math.sqrt(204) for i in range(8)]
    inputs: dict[str, list[float] | float] = {"phi": phi, "amplitudes": amplitudes}
    if inverse:
        inputs.update(even=list(EVEN_PHASES), odd=list(ODD_PHASES))
    p = synth.synthesize(
        inverse_trial if inverse else counter_trial,
        classical_inputs=inputs,
        memory_limit_bytes=64 * 1024**2,
    )
    circuit = coherent_circuit(phi, amplitudes)
    if not inverse:
        circuit.any(2, 3, 4, 5, 6, 8, 9, 10, matrix=counter_matrix())
    torch.testing.assert_close(
        state(p), fq.run(circuit).statevector().flatten(), atol=1e-10, rtol=0
    )


@qfunc
def aliased(
    even: list[float],
    odd: list[float],
    index: Output[QArray[QBit]],
    time: Output[QArray[QBit]],
    bit: Output[QBit],
) -> None:
    allocate(2, index)
    allocate(2, time)
    allocate(bit)
    native.step(even, odd, 0.0, 0.5, index, time, bit, bit, bit)


def test_cross_call_alias_refuses() -> None:
    with pytest.raises(HybridVerificationError, match="disjoint"):
        synth.synthesize(
            aliased,
            classical_inputs={"even": list(EVEN_PHASES), "odd": list(ODD_PHASES)},
        )


@pytest.mark.parametrize("duration", [0.0, -1.0, 5.0, float("nan"), float("inf"), True])
def test_invalid_duration_refuses(duration: float) -> None:
    with pytest.raises(ValueError, match="duration"):
        native.build_program(duration)


@pytest.mark.parametrize(
    "even,odd",
    [
        (EVEN_PHASES[:-1], ODD_PHASES),
        (EVEN_PHASES, ODD_PHASES[:-1]),
        ((float("nan"),) + EVEN_PHASES[1:], ODD_PHASES),
        ((True,) + EVEN_PHASES[1:], ODD_PHASES),
    ],
)
def test_invalid_phases_refuse(even: tuple[float, ...], odd: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="phase"):
        native.build_program(even=even, odd=odd)


@pytest.mark.parametrize(
    "limits",
    [
        {"max_qubits": 12},
        {"max_workspace": 0},
        {"max_operations": 100},
        {"max_depth": 100},
        {"memory_limit_bytes": 1024},
    ],
)
def test_resource_budgets_refuse(limits: Limits) -> None:
    with pytest.raises(
        (PlanningError, HybridVerificationError),
        match="budget|limit|requires|exceed|infeasible",
    ):
        synth.synthesize(
            native.main, classical_inputs=native.classical_inputs(), **limits
        )
