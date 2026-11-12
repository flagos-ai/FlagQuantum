"""Coherent affine scalar sums, full translation permutations and clean workspace."""

import math
from pathlib import Path
from typing import TypedDict

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.number_theory_and_cryptography.elliptic_curves.elliptic_curve_discrete_log import (
    main as native,
)
from examples.synthesis.algorithms.number_theory_and_cryptography.elliptic_curves.elliptic_curve_discrete_log.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.algorithms.number_theory_and_cryptography.elliptic_curves.elliptic_curve_discrete_log.curve import (
    add,
    encode,
    translation_steps,
)
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    X,
    adjoint,
    allocate,
    inplace_prepare_amplitudes,
    phase,
    qfunc,
    unitary,
)

pytestmark = pytest.mark.integration
ORIGINAL = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/corpus/original/algorithms/number_theory_and_cryptography/elliptic_curves/elliptic_curve_discrete_log.qmod"
)


class Limits(TypedDict, total=False):
    max_qubits: int
    max_workspace: int
    max_operations: int
    max_depth: int
    memory_limit_bytes: int


def state(program: synth.Program, size: int) -> torch.Tensor:
    raw = fq.run(program).statevector()
    assert isinstance(raw, torch.Tensor)
    viewed = raw.reshape(size, -1)
    assert float(viewed[:, 1:].abs().square().sum()) < 1e-22
    return viewed[:, 0]


@pytest.mark.parametrize("seed", [(4, 2), (3, 2)])
def test_full_scalar_states_source_native_and_independent(
    seed: tuple[int, int],
) -> None:
    program = native.build_program(seed)
    expected = fq.run(reference_circuit(seed)).statevector().flatten()
    torch.testing.assert_close(state(program, 4096), expected, atol=3e-12, rtol=0)
    text = ORIGINAL.read_text().replace("ecp.x ^= 4;", f"ecp.x ^= {seed[0]};")
    source = (
        synth.synthesize(ORIGINAL)
        if seed == (4, 2)
        else synth.synthesize(text, source_format="qmod")
    )
    torch.testing.assert_close(state(source, 4096), expected, atol=3e-12, rtol=0)
    assert program.report["workspace_qubits"] == 3


@qfunc
def translation_trial(
    steps: list[int],
    amplitudes: list[float],
    phi: float,
    undo: int,
    point: Output[QArray[QBit]],
) -> None:
    allocate(6, point)
    inplace_prepare_amplitudes(amplitudes, 0, point)
    for i in range(6):
        phase(point[i], phi * (i + 1))
    native.translation(steps, point)
    if undo == 1:
        with adjoint():
            native.translation(steps, point)


def independent_mapping(addend: tuple[int, int]) -> list[int]:
    # Find each full group orbit using the curve law, including infinity.
    result = list(range(64))
    for word in range(64):
        x, y = word % 8, word // 8
        if word == 63:
            result[word] = addend[0] + 8 * addend[1]
            continue
        if x >= 7 or y >= 7 or (y * y - x * x * x - 5 * x - 4) % 7:
            continue
        u, v = addend
        if x == u and (y + v) % 7 == 0:
            result[word] = 63
            continue
        numerator, denominator = (
            (3 * x * x + 5, 2 * y) if (x, y) == addend else (v - y, u - x)
        )
        slope = (numerator * pow(denominator % 7, -1, 7)) % 7
        out_x = (slope * slope - x - u) % 7
        result[word] = out_x + 8 * ((slope * (x - out_x) - y) % 7)
    assert sorted(result) == list(range(64))
    return result


@pytest.mark.parametrize("addend", [(0, 5), (2, 1), (0, 2)])
@pytest.mark.parametrize("undo", [False, True])
def test_all_translation_words_and_complex_inverse(
    addend: tuple[int, int], undo: bool
) -> None:
    amplitudes = tuple(
        ((-1) ** i) * (i + 1) / math.sqrt(sum(j * j for j in range(1, 65)))
        for i in range(64)
    )
    phi = 0.37
    p = synth.synthesize(
        translation_trial,
        classical_inputs={
            "steps": translation_steps(addend),
            "amplitudes": list(amplitudes),
            "phi": phi,
            "undo": int(undo),
        },
    )
    vector = torch.tensor(
        [
            a
            * complex(
                math.cos(phi * sum(((word >> i) & 1) * (i + 1) for i in range(6))),
                math.sin(phi * sum(((word >> i) & 1) * (i + 1) for i in range(6))),
            )
            for word, a in enumerate(amplitudes)
        ],
        dtype=torch.complex128,
    )
    expected = vector.clone()
    if not undo:
        expected[torch.tensor(independent_mapping(addend))] = vector
    torch.testing.assert_close(state(p, 64), expected, atol=1e-11, rtol=0)


INVERSE = tuple(
    tuple(
        complex(math.cos(-2 * math.pi * r * c / 8), math.sin(-2 * math.pi * r * c / 8))
        / math.sqrt(8)
        for c in range(8)
    )
    for r in range(8)
)


@qfunc
def coherent(
    amplitudes: list[float],
    seed: list[int],
    g0: list[int],
    g1: list[int],
    g2: list[int],
    scalars: Output[QArray[QBit]],
    point: Output[QArray[QBit]],
) -> None:
    allocate(6, scalars)
    allocate(6, point)
    inplace_prepare_amplitudes(amplitudes, 0, scalars)
    for i in range(6):
        if seed[i] == 1:
            X(point[i])
    native.scalar_add(g0, g1, g2, point, scalars[:3])
    native.scalar_add(g0, g1, g2, point, scalars[3:])
    unitary(INVERSE, scalars[:3])
    unitary(INVERSE, scalars[3:])


def bindings(seed: tuple[int, int] = (4, 2)) -> dict[str, list[int]]:
    return {
        "seed": [(encode(seed) >> i) & 1 for i in range(6)],
        "g0": translation_steps((0, 5)),
        "g1": translation_steps((2, 1)),
        "g2": translation_steps((0, 2)),
    }


@pytest.mark.parametrize("seed", [(4, 2), (3, 2)])
def test_nonuniform_coherent_scalars(seed: tuple[int, int]) -> None:
    amplitudes = tuple(
        ((-1) ** i) * (i + 1) / math.sqrt(sum(j * j for j in range(1, 65)))
        for i in range(64)
    )
    p = synth.synthesize(
        coherent, classical_inputs={**bindings(seed), "amplitudes": list(amplitudes)}
    )
    expected = fq.run(reference_circuit(seed, amplitudes)).statevector().flatten()
    torch.testing.assert_close(state(p, 4096), expected, atol=1e-11, rtol=0)


@qfunc
def aliased(
    g0: list[int], g1: list[int], g2: list[int], point: Output[QArray[QBit]]
) -> None:
    allocate(6, point)
    native.scalar_add(g0, g1, g2, point, point[:3])


def test_overlapping_scalar_and_point_refuse() -> None:
    values = bindings()
    values.pop("seed")
    with pytest.raises(HybridVerificationError, match="disjoint|overlap"):
        synth.synthesize(aliased, classical_inputs=values)


@pytest.mark.parametrize("seed", [(0, 0), (0, 5), (7, 7)])
def test_unproved_source_seed_refuses(seed: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="seed"):
        native.build_program(seed)


@pytest.mark.parametrize(
    "limits",
    [
        {"max_qubits": 11},
        {"max_workspace": 0},
        {"max_operations": 100},
        {"max_depth": 100},
        {"memory_limit_bytes": 1024},
    ],
)
def test_budgets_refuse(limits: Limits) -> None:
    with pytest.raises(
        (PlanningError, HybridVerificationError),
        match="budget|limit|requires|exceed|infeasible",
    ):
        synth.synthesize(native.main, classical_inputs=bindings(), **limits)


def test_group_identity_and_inverse() -> None:
    assert add((0, 5), (0, 2)) is None
    assert add(None, (4, 2)) == (4, 2)
    assert add((4, 2), None) == (4, 2)
    assert independent_mapping((0, 5))[63] == encode((0, 5))
