"""Unit tests for two-qubit KAK synthesis into a supercontrolled entangler basis."""

from __future__ import annotations

import cmath
import random

import pytest
import torch

from flagquantum.compiler.two_qubit_synthesis import (
    SUPERCONTROLLED_ENTANGLERS,
    synthesize_two_qubit,
)
from flagquantum.core.ir import Instruction
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.simulation.gate_matrix import gate_matrix

pytestmark = pytest.mark.unit

_TWO_QUBIT_UNITARIES = tuple(
    sorted(
        name
        for name, schema in OPERATOR_SCHEMAS.items()
        if schema.unitary and schema.arity == 2
    )
)
# The register index of `wires == (0, 1)` counts wire 0 as the most significant
# bit, so exchanging the two index bits is the same as relabelling the two wires.
_SWAP_PERMUTATION = torch.tensor(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.complex128
)


def _matrix(instruction: Instruction) -> torch.Tensor:
    width = 2 ** len(instruction.wires)
    matrix = gate_matrix(
        instruction, bsz=1, device=torch.device("cpu"), dtype=torch.complex128
    )
    return matrix.reshape(-1, width, width)[0]


def _source(name: str, angles: dict[str, float] | None = None) -> torch.Tensor:
    schema = OPERATOR_SCHEMAS[name]
    params = angles if angles is not None else dict.fromkeys(schema.parameters, 0.7137)
    return _matrix(Instruction(name, (0, 1), params=params))


# The register order the emitted leaves are lifted into: `register[0]` is the
# most significant index bit, exactly as `synthesize_two_qubit` reads `wires`.
_DEFAULT_REGISTER = (0, 1)


def _embed(instruction: Instruction, register: tuple[int, int]) -> torch.Tensor:
    """Lift one emitted instruction into the register `register`, first wire first."""

    matrix = _matrix(instruction)
    if len(instruction.wires) == 2:
        return matrix
    identity = torch.eye(2, dtype=torch.complex128)
    if instruction.wires[0] == register[0]:
        return torch.kron(matrix, identity)
    return torch.kron(identity, matrix)


def _product(
    emitted: tuple[Instruction, ...], register: tuple[int, int] = _DEFAULT_REGISTER
) -> torch.Tensor:
    total = torch.eye(4, dtype=torch.complex128)
    for instruction in emitted:
        total = _embed(instruction, register) @ total
    return total


def _entanglers(emitted: tuple[Instruction, ...], entangler: str) -> int:
    return sum(1 for item in emitted if item.name == entangler)


def _overlap(
    emitted: tuple[Instruction, ...],
    source: torch.Tensor,
    register: tuple[int, int] = _DEFAULT_REGISTER,
) -> torch.Tensor:
    return torch.trace(source.conj().T @ _product(emitted, register)) / 4


def _gap(
    emitted: tuple[Instruction, ...],
    source: torch.Tensor,
    register: tuple[int, int] = _DEFAULT_REGISTER,
) -> float:
    """Max entry of the residual once the single global phase is divided out."""

    overlap = _overlap(emitted, source, register)
    assert abs(overlap) > 0.5, "the synthesized product is not close to the source"
    return float(torch.max(torch.abs(source - _product(emitted, register) / overlap)))


def _random_su4(seed: int) -> list[list[complex]]:
    rng = random.Random(seed)
    rows = [
        [complex(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(4)] for _ in range(4)
    ]
    orthonormal: list[torch.Tensor] = []
    for column in range(4):
        vector = torch.tensor(
            [rows[row][column] for row in range(4)], dtype=torch.complex128
        )
        for previous in orthonormal:
            vector = vector - torch.vdot(previous, vector) * previous
        orthonormal.append(vector / torch.linalg.vector_norm(vector))
    matrix = torch.stack(orthonormal, dim=1)
    determinant = cmath.phase(complex(torch.linalg.det(matrix)))
    matrix = matrix * cmath.exp(-1j * determinant / 4)
    return [[complex(item) for item in row] for row in matrix.tolist()]


def test_the_declared_two_qubit_group_is_the_whole_arity_two_unitary_group() -> None:
    # Non-vacuity: the reach below only describes the two-qubit group if the group
    # really is every declared arity-2 unitary, not a convenient subset.
    assert len(_TWO_QUBIT_UNITARIES) == 11
    assert _TWO_QUBIT_UNITARIES == (
        "cphase",
        "crx",
        "cry",
        "crz",
        "cx",
        "cy",
        "cz",
        "rxx",
        "ryy",
        "rzz",
        "swap",
    )
    assert SUPERCONTROLLED_ENTANGLERS == ("cx", "cz")


@pytest.mark.parametrize("entangler", SUPERCONTROLLED_ENTANGLERS)
@pytest.mark.parametrize("z_rotation", ("rz", "phase", "u1"))
def test_every_declared_two_qubit_unitary_reaches_the_entangler_basis(
    entangler: str, z_rotation: str
) -> None:
    """Every declared arity-2 unitary reconstructs, with a pinned gate count."""

    counts: dict[str, int] = {}
    lengths: dict[str, int] = {}
    for name in _TWO_QUBIT_UNITARIES:
        source = _source(name)
        emitted = synthesize_two_qubit(
            source.tolist(),
            wires=(0, 1),
            entangler=entangler,
            z_rotation=z_rotation,
        )
        assert emitted is not None, f"{name} has no synthesis into {entangler}"
        assert _gap(emitted, source) < 1e-9
        counts[name] = _entanglers(emitted, entangler)
        lengths[name] = len(emitted)
    assert set(counts) == set(_TWO_QUBIT_UNITARIES)
    assert min(counts.values()) >= 1
    assert max(counts.values()) <= 3
    assert max(lengths.values()) <= 30


def test_the_entangler_count_distribution_is_pinned() -> None:
    """The trace choice is a fixed function of the target's Weyl point."""

    distribution = {}
    for name in _TWO_QUBIT_UNITARIES:
        emitted = synthesize_two_qubit(
            _source(name).tolist(), wires=(0, 1), entangler="cx", z_rotation="rz"
        )
        assert emitted is not None
        distribution[name] = _entanglers(emitted, "cx")
    assert distribution == {
        "cphase": 2,
        "crx": 2,
        "cry": 2,
        "crz": 2,
        "cx": 1,
        "cy": 1,
        "cz": 1,
        "rxx": 2,
        "ryy": 2,
        "rzz": 2,
        "swap": 3,
    }


@pytest.mark.parametrize("seed", range(8))
def test_a_random_special_unitary_reconstructs_up_to_one_global_phase(
    seed: int,
) -> None:
    matrix = _random_su4(seed)
    source = torch.tensor(matrix, dtype=torch.complex128)
    for entangler in SUPERCONTROLLED_ENTANGLERS:
        emitted = synthesize_two_qubit(
            matrix, wires=(0, 1), entangler=entangler, z_rotation="rz"
        )
        assert emitted is not None
        assert _gap(emitted, source) < 1e-8
        assert _entanglers(emitted, entangler) <= 3


def test_the_global_phase_is_dropped_rather_than_recorded() -> None:
    """The reconstruction is equal up to a phase, not exactly equal.

    FlagQuantum IR has no field for a global phase, so a consumer comparing raw
    statevectors has to know the difference is real. The overlap magnitude is
    still one to machine precision.
    """

    source = torch.tensor(_random_su4(3), dtype=torch.complex128)
    emitted = synthesize_two_qubit(
        source.tolist(), wires=(0, 1), entangler="cx", z_rotation="rz"
    )
    assert emitted is not None
    overlap = _overlap(emitted, source)
    assert abs(abs(overlap) - 1.0) < 1e-9
    assert float(torch.max(torch.abs(source - _product(emitted)))) > 1e-3
    assert abs(cmath.phase(complex(overlap))) > 1e-3


def test_a_product_unitary_needs_no_entangler() -> None:
    """A purely local two-qubit unitary pays for nothing two-qubit."""

    source = torch.kron(
        torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
        torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128),
    )
    emitted = synthesize_two_qubit(
        source.tolist(), wires=(0, 1), entangler="cx", z_rotation="rz"
    )
    assert emitted is not None
    assert _entanglers(emitted, "cx") == 0
    assert _gap(emitted, source) < 1e-9


def test_reversing_the_wires_mirrors_the_entangler() -> None:
    """The entangler is the control-first one in whatever register order is given.

    `wires=(1, 0)` reads the matrix with wire 1 most significant, so the same
    physical operation is the bit-swap conjugate of the `wires=(0, 1)` matrix.
    """

    source = _source("crx")
    forward = synthesize_two_qubit(
        source.tolist(), wires=(0, 1), entangler="cx", z_rotation="rz"
    )
    mirrored_source = _SWAP_PERMUTATION @ source @ _SWAP_PERMUTATION
    mirrored = synthesize_two_qubit(
        mirrored_source.tolist(), wires=(1, 0), entangler="cx", z_rotation="rz"
    )
    assert forward is not None
    assert mirrored is not None
    assert _entanglers(forward, "cx") == _entanglers(mirrored, "cx")
    assert {item.wires for item in forward if item.name == "cx"} == {(0, 1)}
    assert {item.wires for item in mirrored if item.name == "cx"} == {(1, 0)}
    assert all(len(item.wires) == 2 or item.wires[0] in (0, 1) for item in forward)
    assert _gap(mirrored, mirrored_source, (1, 0)) < 1e-9


def test_an_unsupported_entangler_is_refused() -> None:
    source = _source("swap").tolist()
    for entangler in ("rzz", "ecr", "swap", ""):
        assert (
            synthesize_two_qubit(
                source, wires=(0, 1), entangler=entangler, z_rotation="rz"
            )
            is None
        )


def test_a_non_unitary_matrix_is_refused() -> None:
    for matrix in (
        [[1.0, 0.0, 0.0, 0.0]] * 4,
        [[2.0 if i == j else 0.0 for j in range(4)] for i in range(4)],
    ):
        with pytest.raises(ValueError):
            synthesize_two_qubit(matrix, wires=(0, 1), entangler="cx", z_rotation="rz")


def test_a_matrix_of_the_wrong_shape_is_refused() -> None:
    for matrix in (
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.0] * 4 for _ in range(3)],
        [[0.0] * 3 for _ in range(4)],
        [],
    ):
        with pytest.raises(ValueError):
            synthesize_two_qubit(matrix, wires=(0, 1), entangler="cx", z_rotation="rz")


def test_wires_must_be_two_distinct_non_negative_integers() -> None:
    source = _source("swap").tolist()
    for wires in ((0,), (0, 1, 2), (1, 1), (-1, 0)):
        with pytest.raises((TypeError, ValueError)):
            synthesize_two_qubit(source, wires=wires, entangler="cx", z_rotation="rz")


def test_a_basis_without_a_z_rotation_cannot_synthesize() -> None:
    source = _source("swap").tolist()
    assert (
        synthesize_two_qubit(source, wires=(0, 1), entangler="cx", z_rotation="h")
        is None
    )


def test_the_synthesis_is_deterministic() -> None:
    matrix = _random_su4(11)
    first = synthesize_two_qubit(matrix, wires=(0, 1), entangler="cz", z_rotation="rz")
    second = synthesize_two_qubit(matrix, wires=(0, 1), entangler="cz", z_rotation="rz")
    assert first is not None
    assert second is not None
    assert [(item.name, item.wires, item.params) for item in first] == [
        (item.name, item.wires, item.params) for item in second
    ]


def test_the_module_carries_no_second_source_of_truth_for_gate_matrices() -> None:
    """The synthesizer takes a matrix; it does not own a gate matrix table."""

    import flagquantum.compiler.two_qubit_synthesis as module

    assert module.__file__ is not None
    with open(module.__file__, encoding="utf-8") as handle:
        source = handle.read()
    for forbidden in ("GATE_MAT_DICT", "import torch", "import numpy", "simulation"):
        assert forbidden not in source, forbidden
    assert "one_qubit_synthesis" in source


def test_a_target_in_the_entangler_class_costs_one_entangler() -> None:
    """The two ends of the trace choice, so neither can silently drift."""

    cheap = synthesize_two_qubit(
        _source("cx").tolist(), wires=(0, 1), entangler="cx", z_rotation="rz"
    )
    dear = synthesize_two_qubit(
        _source("swap").tolist(), wires=(0, 1), entangler="cx", z_rotation="rz"
    )
    assert cheap is not None
    assert dear is not None
    assert _entanglers(cheap, "cx") == 1
    assert _entanglers(dear, "cx") == 3
