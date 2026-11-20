"""Complete-operator and rejection checks for native CX/SX/RZ pulse synthesis."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from flagquantum.compiler.optimization import two_qubit_synthesis as synthesis
from flagquantum.core.ir import Instruction
from tests.unit.test_compilation_two_qubit_synthesis import (
    _gap,
    _matrix,
    _product,
    _random_su4,
    _source,
)

pytestmark = pytest.mark.unit


def _synthesize(
    source: torch.Tensor, pair: tuple[int, int] = (0, 1)
) -> tuple[Instruction, ...]:
    emitted = synthesis.synthesize_two_qubit(
        source,
        qubits=pair,
        entangler="cx",
        z_rotation="rz",
        metadata={"source": "matrix"},
    )
    assert emitted is not None
    return emitted


@pytest.mark.parametrize("seed", range(16))
@pytest.mark.parametrize("pair", [(0, 1), (7, 2)])
def test_three_cx_uses_ten_pulses_and_preserves_the_complete_operator(
    seed: int, pair: tuple[int, int]
) -> None:
    source = torch.tensor(_random_su4(8100 + seed), dtype=torch.complex128)
    emitted = _synthesize(source, pair)
    assert sum(item.name == "cx" for item in emitted) == 3
    assert sum(item.name == "sx" for item in emitted) <= 10
    assert {item.wires for item in emitted if item.name == "cx"} == {pair}
    assert all(item.metadata == {"source": "matrix"} for item in emitted)
    assert _gap(emitted, source, pair) < 1e-12
    assert emitted == _synthesize(source, pair)


@pytest.mark.parametrize("opcode", ["cphase", "rxx", "dcx", "iswap"])
def test_two_cx_preserves_direction_and_reduces_local_gates(opcode: str) -> None:
    frame = [
        _matrix(
            Instruction(
                "u3",
                (0,),
                params={
                    "theta": 0.42 + index * 0.37,
                    "phi": -0.72 + index * 0.21,
                    "lbd": 0.53 - index * 0.47,
                },
            )
        )
        for index in range(4)
    ]
    source = (
        torch.kron(frame[0], frame[1])
        @ _source(opcode)
        @ torch.kron(frame[2], frame[3])
    )
    with patch.object(synthesis, "_verified_cx_pulses", return_value=False):
        baseline = _synthesize(source)
    candidate = _synthesize(source)
    assert len(candidate) < len(baseline)
    assert sum(item.name == "sx" for item in candidate) <= sum(
        item.name == "sx" for item in baseline
    )
    assert {item.wires for item in candidate if item.name == "cx"} == {(0, 1)}
    assert _gap(candidate, source) < 1e-12


@pytest.mark.parametrize("offset", [0.0, 1e-3, -1e-3, 1e-9, -1e-9, 1e-11, -1e-11])
@pytest.mark.parametrize("coordinate", [0.0, math.pi / 4])
def test_weyl_boundaries_keep_the_full_operator(
    coordinate: float, offset: float
) -> None:
    source = _source("rxx", {"theta": 2 * coordinate + offset})
    source = _source("ryy", {"theta": 0.34}) @ source
    source = _source("rzz", {"theta": offset}) @ source
    emitted = _synthesize(source)
    assert _gap(emitted, source) < 1e-12


def test_zero_first_entry_does_not_prevent_phase_verification() -> None:
    source = _source("swap")
    source = (
        torch.kron(
            torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
            torch.eye(2, dtype=torch.complex128),
        )
        @ source
    )
    assert source[0, 0] == 0
    emitted = _synthesize(source)
    assert synthesis._verified_cx_pulses(source.tolist(), emitted, (0, 1))
    assert _gap(emitted, source) < 1e-12


@pytest.mark.parametrize(
    "corruption", ["remove_rotation", "reverse_control", "zero_operator"]
)
def test_complete_operator_verifier_rejects_corrupted_candidates(
    corruption: str,
) -> None:
    source = torch.tensor(_random_su4(8200), dtype=torch.complex128)
    emitted = list(_synthesize(source))
    if corruption == "remove_rotation":
        del emitted[
            next(index for index, item in enumerate(emitted) if item.name == "rz")
        ]
    elif corruption == "reverse_control":
        index = next(index for index, item in enumerate(emitted) if item.name == "cx")
        emitted[index] = replace(emitted[index], wires=emitted[index].wires[::-1])
    else:
        emitted[0] = Instruction("opaque", (0, 1), matrix=[[0j] * 4 for _ in range(4)])
    assert not synthesis._verified_cx_pulses(source.tolist(), emitted, (0, 1))


def test_rejected_optimization_retains_the_general_synthesis() -> None:
    source = torch.tensor(_random_su4(8300), dtype=torch.complex128)
    candidate = _synthesize(source)
    with patch.object(synthesis, "_cx_pulse_leaves", return_value=candidate[1:]):
        rejected = _synthesize(source)
    assert {item.wires for item in rejected if item.name == "cx"} == {(0, 1)}
    assert len(rejected) > len(candidate)
    assert _gap(rejected, source) < 1e-12
    assert not torch.allclose(
        _product(candidate[1:]), _product(candidate), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("pair", [(0, 1), (7, 2)])
def test_verifier_handles_interleaved_local_gates_and_reversed_cx(
    pair: tuple[int, int],
) -> None:
    emitted = (
        Instruction("ry", (pair[0],), params={"theta": 0.91}),
        Instruction("sx", (pair[1],)),
        Instruction("cx", pair[::-1]),
        Instruction("rz", (pair[0],), params={"theta": -0.73}),
        Instruction("rx", (pair[1],), params={"theta": 0.32}),
        Instruction("cx", pair),
        Instruction("ry", (pair[1],), params={"theta": -0.21}),
    )
    source = _product(emitted, pair)
    assert synthesis._verified_cx_pulses(source.tolist(), emitted, pair)
    assert not synthesis._verified_cx_pulses(source.tolist(), emitted[1:], pair)


@pytest.mark.parametrize(
    "invalid",
    [
        Instruction("sx", (9,)),
        Instruction(
            "rz", (0,), params={"theta": torch.tensor(0.1, requires_grad=True)}
        ),
        Instruction("rz", (0,), params={"theta": [0.1, 0.2]}),
    ],
)
def test_verifier_rejects_unreadable_or_outside_local_gates(
    invalid: Instruction,
) -> None:
    assert not synthesis._verified_cx_pulses(
        torch.eye(4, dtype=torch.complex128).tolist(), (invalid,), (0, 1)
    )


def test_concurrent_synthesis_keeps_inputs_and_metadata_local() -> None:
    source = _random_su4(8400)
    targets = tuple(synthesis.SUPERCONTROLLED_ENTANGLERS)

    def compile_case(index: int) -> tuple[Instruction, ...]:
        emitted = synthesis.synthesize_two_qubit(
            source,
            qubits=(7, 2),
            entangler=targets[index % len(targets)],
            z_rotation="rz",
            metadata={"case": index},
        )
        assert emitted is not None
        return emitted

    expected = [compile_case(index) for index in range(18)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(compile_case, index) for index in reversed(range(18))
        ]
        actual = [future.result(timeout=30) for future in futures]
    for index, emitted in zip(reversed(range(18)), actual, strict=True):
        assert emitted == expected[index]
        assert all(item.metadata == {"case": index} for item in emitted)
        assert (
            _gap(emitted, torch.tensor(source, dtype=torch.complex128), (7, 2)) < 1e-12
        )
