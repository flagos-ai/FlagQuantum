"""Independent complex-phase and density oracles for native Ising actions."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Literal
from unittest.mock import patch

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir import IRValidationError
from flagquantum.core.ir.program.lowering import circuit_to_program
from flagquantum.core.ir.program.physical import IsingContext, IsingExchange
from flagquantum.core.ising import IsingBraidProgram
from flagquantum.core.rydberg import RydbergProgram
from flagquantum.errors import CapabilityError
from flagquantum.operators import SuperOperator
from flagquantum.simulation.ising_braid import (
    _exchange_unitary,
    _fusion_matrices,
    evolve_ising_braid,
)

pytestmark = pytest.mark.unit
Channel = Literal["1", "psi"]
Word = tuple[tuple[int, bool], ...]


def _reference_generator(left: int, clockwise: bool) -> torch.Tensor:
    # Closed-form matrix entries, independent of the production F^dag R F path.
    if left == 1:
        matrix = torch.tensor([[1, -1j], [-1j, 1]], dtype=torch.complex128) * (
            complex(0.9238795325112867, 0.3826834323650898) / math.sqrt(2)
        )
    else:
        matrix = torch.diag(
            torch.tensor(
                [
                    complex(0.9238795325112867, -0.3826834323650898),
                    complex(0.3826834323650898, 0.9238795325112867),
                ],
                dtype=torch.complex128,
            )
        )
    return matrix.mH if clockwise else matrix


def _program(initial: Channel, word: Word) -> IsingBraidProgram:
    program = IsingBraidProgram(initial_channel=initial)
    for left, clockwise in word:
        program.exchange(left, clockwise=clockwise)
    return program


def _reference_density(initial: Channel, word: Word) -> torch.Tensor:
    vector = torch.tensor([initial == "1", initial == "psi"], dtype=torch.complex128)
    for left, clockwise in word:
        vector = _reference_generator(left, clockwise) @ vector
    return torch.outer(vector, vector.conj())


def test_fusion_and_exchange_matrices_keep_the_reviewed_complex_gauge() -> None:
    fusion, exchange = _fusion_matrices()
    expected_fusion = torch.tensor(
        [[1, 1], [1, -1]], dtype=torch.complex128
    ) / math.sqrt(2)
    torch.testing.assert_close(fusion, expected_fusion, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        exchange, _reference_generator(0, False), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        fusion @ fusion, torch.eye(2, dtype=torch.complex128), atol=1e-12, rtol=0
    )
    fusion.zero_()
    exchange.zero_()
    again_fusion, again_exchange = _fusion_matrices()
    assert again_fusion.count_nonzero() == 4
    assert again_exchange.count_nonzero() == 2


@pytest.mark.parametrize("left", [0, 1, 2])
@pytest.mark.parametrize("clockwise", [False, True])
def test_generators_preserve_full_complex_phase(left: int, clockwise: bool) -> None:
    actual = _exchange_unitary(IsingExchange(left, clockwise))
    assert actual.device.type == "cpu" and actual.dtype == torch.complex128
    torch.testing.assert_close(
        actual, _reference_generator(left, clockwise), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        actual.mH @ actual,
        torch.eye(2, dtype=torch.complex128),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("left", [0, 1])
@pytest.mark.parametrize("clockwise", [False, True])
def test_both_adjacent_braid_relations_hold_as_complex_matrices(
    left: int, clockwise: bool
) -> None:
    a = _exchange_unitary(IsingExchange(left, clockwise))
    b = _exchange_unitary(IsingExchange(left + 1, clockwise))
    torch.testing.assert_close(a @ b @ a, b @ a @ b, atol=1e-12, rtol=0)


@pytest.mark.parametrize("first_direction", [False, True])
@pytest.mark.parametrize("second_direction", [False, True])
def test_distant_generators_commute(
    first_direction: bool, second_direction: bool
) -> None:
    a = _exchange_unitary(IsingExchange(0, first_direction))
    b = _exchange_unitary(IsingExchange(2, second_direction))
    torch.testing.assert_close(a @ b, b @ a, atol=1e-12, rtol=0)


@pytest.mark.parametrize("initial", ["1", "psi"])
@pytest.mark.parametrize(
    "word",
    [
        (),
        ((1, False),),
        ((1, True),),
        ((0, False), (1, False)),
        ((1, False), (0, False)),
        ((2, True), (1, False), (0, True), (1, True)),
        ((0, False), (1, False), (0, False)),
        ((1, False), (0, False), (1, False)),
        ((1, False), (2, False), (1, False)),
        ((2, False), (1, False), (2, False)),
        ((1, False), (0, True), (2, False), (1, True)) * 25,
    ],
)
def test_full_complex_density_matches_an_independent_vector_oracle(
    initial: Channel, word: Word
) -> None:
    actual = evolve_ising_braid(_program(initial, word).to_ir())
    assert actual.shape == (2, 2)
    assert actual.device.type == "cpu" and actual.dtype == torch.complex128
    torch.testing.assert_close(
        actual, _reference_density(initial, word), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(actual, actual.mH, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        torch.trace(actual),
        torch.tensor(1, dtype=torch.complex128),
        atol=1e-12,
        rtol=0,
    )
    assert float(torch.linalg.eigvalsh(actual).min()) >= -1e-12


@pytest.mark.parametrize("initial", ["1", "psi"])
@pytest.mark.parametrize("left", [0, 1, 2])
def test_exchange_and_inverse_restore_each_initial_charge(
    initial: Channel, left: int
) -> None:
    program = _program(initial, ((left, False), (left, True)))
    torch.testing.assert_close(
        evolve_ising_braid(program.to_ir()),
        program.projector(initial),
        atol=1e-12,
        rtol=0,
    )


def test_noncommuting_word_order_changes_coherence_not_just_populations() -> None:
    first = evolve_ising_braid(_program("1", ((0, False), (1, False))).to_ir())
    second = evolve_ising_braid(_program("1", ((1, False), (0, False))).to_ir())
    torch.testing.assert_close(first.diagonal(), second.diagonal(), atol=1e-12, rtol=0)
    assert float(torch.max(torch.abs(first - second))) > 0.5
    assert float(first.imag.abs().max()) > 0.4


def test_density_action_is_phase_invariant_while_source_words_stay_distinct() -> None:
    reference = IsingBraidProgram().exchange(1)
    central = IsingBraidProgram().exchange(1)
    for _ in range(4):
        central.exchange(0)
    generator = _exchange_unitary(IsingExchange(0, False))
    unitary = torch.linalg.matrix_power(generator, 4)
    identity = torch.eye(2, dtype=torch.complex128)
    torch.testing.assert_close(unitary, -1j * identity, atol=1e-12, rtol=0)
    assert not torch.allclose(unitary, identity, atol=1e-12, rtol=0)
    assert central.to_ir() != reference.to_ir()
    torch.testing.assert_close(
        evolve_ising_braid(central.to_ir()),
        evolve_ising_braid(reference.to_ir()),
        atol=1e-12,
        rtol=0,
    )


def test_cpu_profile_does_not_inherit_a_foreign_default_device() -> None:
    source = IsingBraidProgram().exchange(1).to_ir()
    with torch.device("meta"):
        actual = evolve_ising_braid(source)
    assert actual.device.type == "cpu"
    torch.testing.assert_close(
        actual, _reference_density("1", ((1, False),)), atol=1e-12, rtol=0
    )


def test_finite_action_reuses_superoperator_without_exponentials_or_dense_maps() -> (
    None
):
    record = _program("1", ((1, False), (0, True), (2, False))).to_ir()
    with (
        patch.object(
            SuperOperator,
            "left_right_multiply",
            wraps=SuperOperator.left_right_multiply,
        ) as action,
        patch.object(SuperOperator, "dense", side_effect=AssertionError("dense map")),
        patch("torch.matrix_exp", side_effect=AssertionError("invented Hamiltonian")),
    ):
        result = evolve_ising_braid(record)
    assert action.call_count == 3
    torch.testing.assert_close(
        result,
        _reference_density("1", ((1, False), (0, True), (2, False))),
        atol=1e-12,
        rtol=0,
    )


def test_results_are_independent_and_execution_does_not_consume_randomness() -> None:
    source = IsingBraidProgram().exchange(1).to_ir()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(853)
        before = torch.random.get_rng_state().clone()
        first = evolve_ising_braid(source)
        assert torch.equal(before, torch.random.get_rng_state())
        first.zero_()
        second = evolve_ising_braid(source)
    torch.testing.assert_close(
        second, _reference_density("1", ((1, False),)), atol=1e-12, rtol=0
    )


def test_invalid_source_uses_core_diagnostics_before_numerical_work() -> None:
    source = IsingBraidProgram().to_ir()
    context = source.physical_context
    assert isinstance(context, IsingContext)
    invalid_context: Callable[..., IsingContext] = replace
    broken = replace(
        source, physical_context=invalid_context(context, total_charge="psi")
    )
    with (
        patch.object(SuperOperator, "left_right_multiply") as numerical,
        pytest.raises(IRValidationError, match="physical.invalid_context"),
    ):
        evolve_ising_braid(broken)
    assert not numerical.called


def test_other_profiles_cannot_enter_the_finite_ising_action() -> None:
    rydberg = RydbergProgram(
        atoms=(("a", "site"),),
        sites={"site": (0, 0)},
        length_unit="um",
        time_unit="us",
        c6=1,
        loss_rate=0,
        minimum_separation=1,
        transport_channels=("move",),
        drive_channels=("drive",),
    )
    circuit = circuit_to_program(fq.Circuit(1).to_ir())
    for source in (rydberg.to_ir(), circuit.record):
        assert source is not None
        with pytest.raises(CapabilityError, match="native Ising profile"):
            evolve_ising_braid(source)


def test_root_entry_uses_the_admitted_runtime_path() -> None:
    program = IsingBraidProgram().exchange(1)
    plan = fq.plan(program)
    assert plan.analysis is None
    result = fq.run(plan)
    assert result.state is not None
    torch.testing.assert_close(
        result.state[0], evolve_ising_braid(program.to_ir()), atol=1e-12, rtol=0
    )
