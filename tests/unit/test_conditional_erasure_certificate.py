"""Independent Kraus and fault checks for the conditional channel certificate."""

from fractions import Fraction
from importlib import import_module
from itertools import product
from typing import Any

import pytest
import torch

from flagquantum.compiler.formats.kmx_circuit import read_kmx_circuit
from flagquantum.compiler.registers.primitives.global_phase import ccz_ir
from flagquantum.compiler.registers.primitives.measurement import (
    measure_x_clear_ir,
    reset_x_register_ir,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import Instruction
from tests.unit.circuit_asset_fixtures import instructions
from tests.unit.conditional_erasure_certificate import certify_conditional_erasure_map
from tests.unit.measurement_checks import expand_reset_branches, forced_branch

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
def test_locked_r_hmr_basis_semantics_and_independent_full_kraus_maps() -> None:
    backend: Any = import_module("kickmix")
    assert backend.__version__ == "0.1" and hasattr(backend, "Simulator")
    assert str(backend.__file__).endswith(".so")
    mask = (1 << 64) - 1
    initial = sum((lane & 1) << lane for lane in range(64))
    predicate = sum(((lane >> 1) & 1) << lane for lane in range(64))
    for conditional in (False, True):
        active_mask = predicate if conditional else mask
        observed: dict[int, set[int]] = {0: set(), 1: set()}
        for seed in (0, 41):
            results = {}
            for name in ("R", "HMR"):
                simulator = backend.Simulator(batch_size=64, seed=seed)
                simulator.write_across_shots(backend.q(0), initial)
                simulator.write_across_shots(backend.b(0), predicate)
                simulator.write_across_shots(backend.b(1), 0)
                text = name + " q0" + (" b1" if name == "HMR" else "")
                simulator.do(backend.Circuit(text + (" if b0" if conditional else "")))
                phases = tuple(
                    Fraction(simulator.read_shot_phase(lane)) % 2 for lane in range(64)
                )
                results[name] = phases
                assert simulator.read_across_shots(backend.q(0), out=int) == initial & (
                    mask ^ active_mask
                )
                if name == "HMR":
                    outcomes = simulator.read_across_shots(backend.b(1), out=int)
                    for lane, phase in enumerate(phases):
                        bit = (initial >> lane) & 1
                        outcome = (outcomes >> lane) & 1
                        assert phase == bit * outcome
                        if (active_mask >> lane) & 1:
                            observed[bit].add(outcome)
                        else:
                            assert outcome == 0
            # Identical source seeds expose the same first RNG draw. R's draw
            # is independently observed through HMR, not inferred from R phase.
            assert results["R"] == results["HMR"]
        assert observed == {0: {0, 1}, 1: {0, 1}}
    for name, active in product(("R", "HMR"), (False, True)):
        text = name + " q0" + (" b1" if name == "HMR" else "") + " if b0"
        circuit = read_kmx_circuit(text)
        gates = expand_reset_branches(
            instructions(circuit, int(active)), first_slot=circuit.bits
        )
        for vector in (
            torch.tensor([1, 0], dtype=torch.complex128),
            torch.tensor([0, 1], dtype=torch.complex128),
            torch.tensor([0.3 + 0.4j, 0.5 - 0.2j], dtype=torch.complex128),
        ):
            for outcome in (0, 1):
                if active:
                    expected = torch.zeros_like(vector)
                    expected[0] = (vector[0] + (-1) ** outcome * vector[1]) / 2**0.5
                else:
                    expected = vector if outcome == 0 else torch.zeros_like(vector)
                torch.testing.assert_close(
                    forced_branch(vector, gates, (outcome,)),
                    expected,
                    atol=2e-14,
                    rtol=0,
                )


def conditional_cleanup() -> tuple[Instruction, ...]:
    """An earlier reset outcome selects whether a later copy is X-erased."""
    return (
        ClassicalXor((), 1).instruction(),
        *expand_reset_branches(reset_x_register_ir((0,)), first_slot=0),
        Instruction("cx", (1, 0)),
        *measure_x_clear_ir(0, 1, conditions=((0, 1),)),
        Instruction("z", (1,), metadata={"conditions": ((1, 1),)}),
        Instruction("cx", (1, 0), metadata={"conditions": ((0, 0),)}),
    )


def test_each_local_kraus_column_and_conditional_normalization_with_spectator() -> None:
    gates = conditional_cleanup()
    mapping = {word: word for word in product((0, 1), repeat=2)}
    proof = certify_conditional_erasure_map(
        gates, n_wires=3, logical_wires=(1, 2), mapping=mapping
    )
    assert proof["effects"] == 2 and proof["conditional_effects"] == 1
    assert proof["allowed_histories"] == 3
    # Independent primitive projection includes every column and a fully complex
    # state entangling logical data with the untouched spectator.
    columns = torch.eye(4, dtype=torch.complex128)
    complex_state = torch.tensor([1 + 2j, 3 - 1j, -2 + 4j, 5j], dtype=torch.complex128)
    complex_state /= complex_state.norm()
    probabilities = []
    for history in product((0, 1), repeat=2):
        factor = 0.0 if history == (0, 1) else 2 ** (-0.5 if history[0] == 0 else -1)
        probabilities.append(factor**2)
        for vector in (*columns, complex_state):
            state = torch.zeros(8, dtype=torch.complex128)
            state[:4] = vector
            actual = forced_branch(state, gates, history)
            torch.testing.assert_close(actual, factor * state, atol=2e-14, rtol=0)
            assert torch.count_nonzero(actual[4:]) == 0
    assert sum(probabilities) == pytest.approx(1.0)


@pytest.mark.parametrize("active", (False, True))
def test_conditional_ccz_macro_matches_full_complex_operator(active: bool) -> None:
    setup = ClassicalXor((), 0, constant=active).instruction()
    block = ccz_ir((0, 1, 2), conditions=((0, 1),))
    labels = torch.arange(16)
    phase = torch.where((labels >> 1 == 7) & active, -1.0, 1.0)
    for column in range(16):
        state = torch.zeros(16, dtype=torch.complex128)
        state[column] = torch.tensor(0.6 + 0.8j, dtype=state.dtype)
        torch.testing.assert_close(
            forced_branch(state, (setup, *block), ()), phase * state, atol=2e-14, rtol=0
        )
    mapping = {word: word for word in product((0, 1), repeat=3)}
    if active:
        with pytest.raises(AssertionError, match="complex branch phase"):
            certify_conditional_erasure_map(
                (setup, *block), n_wires=3, logical_wires=(0, 1, 2), mapping=mapping
            )
    else:
        certify_conditional_erasure_map(
            (setup, *block), n_wires=3, logical_wires=(0, 1, 2), mapping=mapping
        )
    certify_conditional_erasure_map(
        (setup, *block, *block), n_wires=3, logical_wires=(0, 1, 2), mapping=mapping
    )


def test_missing_feedback_and_stale_inactive_results_cannot_be_certified() -> None:
    gates = conditional_cleanup()
    mapping: dict[tuple[int, ...], tuple[int, ...]] = {(0,): (0,), (1,): (1,)}
    with pytest.raises(AssertionError, match="complex branch phase"):
        certify_conditional_erasure_map(
            tuple(gate for gate in gates if gate.name != "z"),
            n_wires=2,
            logical_wires=(1,),
            mapping=mapping,
        )
    with pytest.raises(AssertionError, match="Inactive result slot is not zero"):
        certify_conditional_erasure_map(
            (ClassicalXor((), 1, constant=True).instruction(), *gates[1:]),
            n_wires=2,
            logical_wires=(1,),
            mapping=mapping,
        )
