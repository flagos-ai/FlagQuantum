"""The stabilizer engine's positioned-noise entry point.

`sample_stabilizer` measures a requested wire list once at the end of a noiseless
circuit. This module checks the second entry point, which executes a program's
own measurements in order and accepts single-wire bit-flip channels between
them. Everything it asserts is about position: which instruction a flip was
placed before, which column a record lands in, and which placements the entry
point refuses rather than approximating.

The engine's own noiseless contract is in `test_stabilizer_engine.py`, and the
distributional comparison of `sample_stabilizer` against the dense amplitude path
is in `test_stabilizer_sampling.py`. Here the programs are deterministic wherever
a position is being pinned, so a wrong placement fails rather than being averaged
away by shot noise.
"""

from __future__ import annotations

import builtins
import sys

import pytest
import torch

from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode
from flagquantum.errors import CapabilityError, ValidationError
from flagquantum.noise import (
    KrausChannel,
    amplitude_damping_channel,
    bit_flip_channel,
    depolarizing_channel,
    phase_flip_channel,
)
from flagquantum.simulation.stabilizer import (
    StabilizerDependencyError,
    sample_noisy_measurements,
    sample_stabilizer,
)

# The engine is an optional distribution, so the file skips rather than failing
# to import when it is absent.
pytest.importorskip("stim")

pytestmark = pytest.mark.unit

IDENTITY = torch.eye(2, dtype=torch.complex128)
PAULI_X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)


def _ir(n_wires: int, *instructions: Instruction) -> CircuitIR:
    return CircuitIR(n_wires=n_wires, instructions=instructions)


def _gate(name: str, *wires: int) -> Instruction:
    return Instruction(name=name, wires=tuple(wires))


def _measure(wire: int) -> Instruction:
    """A measurement as the hybrid lowering emits one: a single dynamic wire."""

    return Instruction(name="measure", wires=(wire,), metadata={"is_dynamic": True})


def _reset(wire: int) -> Instruction:
    return Instruction(name="reset", wires=(wire,), metadata={"is_dynamic": True})


def _flip(wire: int, probability: float = 1.0) -> Instruction:
    """A native bit-flip channel instruction, built the way the compiler does."""

    return Instruction(
        name="bit_flip",
        wires=(wire,),
        matrix=bit_flip_channel(probability).kraus,
        metadata={"is_channel": True},
    )


def test_the_flip_lands_on_the_side_of_the_readout_it_was_placed_on() -> None:
    """Position is the whole contract, so it is asserted in both directions.

    A flip after the readout cannot reach it and a flip before it does, which is
    the difference between an error the sampler reports and an error it silently
    moved.
    """

    before = _ir(1, _gate("x", 0), _flip(0), _measure(0))
    after = _ir(1, _gate("x", 0), _measure(0), _flip(0))

    assert sample_noisy_measurements(before, shots=2, seed=1).tolist() == [[0]] * 2
    assert sample_noisy_measurements(after, shots=2, seed=1).tolist() == [[1]] * 2


def test_a_flip_between_two_gates_changes_what_the_second_gate_sees() -> None:
    """A flip is an operation in the circuit, not a decoration on a readout.

    A flip on the control before the CNOT propagates to the target and both wires
    read one; the same flip after the CNOT leaves the target at zero. Both wires
    are read out, so a placement the engine rounded to the nearest gate would
    show up as the wrong pair.
    """

    copied = _ir(2, _flip(0), _gate("cx", 0, 1), _measure(0), _measure(1))
    trailing = _ir(2, _gate("cx", 0, 1), _flip(0), _measure(0), _measure(1))

    assert sample_noisy_measurements(copied, shots=2, seed=1).tolist() == [[1, 1]] * 2
    assert sample_noisy_measurements(trailing, shots=2, seed=1).tolist() == [[1, 0]] * 2


def test_the_record_order_is_the_program_order_not_the_wire_order() -> None:
    """A caller reads the columns against its own instruction list.

    Recording in ascending wire order happens to agree with program order for a
    program that measures its wires in ascending order, so the wires are measured
    in descending order here and two different values are read out.
    """

    program = _ir(2, _gate("x", 1), _measure(1), _measure(0))

    assert sample_noisy_measurements(program, shots=2, seed=1).tolist() == [[1, 0]] * 2


def test_terminal_wires_follow_the_program_measurements_in_the_requested_order() -> (
    None
):
    """The terminal block is appended after the record, in the order given.

    A detector error model reads its terminal data parities off the end of the
    record, so the block's offset and its column order both have to be the
    caller's rather than a canonical wire order.
    """

    program = _ir(3, _gate("x", 2), _measure(0), _reset(0))

    ascending = sample_noisy_measurements(
        program, shots=2, terminal_qubits=(1, 2), seed=1
    )
    descending = sample_noisy_measurements(
        program, shots=2, terminal_qubits=(2, 1), seed=1
    )

    assert ascending.tolist() == [[0, 0, 1]] * 2
    assert descending.tolist() == [[0, 1, 0]] * 2


def test_the_same_wire_is_recorded_once_per_measurement() -> None:
    """Two readouts of one wire are two columns, not one column combined.

    A syndrome round measures and resets its ancilla, so the same wire is read
    once per round and the record has to keep the rounds apart.
    """

    program = _ir(
        1,
        _measure(0),
        _flip(0),
        _measure(0),
        _flip(0),
        _measure(0),
    )

    assert (
        sample_noisy_measurements(program, shots=2, seed=1).tolist() == [[0, 1, 0]] * 2
    )


def test_a_repeated_seed_reproduces_the_same_record() -> None:
    """Determinism holds for a seed on one engine version and one machine."""

    program = _ir(
        2,
        _flip(0, 0.5),
        _gate("h", 1),
        _gate("cx", 1, 0),
        _flip(0, 0.25),
        _measure(0),
        _measure(1),
    )

    first = sample_noisy_measurements(program, shots=64, seed=20260930)
    assert torch.equal(
        first, sample_noisy_measurements(program, shots=64, seed=20260930)
    )


def test_the_record_is_an_integer_tensor_with_one_column_per_readout() -> None:
    program = _ir(2, _gate("h", 0), _measure(0), _flip(1, 0.5), _measure(1))

    record = sample_noisy_measurements(program, shots=5, terminal_qubits=(0,), seed=3)

    assert record.dtype == torch.int64
    assert record.shape == (5, 3)


def test_the_planned_entry_point_still_refuses_a_channel() -> None:
    """The two entry points do not share a contract, and neither is a fallback.

    A caller that routed this program through the planned stabilizer mode would
    be told the circuit is noiseless and would get samples from a different
    circuit, so the refusal is asserted next to the acceptance rather than
    assumed from a docstring.
    """

    program = _ir(1, _gate("h", 0), _flip(0, 0.5), _measure(0))

    assert sample_noisy_measurements(program, shots=2, seed=1).shape == (2, 1)
    with pytest.raises(CapabilityError, match="noise channel"):
        sample_stabilizer(program, shots=2, qubits=[0], seed=1)


@pytest.mark.parametrize(
    "channel",
    [
        depolarizing_channel(0.1),
        phase_flip_channel(0.1),
        amplitude_damping_channel(0.1),
    ],
)
def test_a_channel_that_is_not_a_single_pauli_error_is_refused(
    channel: KrausChannel,
) -> None:
    """Only the bit flip is a Pauli error at one position.

    An amplitude-damping or depolarizing channel is not one Pauli error however
    it is placed, so the engine refuses it by name instead of sampling the
    nearest Pauli approximation.
    """

    program = _ir(
        1,
        Instruction(
            name=channel.name,
            wires=(0,),
            matrix=channel.kraus,
            metadata={"is_channel": True},
        ),
        _measure(0),
    )

    with pytest.raises(CapabilityError, match=channel.name):
        sample_noisy_measurements(program, shots=2, seed=1)


@pytest.mark.parametrize("opcode", ["t", "tdg", "rx", "ry"])
def test_a_non_clifford_instruction_is_refused_by_opcode(opcode: str) -> None:
    program = _ir(
        2,
        Instruction(name=opcode, wires=(0,), params={"theta": 0.3}),
        _measure(0),
    )

    with pytest.raises(CapabilityError, match=opcode):
        sample_noisy_measurements(program, shots=2, seed=1)


def test_the_refusal_names_what_the_entry_point_accepts() -> None:
    """A refusal that does not say what would work forces the caller to guess."""

    program = _ir(1, Instruction(name="t", wires=(0,)), _measure(0))

    with pytest.raises(CapabilityError) as info:
        sample_noisy_measurements(program, shots=2, seed=1)

    message = str(info.value)
    assert "measure" in message
    assert "reset" in message
    assert "bit_flip" in message
    assert "fails closed rather than approximating" in message


def test_a_channel_with_the_wrong_operator_count_is_refused() -> None:
    """Three operators is not a bit flip however the instruction is named.

    The name alone is not evidence: an instruction called `bit_flip` carrying a
    different channel would otherwise be sampled as the flip its name claims, and
    the caller could not tell the two apart from the samples.
    """

    program = _ir(
        1,
        Instruction(
            name="bit_flip",
            wires=(0,),
            matrix=(
                IDENTITY,
                PAULI_X,
                torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128),
            ),
            metadata={"is_channel": True},
        ),
        _measure(0),
    )

    with pytest.raises(CapabilityError, match="exactly two"):
        sample_noisy_measurements(program, shots=2, seed=1)


def test_a_channel_whose_operators_are_not_the_bit_flip_pair_is_refused() -> None:
    """The operators are what executes, so they are what is checked."""

    program = _ir(
        1,
        Instruction(
            name="bit_flip",
            wires=(0,),
            matrix=phase_flip_channel(0.5).kraus,
            metadata={"is_channel": True},
        ),
        _measure(0),
    )

    with pytest.raises(CapabilityError, match="bit-flip pair"):
        sample_noisy_measurements(program, shots=2, seed=1)


def test_a_flip_probability_above_one_is_refused() -> None:
    """A weight no probability can carry is refused, never clamped."""

    program = _ir(
        1,
        Instruction(
            name="bit_flip",
            wires=(0,),
            matrix=(IDENTITY, 2.0**0.5 * PAULI_X),
            metadata={"is_channel": True},
        ),
        _measure(0),
    )

    with pytest.raises(ValidationError, match="not a probability"):
        sample_noisy_measurements(program, shots=2, seed=1)


def test_a_measurement_naming_more_than_one_wire_is_refused() -> None:
    """One recorded bit per measurement is what makes a record readable.

    A two-wire measurement would record two bits into one column, and the caller
    reading the model's detectors could not tell which wire it read.
    """

    program = _ir(
        2,
        Instruction(name="measure", wires=(0, 1), metadata={"is_dynamic": True}),
    )

    with pytest.raises(CapabilityError, match="one bit per measurement"):
        sample_noisy_measurements(program, shots=2, seed=1)


def test_a_reset_naming_more_than_one_wire_is_refused() -> None:
    program = _ir(
        2,
        Instruction(name="reset", wires=(0, 1), metadata={"is_dynamic": True}),
    )

    with pytest.raises(CapabilityError, match="one wire"):
        sample_noisy_measurements(program, shots=2, seed=1)


def test_a_program_carrying_lowered_measurement_nodes_is_refused() -> None:
    """The nodes and the program's own measurements are two answers to one question."""

    program = CircuitIR(
        n_wires=1,
        instructions=(_gate("x", 0), _measure(0)),
        measurements=(MeasurementNode("sample", (0,), shots=4),),
    )

    with pytest.raises(CapabilityError, match="lowered measurement nodes"):
        sample_noisy_measurements(program, shots=4, seed=1)


def test_a_program_with_no_readout_at_all_is_refused() -> None:
    """Zero recorded columns is not a sampling request."""

    with pytest.raises(ValidationError, match="at least one recorded measurement"):
        sample_noisy_measurements(_ir(1, _gate("x", 0)), shots=2, seed=1)


@pytest.mark.parametrize("shots", [0, -1, True, 1.5, "4", None])
def test_the_shot_count_must_be_a_positive_integer(shots: object) -> None:
    program = _ir(1, _measure(0))

    with pytest.raises(ValidationError, match="shot count"):
        sample_noisy_measurements(program, shots=shots, seed=1)


@pytest.mark.parametrize("seed", [1.5, True, "7", -1, 2**64, 2**64 + 1])
def test_the_seed_must_be_an_integer_the_engine_accepts(seed: object) -> None:
    """A seed outside the engine's range is refused, never truncated or wrapped."""

    program = _ir(1, _measure(0))

    with pytest.raises(ValidationError, match="seed"):
        sample_noisy_measurements(program, shots=2, seed=seed)


@pytest.mark.parametrize("wires", [[2], [0, 3]])
def test_a_terminal_wire_outside_the_circuit_is_refused(wires: list[int]) -> None:
    program = _ir(2, _measure(0))

    with pytest.raises(ValidationError, match="outside circuit range"):
        sample_noisy_measurements(program, shots=2, terminal_qubits=wires, seed=1)


def test_a_missing_engine_names_the_extra_that_provides_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The positioned entry point fails closed on the same seam.

    A caller reaching this entry point without the extra has to be told the one
    command that fixes it, not handed an ImportError from inside a helper.
    """

    real_import = builtins.__import__

    def refuse(name: str, *args: object, **kwargs: object):
        if name == "stim" or name.startswith("stim."):
            raise ImportError("No module named 'stim'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "stim", raising=False)
    monkeypatch.setattr(builtins, "__import__", refuse)

    program = _ir(1, _flip(0, 0.1), _measure(0))

    with pytest.raises(StabilizerDependencyError, match=r"flagquantum\[stim\]") as info:
        sample_noisy_measurements(program, shots=2, seed=1)

    assert isinstance(info.value, CapabilityError)
    assert isinstance(info.value, ImportError)
