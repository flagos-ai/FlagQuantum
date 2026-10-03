"""The stabilizer engine's positioned-noise entry point.

`sample_stabilizer` measures a requested wire list once at the end of a noiseless
circuit. This module checks the second entry point, which executes a program's
own measurements in order and accepts positioned channels whose Kraus operators
are a mixture of Pauli frames. Everything it asserts is about position or about
classification: which instruction an error was placed before, which column a
record lands in, which frames a channel's operators name, and which placements
and which channels the entry point refuses rather than approximating.

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
    coherent_overrotation_channel,
    depolarizing_channel,
    phase_damping_channel,
    phase_flip_channel,
    reset_error_channel,
    thermal_relaxation_channel,
    two_qubit_depolarizing_channel,
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
PAULI_Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
PAULI_Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
_PAULI = {"I": IDENTITY, "X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z}


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


def _channel(
    name: str, operators: tuple[torch.Tensor, ...], wire: int = 0
) -> Instruction:
    """A channel instruction holding the operators the caller built.

    The operators are passed through as they are rather than wrapped in a
    `KrausChannel`, because two of the cases below are about what the engine does
    with an operator list that is not a channel at all.
    """

    return Instruction(
        name=name,
        wires=(wire,),
        matrix=operators,
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
        program, shots=2, terminal_wires=(1, 2), seed=1
    )
    descending = sample_noisy_measurements(
        program, shots=2, terminal_wires=(2, 1), seed=1
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

    record = sample_noisy_measurements(program, shots=5, terminal_wires=(0,), seed=3)

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
        sample_stabilizer(program, shots=2, wires=[0], seed=1)


@pytest.mark.parametrize(
    "channel",
    [
        amplitude_damping_channel(0.1),
        phase_damping_channel(0.1),
        reset_error_channel(0.1),
        thermal_relaxation_channel(0.4, 0.8, 0.05),
    ],
)
def test_a_channel_with_no_pauli_frame_is_refused(channel: KrausChannel) -> None:
    """The accepted channels are a class, and these four are outside it.

    None of them is a mixture of unitaries at all - an amplitude damping channel
    maps a state off the unitary group, and a thermal relaxation channel is built
    from its square roots - so no Pauli frame represents one however it is
    placed, and the engine refuses it by name instead of sampling the nearest
    Pauli approximation.
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


def test_a_unitary_that_is_not_a_pauli_is_refused() -> None:
    """A mixture of unitaries is necessary for a frame and not sufficient.

    A coherent over-rotation is one unitary branch with probability one, so it is
    a unitary mixture by construction, and it is still not a frame: a rotation by
    an angle that is not a multiple of pi moves a stabilizer state off the
    stabilizer states. Accepting it as though it were a Pauli would put a
    non-Clifford error into a Clifford sample and nothing downstream could tell.
    """

    channel = coherent_overrotation_channel(0.3)
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

    with pytest.raises(CapabilityError, match="not a Pauli operator"):
        sample_noisy_measurements(program, shots=2, seed=1)


def test_a_channel_on_more_than_two_wires_is_refused() -> None:
    """The engine's frames are one- and two-wire frames, and it says so."""

    identity = torch.eye(8, dtype=torch.complex128)
    flip_first_wire = torch.kron(PAULI_X, torch.eye(4, dtype=torch.complex128))
    channel = KrausChannel(
        "three_wire_flip",
        (0.5**0.5 * identity, 0.5**0.5 * flip_first_wire),
    )
    program = _ir(
        3,
        Instruction(
            name=channel.name,
            wires=(0, 1, 2),
            matrix=channel.kraus,
            metadata={"is_channel": True},
        ),
        _measure(0),
    )

    with pytest.raises(CapabilityError, match="one or two wires"):
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
    assert "Pauli" in message
    assert "fails closed rather than approximating" in message


@pytest.mark.parametrize(
    ("channel", "expected"),
    [(bit_flip_channel(1.0), 1), (phase_flip_channel(1.0), 0)],
)
def test_the_operators_decide_which_frame_executes(
    channel: KrausChannel, expected: int
) -> None:
    """The name is not evidence, so the operators are what the engine reads.

    Both programs below name their instruction `bit_flip` and only one of them
    flips the readout, because a phase flip leaves `|0>` where it was. An engine
    that dispatched on the name would answer the same way twice, and nothing in
    the samples would say which channel it executed.
    """

    program = _ir(
        1,
        Instruction(
            name="bit_flip",
            wires=(0,),
            matrix=channel.kraus,
            metadata={"is_channel": True},
        ),
        _measure(0),
    )

    assert (
        sample_noisy_measurements(program, shots=8, seed=1).tolist() == [[expected]] * 8
    )


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_a_pauli_channel_at_the_edge_of_its_range_is_still_a_frame(
    probability: float,
) -> None:
    """A branch with weight zero is a branch, not a hole in the channel.

    Every bit flip at probability zero and every depolarizing channel at one
    carries a Kraus operator that is the zero matrix. Reading one as a general
    channel would refuse a channel that is a Pauli error with certainty, so the
    boundary is asserted here rather than left to the interior cases.
    """

    program = _ir(1, _gate("x", 0), _flip(0, probability), _measure(0))

    assert (
        sample_noisy_measurements(program, shots=8, seed=1).tolist()
        == [[1 - int(probability)]] * 8
    )


def test_a_kraus_list_that_is_not_a_channel_is_refused_as_input() -> None:
    """A malformed channel is the caller's to fix, and the error says so.

    Operators whose weights sum above one are not a channel at all, so there is
    nothing to classify and nothing to route around. Reporting that as a missing
    capability would send the caller looking for another engine instead of
    another channel.
    """

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

    with pytest.raises(ValidationError, match="not a well-formed channel"):
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


def _frame(label: str, *, probability: float = 1.0) -> Instruction:
    """A channel carrying exactly one non-identity frame.

    A channel whose whole weight sits on one frame is deterministic, so the
    record it produces is the frame's own action with no shot noise to argue
    with. That is what makes it a witness for the *position* the frame was
    translated to, which is the one thing a transposed argument list would move.
    """

    matrix = _PAULI[label[0]]
    for character in label[1:]:
        matrix = torch.kron(matrix, _PAULI[character])
    size = int(matrix.shape[0])
    return Instruction(
        name=f"frame_{label}",
        wires=tuple(range(len(label))),
        matrix=(
            (1.0 - probability) ** 0.5 * torch.eye(size, dtype=torch.complex128),
            probability**0.5 * matrix,
        ),
        metadata={"is_channel": True},
    )


def test_a_depolarizing_channel_is_executed_as_its_four_frames() -> None:
    """The widest one-wire mixture is a channel this engine used to refuse.

    A depolarizing channel is the identity with weight ``1 - p`` and each Pauli
    with weight ``p / 3``. Only the two bit-flipping frames reach a readout of
    ``|0>``, so the flip rate is ``2p / 3``, and a translation that kept the
    channel's four operators with three arguments - or that dropped the identity
    - would answer a different rate rather than fail to answer at all.
    """

    channel = depolarizing_channel(0.3)
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

    rate = float(sample_noisy_measurements(program, shots=4000, seed=5).float().mean())

    assert abs(rate - 0.3 * 2 / 3) < 0.02


def test_a_two_wire_pauli_channel_is_executed_as_a_frame() -> None:
    """Two wires is the other frame width the engine tracks.

    Two-qubit depolarizing noise is fifteen non-identity frames rather than
    three, so this is where a dropped branch shows up: of the fifteen, exactly
    eight anticommute with the parity that one CNOT and one readout measure,
    which puts the flip rate at eight fifteenths of the channel's weight.
    """

    channel = two_qubit_depolarizing_channel(0.5)
    program = _ir(
        2,
        Instruction(
            name=channel.name,
            wires=(0, 1),
            matrix=channel.kraus,
            metadata={"is_channel": True},
        ),
        _gate("cx", 0, 1),
        _measure(1),
    )

    rate = float(sample_noisy_measurements(program, shots=4000, seed=5).float().mean())

    assert abs(rate - 0.5 * 8 / 15) < 0.02


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("XI", [1, 0]),
        ("IX", [0, 1]),
        ("YI", [1, 0]),
        ("IY", [0, 1]),
        ("XX", [1, 1]),
        ("ZZ", [0, 0]),
    ],
)
def test_a_two_wire_frame_lands_on_the_argument_its_label_names(
    label: str, expected: list[int]
) -> None:
    """The frame label, the operator, and the engine argument are one claim.

    Every one of these channels carries a single frame at full weight, so the
    record is the frame's action on `|00>` and nothing else. A transposed
    argument list - the one thing that separates `XI` from `IX`, or `ZI` from
    `IZ` - would read out the other wire's pair here, and the two-wire
    depolarizing rate above could not see it because that channel weighs every
    frame the same.
    """

    program = _ir(2, _frame(label), _measure(0), _measure(1))

    assert (
        sample_noisy_measurements(program, shots=8, seed=1).tolist() == [expected] * 8
    )


def test_the_pauli_vocabulary_separates_the_two_flipping_frames() -> None:
    """A frame table that swapped two one-wire entries would answer a different frame.

    `X` and `Y` both flip a measurement of `|0>`, so no readout of that state can
    tell them apart. Reading the frame in place between two Hadamards can: `X`
    closes the interferometer and `Y` opens it, so one of the two records is
    deterministic and the other is not. Without this, a table whose entries were
    ordered `I, Y, X, Z` would put the `X` branch into the engine's `Y` argument
    and every one-wire readout would look correct anyway.
    """

    program = _ir(
        1,
        _gate("h", 0),
        _frame("X"),
        _gate("h", 0),
        _measure(0),
    )

    assert sample_noisy_measurements(program, shots=8, seed=1).tolist() == [[0]] * 8


def test_a_branch_that_is_a_frame_times_a_global_phase_is_executed() -> None:
    """A phase on a branch is unobservable, so the frame is read through it.

    `-i X` and `X` predict the same measurement outcomes, and a channel holding
    either is the same Pauli error. An engine that compared matrix entries rather
    than projecting the branch onto the Pauli basis would refuse this channel for
    a difference no sample can see.
    """

    phased = -1j * PAULI_X

    program = _ir(1, _channel("phased_flip", (phased,)), _measure(0))

    assert sample_noisy_measurements(program, shots=8, seed=1).tolist() == [[1]] * 8


def test_a_branch_that_is_a_frame_only_up_to_the_tolerance_is_executed() -> None:
    """The frame match carries the channel type's own tolerance, not an exact test.

    The operator below is `X` with one entry off by ``1e-9``: it is not the
    identity under an exact comparison and it is a bit flip to every digit a
    sampler can resolve. An exact test would refuse a channel that the channel
    type has already accepted as trace preserving, and the caller would have no
    operator left to fix.
    """

    nearly_flip = torch.tensor([[0, 1 + 1e-9], [1, 0]], dtype=torch.complex128)

    program = _ir(1, _channel("nearly_flip", (nearly_flip,)), _measure(0))

    assert sample_noisy_measurements(program, shots=8, seed=1).tolist() == [[1]] * 8


def test_two_branches_on_one_frame_are_added_rather_than_overwritten() -> None:
    """A frame is one argument however many branches reach it.

    This channel holds two distinguishable Kraus operators that are both `X`, so
    the frame's weight is their sum and not the last one read. Overwriting would
    report half the flip rate of a channel that flips with probability one half,
    which is a sampled answer rather than a refusal and would be believed.
    """

    channel = KrausChannel(
        "halved_flip",
        (0.5 * PAULI_X, 0.5 * PAULI_X, 0.5**0.5 * IDENTITY),
    )
    program = _ir(1, _channel(channel.name, channel.kraus), _measure(0))

    rate = float(sample_noisy_measurements(program, shots=4000, seed=5).float().mean())

    assert abs(rate - 0.5) < 0.02


def test_a_channel_whose_non_identity_weight_exceeds_one_is_refused() -> None:
    """The engine's channels take the identity as a remainder, so one is the ceiling.

    The operator below scales by ``1 + 5e-7``, which is inside the tolerance the
    channel type uses for trace preservation and inside the tolerance the
    mixture classifier uses, so it reaches the engine as a well-formed bit flip
    whose weight is above one. Passing it on would let the engine's own gate
    refuse it, or worse, silently renormalise a channel the caller built with a
    weight it did not intend.
    """

    overshoot = 1.0 + 5e-7

    program = _ir(
        1,
        _channel("overweight_flip", (overshoot**0.5 * PAULI_X,)),
        _measure(0),
    )

    with pytest.raises(CapabilityError, match="identity as the remainder"):
        sample_noisy_measurements(program, shots=2, seed=1)


def test_a_channel_with_no_operators_is_refused_as_a_capability() -> None:
    """An instruction that carries no channel at all is not a channel of weight zero.

    There is nothing to classify, so the refusal names the missing capability
    rather than reporting a malformed channel: the caller has no operator to fix
    and needs to know that this engine translates operators, not names.
    """

    program = _ir(1, _channel("no_operators", ()), _measure(0))

    with pytest.raises(CapabilityError, match="carries no Kraus operators"):
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
        sample_noisy_measurements(program, shots=2, terminal_wires=wires, seed=1)


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
