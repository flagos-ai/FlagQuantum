"""Reading a detector error model back into a circuit, and running that circuit.

`flagquantum/qec/dem.py` writes a model out as stim text and reads stim text back
in; `flagquantum/qec/dem_construction.py` derives a model from a circuit and a
noise record. Neither is a route *from* a model *to* something executable, and
that is the gap this file closes. `circuit_from_detector_error_model` returns a
circuit whose channels are the model's own mechanisms, and
`detector_error_model_from_circuit` reads such a circuit back into the model it
stands for.

The evidence is in three parts, strongest first.

The round trip is exact and is asserted on every model below, including the
models of a repetition code, a Steane code and a rotated surface patch at one
round and at three. It is exact rather than approximate because the two halves are
read from different parts of the instruction: a frame comes from the branch
operators, and a probability comes from the instruction's declared masses. That
division is not a convenience -- a branch is written `sqrt(p) * U`, so reading `p`
back means squaring a square root, and `math.sqrt(p) ** 2 != p` for roughly half of
all doubles. `test_the_declared_mass_is_the_probability_the_model_states` measures
that failure rather than describing it, so the design decision is pinned by the
number that forced it.

The realization runs. It is a program of Pauli-frame channels, which is what
`flagquantum.simulation.stabilizer` samples, so the model reaches a second route to
the same rates that shares no arithmetic with `DetectorErrorModel.dem_sampling`:
the model's own route draws one uniform per mechanism and XORs the signatures that
fired, and the realization's route goes through a stabilizer engine. The two are
held against each other here, and the sampled rates are also held against the
rates the model *states* rather than against a second sample. At probability one
the answer is deterministic, and that check is exact rather than statistical, so a
frame placed at the wrong wire fails as a wrong bit instead of averaging into a
rate.

The chain to the external reference is transitive rather than repeated here.
`tests/qec/test_dem_stim_interop.py` pins the text this module's models are written
to against stim's own reader, and `tests/qec/test_dem_stim_reference_rates.py` pins
the model's rates to a stim reference circuit. This file pins the realization to
the model, which is the edge that file does not cover.

The last part is the refusals. A reader that states what a realization is has to
refuse what is not one, and each refusal below is a way a program could look like a
realization without being one: a frame after a measurement, a wire no branch flips,
masses that do not sum to one, masses that disagree with the operators they
accompany, an operator that is not a product of bit flips, a bare matrix where a
sequence of branches belongs, and a mechanism at probability zero, which has no
branch to be read back from at all.

The file samples through the stabilizer engine, which is an optional distribution,
so it skips rather than failing to import when `stim` is absent. That skip is
covered: the `coverage` lane installs `stim` and selects `integration`, so the
tests here execute in CI instead of skipping in every lane.
"""

from __future__ import annotations

import math

import pytest
import torch

from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode
from flagquantum.qec import (
    DemError,
    DetectorErrorModel,
    PhenomenologicalNoise,
    RepetitionCode,
    RotatedSurfaceCode,
    SteaneCode,
    build_memory_circuit,
    circuit_from_detector_error_model,
    detector_error_model_from_circuit,
)
from flagquantum.qec.dem_alternatives import fault_groups
from flagquantum.simulation.stabilizer import sample_noisy_measurements

pytest.importorskip("stim")

pytestmark = pytest.mark.integration

# The shot count and seed the rate comparisons use. One seed for the whole file,
# so a failure names the comparison that departed rather than the draw.
_SHOTS = 100_000
_SEED = 20260214
# Five standard errors, against the tightest band the measured departures need:
# the widest over every comparison below is 2.0 sigma, and a band of five leaves
# room for a seed change without leaving room for a wrong rate.
_SIGMA = 5.0
# A physical strength, and the one the migration sweep quotes.
_PROBABILITY = 0.02

_IDENTITY = torch.eye(2, dtype=torch.complex128)
_BIT_FLIP = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
_PHASE_FLIP = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.complex128))

# An id-free model: three mechanisms, one of them observable-only, two sharing a
# detector.
_INDEPENDENT = DetectorErrorModel(
    3,
    1,
    (
        DemError(0.01, (0, 1), (0,)),
        DemError(0.02, (1, 2), ()),
        DemError(0.2, (2,), (0,)),
    ),
)
# Two mechanisms with the same signature and no id, which the model keeps as two
# records and which the realization must keep as two channels.
_IDENTICAL_SIGNATURES = DetectorErrorModel(
    2, 0, (DemError(0.1, (0, 1), ()), DemError(0.2, (0, 1), ()))
)
# Detector 0 is never touched, and observable 1 is never touched: the shape a model
# states is wider than the faults it holds.
_OBSERVABLE_ONLY = DetectorErrorModel(1, 3, (DemError(0.25, (), (0, 2)),))
# A group of three alternatives plus one independent mechanism. The group's masses
# sum to 0.85, so the left-over mass is 0.15 and the group fires nothing on the
# remaining shots.
_ALTERNATIVES = DetectorErrorModel(
    3,
    1,
    (
        DemError(0.4, (0,), (), error_id=5),
        DemError(0.35, (0, 1), (0,), error_id=5),
        DemError(0.1, (2,), (0,), error_id=5),
        DemError(0.02, (1,), ()),
    ),
)
# A group whose masses sum to exactly one, so it fires on every shot and has no
# identity branch: detector 0 is flipped by every member and detector 1 by one.
_ALTERNATIVES_THAT_ALWAYS_FIRE = DetectorErrorModel(
    2,
    1,
    (
        DemError(0.25, (0,), (0,), error_id=9),
        DemError(0.75, (0, 1), (), error_id=9),
    ),
)
# Every mechanism at certainty, so every shot is the XOR of every signature.
_ALWAYS = DetectorErrorModel(
    3, 1, (DemError(1.0, (0, 2), (0,)), DemError(1.0, (2,), ()))
)
_EMPTY = DetectorErrorModel(4, 2, ())

_MODELS = (
    ("independent", _INDEPENDENT),
    ("identical-signatures", _IDENTICAL_SIGNATURES),
    ("observable-only", _OBSERVABLE_ONLY),
    ("alternatives", _ALTERNATIVES),
    ("alternatives-that-always-fire", _ALTERNATIVES_THAT_ALWAYS_FIRE),
    ("always", _ALWAYS),
    ("empty", _EMPTY),
)

# Two codes with different check structure at two round counts, so the round trip
# is measured across the round block rather than at one depth.
_CODES = (
    ("repetition", RepetitionCode(distance=3)),
    ("steane", SteaneCode()),
    ("surface", RotatedSurfaceCode(distance=3)),
)
_ROUNDS = (1, 3)


@pytest.fixture(
    params=[model for _, model in _MODELS], ids=[name for name, _ in _MODELS]
)
def model(request: pytest.FixtureRequest) -> DetectorErrorModel:
    """One of the hand-written models above."""

    return request.param


def _operator(layout: tuple[int, ...], flipped: tuple[int, ...]) -> torch.Tensor:
    """The product of bit flips on ``flipped`` over the wires ``layout``."""

    operator = torch.ones((1, 1), dtype=torch.complex128)
    for wire in layout:
        operator = torch.kron(operator, _BIT_FLIP if wire in flipped else _IDENTITY)
    return operator


def _measure(wire: int) -> Instruction:
    """A measurement of one wire, the way a realization records its register."""

    return Instruction(name="measure", wires=(wire,), metadata={"is_dynamic": True})


def _frame(
    wires: tuple[int, ...],
    branches: tuple[tuple[torch.Tensor, float], ...],
    *,
    error_id: int | None = None,
) -> Instruction:
    """One fault frame: a wire list, one operator per branch, one mass per branch."""

    metadata: dict[str, object] = {"is_channel": True}
    if error_id is not None:
        metadata["error_id"] = error_id
    return Instruction(
        name="dem_frame",
        wires=wires,
        params={"probabilities": tuple(mass for _, mass in branches)},
        matrix=tuple(operator for operator, _ in branches),
        metadata=metadata,
    )


def _realization() -> CircuitIR:
    """A realization of a two-detector, one-observable model with one fault.

    The fault flips detector 0 and observable 0 and leaves detector 1 alone, so the
    program is valid while still stating a wire every mechanism does not touch.
    """

    return CircuitIR(
        n_wires=3,
        instructions=(
            _frame((0, 1), ((_operator((0, 1), (0,)), 1.0),)),
            _measure(0),
            _measure(1),
            _measure(2),
        ),
    )


def _sampled_rates(
    sample: torch.Tensor, model: DetectorErrorModel
) -> tuple[torch.Tensor, torch.Tensor]:
    """The sampled detector rates and observable rates of a realization's shots."""

    split = model.num_detectors
    return sample[:, :split].double().mean(0), sample[:, split:].double().mean(0)


def _worst_departure(got: torch.Tensor, stated: torch.Tensor) -> float:
    """The largest departure of a sampled rate from a stated rate, in sigmas."""

    if not got.numel() or not stated.numel():
        return 0.0
    stated = stated.double()
    error = torch.sqrt(torch.clamp(stated * (1.0 - stated) / _SHOTS, min=1e-18))
    return float(((got - stated).abs() / error).max())


def _register_support(error: DemError, split: int) -> tuple[int, ...]:
    """The register positions one mechanism flips, whatever the split calls them."""

    return tuple(
        sorted(set(error.detectors) | {split + index for index in error.observables})
    )


def test_a_model_round_trips_through_its_realization(model: DetectorErrorModel) -> None:
    """The reading restores the model, shape and mechanisms alike."""

    circuit = circuit_from_detector_error_model(model)
    assert circuit.n_wires == model.num_detectors + model.num_observables

    restored = detector_error_model_from_circuit(
        circuit, num_detectors=model.num_detectors
    )
    assert restored == model
    assert (restored.num_detectors, restored.num_observables) == (
        model.num_detectors,
        model.num_observables,
    )


def test_the_realization_states_one_frame_per_fault_and_one_measurement_per_wire(
    model: DetectorErrorModel,
) -> None:
    """The circuit's structure is the model's structure, not an encoding of it."""

    circuit = circuit_from_detector_error_model(model)
    frames = tuple(item for item in circuit.instructions if item.name == "dem_frame")
    measures = tuple(item for item in circuit.instructions if item.name == "measure")

    assert len(frames) == len(fault_groups(model.errors))
    assert all(item.metadata["is_channel"] is True for item in frames)
    assert tuple(item.wires for item in measures) == tuple(
        (wire,) for wire in range(circuit.n_wires)
    )
    # Every frame precedes every measurement, which is what makes the register the
    # frames act on the register the measurements read.
    assert max(
        (
            index
            for index, item in enumerate(circuit.instructions)
            if item.name == "dem_frame"
        ),
        default=-1,
    ) < min(
        (
            index
            for index, item in enumerate(circuit.instructions)
            if item.name == "measure"
        ),
        default=len(circuit.instructions),
    )


def test_the_realization_of_a_memory_code_model_round_trips() -> None:
    """The round trip holds for the models this domain's construction route makes."""

    for name, code in _CODES:
        for rounds in _ROUNDS:
            model = DetectorErrorModel.from_memory_circuit(
                build_memory_circuit(code, rounds=rounds),
                noise=PhenomenologicalNoise(_PROBABILITY),
            )
            circuit = circuit_from_detector_error_model(model)
            restored = detector_error_model_from_circuit(
                circuit, num_detectors=model.num_detectors
            )
            assert restored == model, (name, rounds)
            # The realization states the model's whole shape rather than only the
            # part its mechanisms reach.
            assert circuit.n_wires == model.num_detectors + model.num_observables


def test_a_group_of_alternatives_becomes_one_channel_whose_masses_sum_to_one() -> None:
    """A group is one fault, so it is one channel and not several."""

    circuit = circuit_from_detector_error_model(_ALTERNATIVES)
    frames = tuple(item for item in circuit.instructions if item.name == "dem_frame")
    grouped = next(item for item in frames if item.metadata.get("error_id") == 5)

    assert grouped.wires == (0, 1, 2, 3)
    assert len(grouped.matrix) == 4
    masses = grouped.params["probabilities"]
    assert len(masses) == 4
    assert math.fsum(masses) == 1.0
    # The three members carry their own masses and the fourth branch is the
    # left-over mass of the group firing none of them, which is the identity.
    assert masses[:3] == (0.4, 0.35, 0.1)
    assert masses[3] == pytest.approx(0.15)
    # The branch that fires nothing is the identity over the frame's four wires,
    # scaled by the left-over mass, not the bare identity.
    assert grouped.matrix[3] == pytest.approx(
        math.sqrt(0.15) * torch.eye(16, dtype=torch.complex128)
    )


def test_a_group_that_always_fires_has_no_identity_branch() -> None:
    """A group holding one shot's worth of probability fires every shot."""

    circuit = circuit_from_detector_error_model(_ALTERNATIVES_THAT_ALWAYS_FIRE)
    frames = tuple(item for item in circuit.instructions if item.name == "dem_frame")

    assert len(frames) == 1
    assert len(frames[0].matrix) == 2
    assert math.fsum(frames[0].params["probabilities"]) == 1.0


def test_the_realization_samples_the_signature_a_fault_states_at_certainty() -> None:
    """At probability one every shot is the XOR of every signature, exactly."""

    expected = [0] * _ALWAYS.num_detectors + [0] * _ALWAYS.num_observables
    for error in _ALWAYS.errors:
        for index in error.detectors:
            expected[index] ^= 1
        for index in error.observables:
            expected[_ALWAYS.num_detectors + index] ^= 1

    shots = sample_noisy_measurements(
        circuit_from_detector_error_model(_ALWAYS), shots=64, seed=_SEED
    )
    assert shots.tolist() == [expected] * 64
    # The model's own route states the same bits, which is what makes this an
    # exact check of the realization rather than of one sampler.
    model_side = _ALWAYS.dem_sampling(shots=64, seed=_SEED)
    assert model_side.detectors.tolist() == [expected[: _ALWAYS.num_detectors]] * 64
    assert model_side.observables.tolist() == [expected[_ALWAYS.num_detectors :]] * 64


def test_a_group_that_always_fires_flips_its_shared_detector_on_every_shot() -> None:
    """Exclusivity is executable: one member fires per shot, so the shared wire does."""

    shots = sample_noisy_measurements(
        circuit_from_detector_error_model(_ALTERNATIVES_THAT_ALWAYS_FIRE),
        shots=_SHOTS,
        seed=_SEED,
    )
    assert bool((shots[:, 0] == 1).all())
    stated = _ALTERNATIVES_THAT_ALWAYS_FIRE.detector_rates()
    assert _worst_departure(shots[:, :2].double().mean(0), stated) <= _SIGMA


def test_the_sampled_rates_agree_with_the_rates_the_model_states(
    model: DetectorErrorModel,
) -> None:
    """The realization's shots carry the rates the model states, to within noise."""

    shots = sample_noisy_measurements(
        circuit_from_detector_error_model(model), shots=_SHOTS, seed=_SEED
    )
    detectors, observables = _sampled_rates(shots, model)
    assert _worst_departure(detectors, model.detector_rates()) <= _SIGMA
    assert _worst_departure(observables, model.observable_rates()) <= _SIGMA


def test_the_two_routes_agree_on_a_model_of_alternatives() -> None:
    """The model's own sampling and the realization's sampling agree per mechanism."""

    circuit_side = sample_noisy_measurements(
        circuit_from_detector_error_model(_ALTERNATIVES), shots=_SHOTS, seed=_SEED
    )
    model_side = _ALTERNATIVES.dem_sampling(shots=_SHOTS, seed=_SEED)

    for got, stated in (
        (circuit_side[:, :3].double().mean(0), model_side.detectors.double().mean(0)),
        (circuit_side[:, 3:].double().mean(0), model_side.observables.double().mean(0)),
    ):
        assert _worst_departure(got, stated) <= _SIGMA


def test_the_declared_mass_is_the_probability_the_model_states() -> None:
    """Why a probability is declared rather than recovered from its operator.

    The operators alone would be enough to *describe* a branch, and this file still
    reads the mass from the parameter instead. The number below is the reason:
    squaring the square root that wrote the operator does not return the mass for
    these values, so a reader that recovered it would return a different model.
    """

    lost = [value for value in (0.01, 0.1, 0.5, 0.9) if math.sqrt(value) ** 2 != value]
    assert lost, "the premise of the declared mass no longer holds for any of these"

    circuit = circuit_from_detector_error_model(_INDEPENDENT)
    restored = detector_error_model_from_circuit(circuit, num_detectors=3)
    assert tuple(error.probability for error in restored.errors) == tuple(
        error.probability for error in _INDEPENDENT.errors
    )


def test_the_split_is_the_callers_to_state() -> None:
    """A wire index does not say which side of the split it falls on."""

    circuit = circuit_from_detector_error_model(_INDEPENDENT)
    as_three = detector_error_model_from_circuit(circuit, num_detectors=3)
    as_two = detector_error_model_from_circuit(circuit, num_detectors=2)

    assert (as_three.num_detectors, as_three.num_observables) == (3, 1)
    assert (as_two.num_detectors, as_two.num_observables) == (2, 2)
    # The mechanisms are the same faults over the same register; only the label the
    # split gives each wire changes.
    assert sorted(_register_support(error, 3) for error in as_three.errors) == sorted(
        _register_support(error, 2) for error in as_two.errors
    )


def test_an_error_id_is_a_label_rather_than_a_quantity() -> None:
    """Renumbering the ids of a model changes no fault and no mass."""

    renumbered = DetectorErrorModel(
        3,
        1,
        tuple(
            DemError(
                error.probability,
                error.detectors,
                error.observables,
                None if error.error_id is None else 4000 + error.error_id,
            )
            for error in _ALTERNATIVES.errors
        ),
    )
    circuit = circuit_from_detector_error_model(renumbered)
    grouped = next(
        item
        for item in circuit.instructions
        if item.metadata.get("error_id") is not None
    )
    assert grouped.metadata["error_id"] == 4005
    assert detector_error_model_from_circuit(circuit, num_detectors=3) == renumbered


def test_the_response_of_the_external_interchange_survives_the_realization() -> None:
    """A realizable model stays as writable as it was before it was realized."""

    stim = pytest.importorskip("stim")
    for _, model in _MODELS:
        if model.stated_error_ids():
            continue
        circuit = circuit_from_detector_error_model(model)
        restored = detector_error_model_from_circuit(
            circuit, num_detectors=model.num_detectors
        )
        text = restored.to_stim_text()
        assert text == model.to_stim_text()
        external = stim.DetectorErrorModel(text)
        assert external.num_detectors == model.num_detectors
        assert external.num_observables == model.num_observables
        assert external.num_errors == model.num_errors


def test_a_model_of_alternatives_is_still_refused_by_the_text_writer() -> None:
    """Realizing a correlated model does not make it transcribable."""

    circuit = circuit_from_detector_error_model(_ALTERNATIVES)
    restored = detector_error_model_from_circuit(circuit, num_detectors=3)
    assert restored.stated_error_ids() == _ALTERNATIVES.stated_error_ids() == (5,)
    with pytest.raises(
        ValueError, match="cannot state that mechanisms are alternatives"
    ):
        restored.to_stim_text()


def test_the_class_route_reads_a_realization_the_same_way() -> None:
    """The reading is reachable from the model class, as construction is."""

    circuit = circuit_from_detector_error_model(_ALTERNATIVES)
    assert DetectorErrorModel.from_circuit(circuit, num_detectors=3) == _ALTERNATIVES


def test_the_class_route_refuses_what_the_reader_refuses() -> None:
    """The class route is the reading itself and not a laxer door to it."""

    with pytest.raises(ValueError, match="num_detectors must split"):
        DetectorErrorModel.from_circuit(_realization(), num_detectors=4)


def test_the_round_trip_is_stable_under_repetition() -> None:
    """A second pass through the pair returns the same program, bit for bit."""

    first = circuit_from_detector_error_model(_ALTERNATIVES)
    model = detector_error_model_from_circuit(first, num_detectors=3)
    second = circuit_from_detector_error_model(model)

    # The programs carry dense operators, and comparing those tensors elementwise
    # would ask a boolean of a matrix. The content hash is over the serialized
    # program, which is the comparison the round trip is about.
    assert first.content_hash == second.content_hash


def test_a_mechanism_at_probability_zero_is_refused() -> None:
    """A fault that never fires has no branch and so has no realization."""

    model = DetectorErrorModel(
        2, 1, (DemError(0.0, (0,), (0,)), DemError(0.1, (1,), ()))
    )
    with pytest.raises(ValueError, match="probability zero"):
        circuit_from_detector_error_model(model)


@pytest.mark.parametrize(
    ("split", "message"),
    [
        (0, "num_detectors must split"),
        (4, "num_detectors must split"),
        (-1, "num_detectors must be non-negative"),
    ],
)
def test_a_split_that_does_not_fit_is_refused(split: int, message: str) -> None:
    """The detector count has to name a side of the register."""

    with pytest.raises(ValueError, match=message):
        detector_error_model_from_circuit(_realization(), num_detectors=split)


@pytest.mark.parametrize("split", [True, 2.5, "2"])
def test_a_split_that_is_not_an_integer_is_refused(split: object) -> None:
    """A flag is not a count."""

    with pytest.raises(TypeError, match="num_detectors must be an integer"):
        detector_error_model_from_circuit(_realization(), num_detectors=split)


def test_a_program_with_an_opcode_that_is_not_a_frame_is_refused() -> None:
    """A realization holds frames and measurements and nothing else."""

    program = CircuitIR(
        n_wires=2,
        instructions=(Instruction(name="h", wires=(0,)), _measure(0), _measure(1)),
    )
    with pytest.raises(ValueError, match="neither a fault frame nor a measurement"):
        detector_error_model_from_circuit(program, num_detectors=1)


def test_a_frame_after_a_measurement_is_refused() -> None:
    """A frame has to reach the register the measurements read, not a later one."""

    program = CircuitIR(
        n_wires=2,
        instructions=(
            _measure(0),
            _frame((1,), ((_operator((1,), (1,)), 1.0),)),
            _measure(1),
        ),
    )
    with pytest.raises(ValueError, match="follows a measurement"):
        detector_error_model_from_circuit(program, num_detectors=1)


def test_a_wire_no_branch_flips_is_refused() -> None:
    """A frame wider than the operator it applies is refused rather than narrowed."""

    program = CircuitIR(
        n_wires=2,
        instructions=(
            _frame((0, 1), ((_operator((0, 1), (0,)), 1.0),)),
            _measure(0),
            _measure(1),
        ),
    )
    with pytest.raises(ValueError, match=r"names wire\(s\) \(1,\)"):
        detector_error_model_from_circuit(program, num_detectors=1)


def test_masses_that_disagree_with_the_operators_are_refused() -> None:
    """The declared masses and the operators are two statements of one channel."""

    program = CircuitIR(
        n_wires=1,
        instructions=(
            _frame(
                (0,),
                (
                    (math.sqrt(0.2) * _operator((0,), (0,)), 0.3),
                    (math.sqrt(0.8) * _operator((0,), ()), 0.7),
                ),
            ),
            _measure(0),
        ),
    )
    with pytest.raises(ValueError, match="must agree"):
        detector_error_model_from_circuit(program, num_detectors=1)


def test_masses_that_do_not_sum_to_one_are_refused() -> None:
    """A channel's branches are every way it can fire."""

    program = CircuitIR(
        n_wires=1,
        instructions=(
            _frame((0,), ((_operator((0,), (0,)), 0.5),)),
            _measure(0),
        ),
    )
    with pytest.raises(ValueError, match="summing to"):
        detector_error_model_from_circuit(program, num_detectors=1)


def test_a_missing_mass_is_refused() -> None:
    """A frame that states no masses states no channel."""

    program = CircuitIR(
        n_wires=1,
        instructions=(
            Instruction(
                name="dem_frame",
                wires=(0,),
                matrix=(_operator((0,), (0,)),),
                metadata={"is_channel": True},
            ),
            _measure(0),
        ),
    )
    with pytest.raises(ValueError, match="does not declare"):
        detector_error_model_from_circuit(program, num_detectors=1)


def test_a_frame_without_operators_is_refused() -> None:
    """A frame that declares no matrix declares no branch, whatever it states."""

    program = CircuitIR(
        n_wires=1,
        instructions=(
            Instruction(
                name="dem_frame",
                wires=(0,),
                params={"probabilities": (1.0,)},
                metadata={"is_channel": True},
            ),
            _measure(0),
        ),
    )
    with pytest.raises(ValueError, match="declares no operators"):
        detector_error_model_from_circuit(program, num_detectors=1)


def test_a_mass_count_that_does_not_match_the_branch_count_is_refused() -> None:
    """One mass per branch, in the order the branches are written."""

    program = CircuitIR(
        n_wires=1,
        instructions=(
            Instruction(
                name="dem_frame",
                wires=(0,),
                params={"probabilities": (0.5, 0.5)},
                matrix=(_operator((0,), (0,)),),
                metadata={"is_channel": True},
            ),
            _measure(0),
        ),
    )
    with pytest.raises(ValueError, match="one mass per branch"):
        detector_error_model_from_circuit(program, num_detectors=1)


def test_a_mass_that_is_not_a_probability_is_refused() -> None:
    """A mass is a real number between zero and one."""

    program = CircuitIR(
        n_wires=1,
        instructions=(
            Instruction(
                name="dem_frame",
                wires=(0,),
                params={"probabilities": (1.5, -0.5)},
                matrix=(
                    _operator((0,), (0,)),
                    _operator((0,), ()),
                ),
                metadata={"is_channel": True},
            ),
            _measure(0),
        ),
    )
    with pytest.raises(ValueError, match="between zero and one"):
        detector_error_model_from_circuit(program, num_detectors=1)


@pytest.mark.parametrize("mass", [True, "0.5"])
def test_a_mass_that_is_not_a_real_number_is_refused(mass: object) -> None:
    """A mass is a number, and a flag is not one however it compares."""

    program = CircuitIR(
        n_wires=1,
        instructions=(
            Instruction(
                name="dem_frame",
                wires=(0,),
                params={"probabilities": (mass,)},
                matrix=(_operator((0,), (0,)),),
                metadata={"is_channel": True},
            ),
            _measure(0),
        ),
    )
    with pytest.raises(TypeError, match="a mass is a real probability"):
        detector_error_model_from_circuit(program, num_detectors=1)


def test_a_bare_operator_where_branches_belong_is_refused() -> None:
    """A frame states one matrix per branch, so the operators are a sequence."""

    program = CircuitIR(
        n_wires=1,
        instructions=(
            Instruction(
                name="dem_frame",
                wires=(0,),
                params={"probabilities": (1.0,)},
                matrix=_operator((0,), (0,)),
                metadata={"is_channel": True},
            ),
            _measure(0),
        ),
    )
    with pytest.raises(ValueError, match="sequence of matrices"):
        detector_error_model_from_circuit(program, num_detectors=1)


def test_an_operator_of_the_wrong_shape_is_refused() -> None:
    """An operator that does not fit its wire list is refused rather than reshaped."""

    program = CircuitIR(
        n_wires=2,
        instructions=(
            _frame((0, 1), ((_operator((0,), (0,)), 1.0),)),
            _measure(0),
            _measure(1),
        ),
    )
    with pytest.raises(ValueError, match="branch 0 has shape"):
        detector_error_model_from_circuit(program, num_detectors=1)


def test_a_branch_that_is_not_a_product_of_bit_flips_is_refused() -> None:
    """A phase flip is not a detector error model mechanism."""

    program = CircuitIR(
        n_wires=1,
        instructions=(
            _frame((0,), ((_PHASE_FLIP, 1.0),)),
            _measure(0),
        ),
    )
    with pytest.raises(ValueError, match="not a product of bit flips"):
        detector_error_model_from_circuit(program, num_detectors=1)


def test_a_measurement_out_of_wire_order_is_refused() -> None:
    """The sample's columns are the wires in order, so the reads are too."""

    program = CircuitIR(
        n_wires=2,
        instructions=(
            _frame((0,), ((_operator((0,), (0,)), 1.0),)),
            _measure(1),
            _measure(0),
        ),
    )
    with pytest.raises(ValueError, match="measures wire 0"):
        detector_error_model_from_circuit(program, num_detectors=1)


def test_a_wire_that_is_never_measured_is_refused() -> None:
    """Every wire carries a column, so every wire is read out."""

    program = CircuitIR(
        n_wires=2,
        instructions=(
            _frame((0,), ((_operator((0,), (0,)), 1.0),)),
            _measure(0),
        ),
    )
    with pytest.raises(ValueError, match="measures each of its"):
        detector_error_model_from_circuit(program, num_detectors=1)


def test_a_lowered_measurement_node_is_refused() -> None:
    """A realization records its measurements as instructions, not as nodes."""

    program = CircuitIR(
        n_wires=2,
        instructions=(
            _frame((0,), ((_operator((0,), (0,)), 1.0),)),
            _measure(0),
            _measure(1),
        ),
        measurements=(MeasurementNode("measure", (0,)),),
    )
    with pytest.raises(ValueError, match="lowered measurement node"):
        detector_error_model_from_circuit(program, num_detectors=1)


def test_an_error_id_that_is_not_a_label_is_refused() -> None:
    """An id names a group, so it is a non-negative integer and nothing else."""

    program = CircuitIR(
        n_wires=1,
        instructions=(
            _frame((0,), ((_operator((0,), (0,)), 1.0),), error_id=-1),
            _measure(0),
        ),
    )
    with pytest.raises(ValueError, match="non-negative integer label"):
        detector_error_model_from_circuit(program, num_detectors=1)
