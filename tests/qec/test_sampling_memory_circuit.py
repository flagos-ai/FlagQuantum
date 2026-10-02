"""The native sampler for a configured memory experiment.

`sample_memory_circuit` places each noise location the configured
`PhenomenologicalNoise` names at the instruction it belongs to, executes the
result through the stabilizer engine, and reads the detection events and
observable flips off the record. The evidence here is in two parts, strongest
first.

The exact part runs the sampler at probability one, where every mechanism fires
on every shot and the answer is deterministic: the sample must be the parity the
model's own `dem_sampling` produces. That check is exact rather than statistical,
so it fails on a placement one instruction away from the right one instead of
averaging the mistake into a rate.

The statistical part runs at a physical strength and compares the sampled rates
against `DetectorErrorModel.detector_rates` and `observable_rates`, including the
detector pair rates that a marginal comparison cannot see. The chain to the
external reference is transitive rather than repeated here:
`test_dem_stim_reference_rates.py` already pins those model rates to a Stim
reference circuit, and this module pins the sampler to the model.

The remaining tests cover the placements the sampler derives — the round block, the
record column of every check's readout, and the two placements each noise family
sits at — together with the refusals that keep a misattributed mechanism from
being sampled at all.
"""

from __future__ import annotations

import builtins
import math
import sys
from dataclasses import dataclass, replace

import pytest
import torch

from flagquantum.qec import (
    DetectorErrorModel,
    MinimumWeightMatchingDecoder,
    PhenomenologicalNoise,
    RepetitionCode,
    RotatedSurfaceCode,
    build_memory_circuit,
    sample_memory_circuit,
)
from flagquantum.qec.circuit import MeasurementRef
from flagquantum.qec.codes import CodeCheck
from flagquantum.qec.pauli import Pauli
from flagquantum.qec.sampling import _measurement_plan, _noise_locations, _noisy_program
from flagquantum.simulation.stabilizer import StabilizerDependencyError

# Sampling executes the program through the stabilizer engine, which is an
# optional distribution, so the file skips rather than failing to import when it
# is absent. NumPy arrives with stim and the statistics below are written in it,
# so it is imported under the same guard, and neither is imported before it.
pytest.importorskip("stim")

import numpy as np
from numpy.typing import NDArray

pytestmark = pytest.mark.integration

# The strength and shot count `test_dem_stim_reference_rates.py` uses, so both
# halves of the chain are measured under one pair of settings.
_SHOTS = 400_000
_SIGMA = 4.0
_PROBABILITY = 0.01
_SEED = 7

# Two codes with different check structure: the repetition code's checks are
# disjoint, and the rotated surface code has X-type checks whose round-zero
# ancilla is prepared in |+>.
_CODES = (
    pytest.param(RepetitionCode(distance=3), 3, id="repetition-d3-r3"),
    pytest.param(RotatedSurfaceCode(distance=3), 3, id="surface-d3-r3"),
)

_NOISES = (
    pytest.param(PhenomenologicalNoise(data_flip=1.0), id="data-only"),
    pytest.param(PhenomenologicalNoise(measurement_flip=1.0), id="measurement-only"),
    pytest.param(PhenomenologicalNoise(data_flip=1.0, measurement_flip=1.0), id="both"),
)


def _memory(code, rounds: int):
    return build_memory_circuit(code, rounds=rounds)


def _tolerance(rates: NDArray[np.float64]) -> NDArray[np.float64]:
    """Four standard errors around each rate, never below one shot's resolution.

    The bound is the standard error of a rate of one half, which no sampled rate
    can exceed at this shot count, so a tolerance above it would no longer be a
    sampling error and the comparison would silently accept any divergence.
    """

    error = np.sqrt(np.maximum(rates * (1.0 - rates), 0.0) / _SHOTS)
    assert float(np.max(error)) <= math.sqrt(0.25 / _SHOTS), float(np.max(error))
    return np.maximum(_SIGMA * error, 1e-12)


def _exact_pair_rates(model: DetectorErrorModel) -> NDArray[np.float64]:
    """Return the exact rate at which each pair of detectors fires together.

    Every mechanism fires independently, so the sign product over the mechanisms
    flipping exactly one of a pair gives the pair's parity correlation and the
    rate follows from it. The result's diagonal is the marginal rate, which the
    marginal test asserts separately, so a derivation that drifted from the
    model's own arithmetic would not go unnoticed here.
    """

    signature = model.detector_error_matrix().numpy() != 0
    factor = np.array(
        [1.0 - 2.0 * error.probability for error in model.errors], dtype=np.float64
    )

    def product(pattern: NDArray[np.bool_]) -> NDArray[np.float64]:
        return np.prod(np.where(pattern, factor, 1.0), axis=-1)

    single = product(signature)
    pair = product(signature[:, None, :] != signature[None, :, :])
    return np.asarray(
        (1.0 - single[:, None] - single[None, :] + pair) / 4.0, dtype=np.float64
    )


@pytest.mark.parametrize(("code", "rounds"), _CODES)
@pytest.mark.parametrize("noise", _NOISES)
def test_every_firing_mechanism_reproduces_the_models_parity(
    code, rounds: int, noise: PhenomenologicalNoise
) -> None:
    """At probability one the sampler's record is the model's own arithmetic.

    Every mechanism fires on every shot, so the shot is fully determined and the
    comparison is exact. A data error moved to either side of its round boundary,
    or a measurement error moved past its readout, changes the parity and fails
    here rather than becoming a small difference in a rate.

    One placement is invisible to any parity comparison and is therefore not what
    this test pins: the repetition and rotated surface code gadgets prepare their
    ancilla with the CNOTs immediately preceding the readout, so moving a
    measurement channel one instruction earlier still reads the same parity. The
    instruction indices are pinned separately by
    `test_a_data_location_precedes_its_round_and_a_measurement_location_its_readout`,
    which is where a caller relies on the position rather than on the parity.
    """

    memory = _memory(code, rounds)
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    sample = sample_memory_circuit(memory, noise=noise, shots=4, seed=_SEED)
    expected = model.dem_sampling(shots=1, seed=0)

    assert sample.shots == 4
    for shot in range(sample.shots):
        assert sample.detectors[shot].tolist() == expected.detectors[0].tolist()
        assert sample.observables[shot].tolist() == expected.observables[0].tolist()


@pytest.mark.parametrize(("code", "rounds"), _CODES)
def test_a_noiseless_experiment_reports_no_detection_event(code, rounds: int) -> None:
    """A deterministic noiseless memory round detects nothing.

    Every detector compares two rounds that would have projected the data into the
    same eigenstate, so a detector firing here would mean the layout or the
    terminal readout block is offset from the syndrome bits it names.
    """

    memory = _memory(code, rounds)
    sample = sample_memory_circuit(
        memory, noise=PhenomenologicalNoise(), shots=64, seed=_SEED
    )

    assert sample.detectors.dtype == torch.int8
    assert sample.observables.dtype == torch.int8
    assert sample.detectors.shape == (64, len(memory.detectors.detectors))
    assert sample.observables.shape == (64, len(memory.observables.observables))
    assert int(sample.detectors.sum()) == 0
    assert int(sample.observables.sum()) == 0


@pytest.mark.parametrize(("code", "rounds"), _CODES)
def test_the_sampled_detector_rates_match_the_model(code, rounds: int) -> None:
    """The sampled rate of every detector has to be the model's predicted rate."""

    memory = _memory(code, rounds)
    noise = PhenomenologicalNoise(data_flip=_PROBABILITY, measurement_flip=_PROBABILITY)
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    sample = sample_memory_circuit(memory, noise=noise, shots=_SHOTS, seed=_SEED)

    observed = np.asarray(sample.detectors.numpy(), dtype=np.float64).mean(axis=0)
    expected = model.detector_rates().numpy()
    deviation = np.abs(observed - expected) / _tolerance(expected)

    assert float(deviation.max()) <= _SIGMA, (
        f"widest detector deviation is {float(deviation.max()):.2f} standard "
        f"errors at detector {int(deviation.argmax())}"
    )


@pytest.mark.parametrize(("code", "rounds"), _CODES)
def test_the_sampled_observable_rates_match_the_model(code, rounds: int) -> None:
    """The observable is the parity a logical failure rate is read from."""

    memory = _memory(code, rounds)
    noise = PhenomenologicalNoise(data_flip=_PROBABILITY, measurement_flip=_PROBABILITY)
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    sample = sample_memory_circuit(memory, noise=noise, shots=_SHOTS, seed=_SEED)

    observed = np.asarray(sample.observables.numpy(), dtype=np.float64).mean(axis=0)
    expected = model.observable_rates().numpy()
    deviation = np.abs(observed - expected) / _tolerance(expected)

    assert float(deviation.max()) <= _SIGMA, (
        f"widest observable deviation is {float(deviation.max()):.2f} standard "
        f"errors at observable {int(deviation.argmax())}"
    )


@pytest.mark.parametrize(("code", "rounds"), _CODES)
def test_the_sampled_detector_pair_rates_match_the_model(code, rounds: int) -> None:
    """Correlations are the half a marginal comparison cannot see.

    Two models with the same detector marginals and different mechanism
    signatures agree on every single-detector rate. The pair rates separate them,
    so they are what says each mechanism was placed where the model placed it
    rather than somewhere that happens to produce the same marginals.
    """

    memory = _memory(code, rounds)
    noise = PhenomenologicalNoise(data_flip=_PROBABILITY, measurement_flip=_PROBABILITY)
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    sample = sample_memory_circuit(memory, noise=noise, shots=_SHOTS, seed=_SEED)

    bits = np.asarray(sample.detectors.numpy(), dtype=np.float64)
    observed = bits.T @ bits / _SHOTS
    expected = _exact_pair_rates(model)
    deviation = np.abs(observed - expected) / _tolerance(expected)
    # The diagonal is the marginal rate, which the test above already compares.
    np.fill_diagonal(deviation, 0.0)

    assert float(deviation.max()) <= _SIGMA, (
        f"widest detector pair deviation is {float(deviation.max()):.2f} standard "
        f"errors at {np.unravel_index(int(deviation.argmax()), deviation.shape)}"
    )


def test_the_model_predicts_the_same_pair_rates_the_lower_bound_allows() -> None:
    """A sanity floor for the pair comparison above.

    The pair rates are derived from the model rather than measured, so the check
    it feeds is only as good as that derivation. Its diagonal has to reproduce the
    model's own marginal arithmetic, and the off-diagonal has to be non-trivial:
    if every pair rate were zero the comparison above would pass without looking
    at any correlation.
    """

    memory = _memory(RepetitionCode(distance=3), 3)
    noise = PhenomenologicalNoise(data_flip=_PROBABILITY, measurement_flip=_PROBABILITY)
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    pair = _exact_pair_rates(model)

    assert np.allclose(np.diag(pair), model.detector_rates().numpy())
    off_diagonal = pair[~np.eye(pair.shape[0], dtype=bool)]
    assert float(off_diagonal.min()) < float(off_diagonal.max())


def test_the_syndrome_round_is_the_programs_own_round_block() -> None:
    """The placement is read off the lowered program, and that is pinned here.

    A round index names one measured block, so the plan has to state where the
    block starts and where each check's readout sits inside it. For the
    three-data-qubit repetition code the emitted round is two CNOTs, a readout and
    a reset per check: eight instructions with the readouts at two and six.
    """

    memory = _memory(RepetitionCode(distance=3), 3)
    plan = _measurement_plan(memory)

    assert plan.block == 8
    assert plan.measure_offsets == (2, 6)
    assert plan.syndrome_columns[(0, 3)] == 0
    assert plan.syndrome_columns[(0, 4)] == 1
    assert plan.syndrome_columns[(2, 3)] == 4
    assert plan.syndrome_columns[(2, 4)] == 5
    assert plan.terminal_columns == {0: 6, 1: 7, 2: 8}


def test_a_data_location_precedes_its_round_and_a_measurement_location_its_readout() -> (
    None
):
    """The two families sit at different instructions, and the plan says which.

    A data error is a round-boundary error, so its instruction index is the
    round's first; a measurement error belongs immediately before the readout of
    the check it corrupts, so its index is that readout's offset inside the block.
    """

    memory = _memory(RepetitionCode(distance=3), 3)
    plan = _measurement_plan(memory)
    noise = PhenomenologicalNoise(data_flip=0.01, measurement_flip=0.02)

    located = {
        (item.kind, item.round_index, item.wire): item.instruction_index
        for item in _noise_locations(memory, plan, noise)
    }

    assert located[("data", 0, 0)] == 0
    assert located[("data", 1, 2)] == plan.block
    assert located[("data", 2, 1)] == 2 * plan.block
    assert located[("measurement", 0, 3)] == 2
    assert located[("measurement", 0, 4)] == 6
    assert located[("measurement", 2, 4)] == 2 * plan.block + 6
    # The index is the readout's own, and a channel emitted there is executed
    # before it. Parity cannot see this position: the CNOT that prepares the
    # ancilla is the instruction before, so one step earlier reads the same bit
    # and only the index says which side of the readout the flip is on.
    assert (
        tuple(
            located[("measurement", 0, check.ancilla_wire)]
            for check in memory.code.checks
        )
        == plan.measure_offsets
    )


def test_a_rounds_data_locations_are_one_group_before_the_round() -> None:
    """Three data wires at one round boundary is one group of three channels.

    The engine executes them before the round's first gate, which is what the
    model's data mechanism describes, so the channels appear as a leading block
    and the round's own instructions follow unchanged.
    """

    memory = _memory(RepetitionCode(distance=3), 3)
    plan = _measurement_plan(memory)
    noise = PhenomenologicalNoise(data_flip=0.01, measurement_flip=0.0)
    program = _noisy_program(plan, _noise_locations(memory, plan, noise), noise)

    leading = program.instructions[:3]
    assert [instruction.name for instruction in leading] == ["bit_flip"] * 3
    assert [instruction.wires for instruction in leading] == [(0,), (1,), (2,)]
    assert all(instruction.metadata.get("is_channel") for instruction in leading)
    # Round zero's own instructions follow the group unchanged; the next group
    # opens the round after it.
    assert (
        program.instructions[3 : 3 + plan.block]
        == plan.program.instructions[: plan.block]
    )
    assert [
        instruction.name
        for instruction in program.instructions[3 + plan.block : 3 + plan.block + 3]
    ] == ["bit_flip"] * 3
    assert (
        sum(1 for instruction in program.instructions if instruction.name == "bit_flip")
        == len(memory.code.data_wires) * memory.rounds
    )


def test_a_zero_probability_family_costs_no_engine_instruction() -> None:
    """A channel that cannot fire is dropped rather than executed every shot."""

    memory = _memory(RepetitionCode(distance=2), 2)
    plan = _measurement_plan(memory)
    quiet = PhenomenologicalNoise()

    assert _noise_locations(memory, plan, quiet) == ()
    assert _noisy_program(plan, (), quiet) == plan.program


def test_the_placement_does_not_depend_on_the_strength_it_is_built_for() -> None:
    """The locations come from the program, so the strengths only scale them.

    A plan that moved with the probabilities would make two sampled rates
    incomparable across a noise sweep, which is the workflow the model exists for.
    """

    memory = _memory(RepetitionCode(distance=3), 2)
    plan = _measurement_plan(memory)

    def positions(noise: PhenomenologicalNoise):
        return [
            (item.kind, item.round_index, item.wire, item.instruction_index)
            for item in _noise_locations(memory, plan, noise)
        ]

    weak = positions(PhenomenologicalNoise(data_flip=0.001, measurement_flip=0.001))
    strong = positions(PhenomenologicalNoise(data_flip=0.2, measurement_flip=0.2))

    assert weak == strong
    assert (
        len(weak)
        == len(memory.code.data_wires) * memory.rounds
        + len(memory.code.checks) * memory.rounds
    )


def test_planning_the_placement_does_not_need_the_optional_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deriving a placement is Core arithmetic and installs nothing.

    A caller planning a sweep should not have to install the sampling engine to
    find out where the noise goes, and a missing engine has to surface at the
    execution call rather than inside the plan.
    """

    real_import = builtins.__import__

    def refuse(name: str, *args: object, **kwargs: object):
        if name == "stim" or name.startswith("stim."):
            raise ImportError("No module named 'stim'")
        return real_import(name, *args, **kwargs)

    memory = _memory(RepetitionCode(distance=3), 2)
    noise = PhenomenologicalNoise(data_flip=0.01, measurement_flip=0.01)
    plan = _measurement_plan(memory)
    program = _noisy_program(plan, _noise_locations(memory, plan, noise), noise)

    monkeypatch.delitem(sys.modules, "stim", raising=False)
    monkeypatch.setattr(builtins, "__import__", refuse)

    assert len(program.instructions) > len(plan.program.instructions)
    with pytest.raises(StabilizerDependencyError, match=r"flagquantum\[stim\]"):
        sample_memory_circuit(memory, noise=noise, shots=4, seed=_SEED)


def test_a_program_that_is_not_a_whole_number_of_rounds_is_refused() -> None:
    """A prologue outside the loop makes the instruction count indivisible."""

    memory = _memory(RepetitionCode(distance=3), 2)
    broken = replace(
        memory,
        source=memory.source.replace(
            "    for round_index in range(rounds):\n",
            "    qp.x(wires=0)\n    for round_index in range(rounds):\n",
        ),
    )

    with pytest.raises(
        ValueError, match="cannot divide into identical syndrome rounds"
    ):
        sample_memory_circuit(
            broken, noise=PhenomenologicalNoise(data_flip=0.01), shots=8, seed=_SEED
        )


def test_a_program_whose_rounds_are_not_identical_is_refused() -> None:
    """Equal-length rounds are not enough: the rounds have to repeat themselves.

    The count above catches a prologue, which no round can own. This catches the
    sharper error: a body whose gates depend on ``round_index`` lowers to the same
    *number* of instructions per round and a different *set*, so a block boundary
    still exists while "round 1" no longer means what the model's round 1 means.
    Dividing it into rounds would attribute a mechanism to a round that did not
    execute it.
    """

    memory = _memory(RepetitionCode(distance=3), 2)
    broken = replace(
        memory,
        source=memory.source.replace(
            "        qp.CNOT(wires=[1, 3])\n",
            "        qp.CNOT(wires=[1, 3])\n        qp.x(wires=round_index)\n",
        ),
    )

    with pytest.raises(ValueError, match="is not identical to round zero"):
        sample_memory_circuit(
            broken, noise=PhenomenologicalNoise(data_flip=0.01), shots=8, seed=_SEED
        )


def test_a_program_that_measures_its_checks_in_another_order_is_refused() -> None:
    """A check index is only a record column if the readouts follow the code.

    The code's declared check order is what the model reads its syndrome bits
    against, so a program that measures the same checks in a different order would
    attribute every mechanism to the wrong check.
    """

    memory = _memory(RepetitionCode(distance=3), 2)
    lines = memory.source.splitlines(keepends=True)
    first = next(
        index
        for index, line in enumerate(lines)
        if "measure" in line and "wires=3)" in line
    )
    second = next(
        index
        for index, line in enumerate(lines)
        if "measure" in line and "wires=4)" in line
    )
    lines[first], lines[second] = lines[second], lines[first]

    with pytest.raises(ValueError, match="declared order"):
        sample_memory_circuit(
            replace(memory, source="".join(lines)),
            noise=PhenomenologicalNoise(data_flip=0.01),
            shots=8,
            seed=_SEED,
        )


def test_a_detector_naming_an_ancilla_no_check_owns_is_refused() -> None:
    """A misattributed mechanism is refused instead of raising a bare `KeyError`.

    A `MemoryCircuit` ties a syndrome reference to the code's *declared ancilla
    wires*, while this module ties one to the code's *checks*. A code that
    declares an ancilla no check reads therefore passes the layout's validation
    and still names a bit nothing records, so the refusal is stated here rather
    than left to a dictionary lookup.
    """

    inner = RepetitionCode(distance=3)
    memory = _memory(inner, 2)
    orphaned = _OrphanAncilla(inner, orphan_wire=5)
    # A syndrome detector compares two rounds, so it reads an ancilla rather than
    # a data wire, which is what makes it the reference that can go orphaned.
    index = next(
        position
        for position, item in enumerate(memory.detectors.detectors)
        if item.parity[0].round_index is not None
    )
    detector = memory.detectors.detectors[index]
    owned = detector.parity[0].wire

    circuit = replace(
        memory,
        code=orphaned,
        detectors=replace(
            memory.detectors,
            detectors=memory.detectors.detectors[:index]
            + (
                replace(
                    detector,
                    parity=(MeasurementRef(detector.parity[0].round_index, 5),)
                    + detector.parity[1:],
                ),
            )
            + memory.detectors.detectors[index + 1 :],
        ),
    )

    assert owned in orphaned.ancilla_wires
    assert 5 in orphaned.ancilla_wires
    assert all(check.ancilla_wire != 5 for check in orphaned.checks)
    with pytest.raises(ValueError, match="which no check owns"):
        sample_memory_circuit(
            circuit, noise=PhenomenologicalNoise(data_flip=0.01), shots=4, seed=_SEED
        )


@dataclass(frozen=True)
class _OrphanAncilla:
    """A repetition code that declares one ancilla wire no check reads.

    The declared wire count and the declared wires are what the memory circuit's
    layout validation reads, so this is the smallest code that reaches the
    sampler with an unattributable detector reference.
    """

    inner: RepetitionCode
    orphan_wire: int

    @property
    def distance(self) -> int:
        return self.inner.distance

    @property
    def num_data_qubits(self) -> int:
        return self.inner.num_data_qubits

    @property
    def num_ancilla_qubits(self) -> int:
        return self.inner.num_ancilla_qubits + 1

    @property
    def data_wires(self) -> tuple[int, ...]:
        return self.inner.data_wires

    @property
    def ancilla_wires(self) -> tuple[int, ...]:
        return self.inner.ancilla_wires + (self.orphan_wire,)

    @property
    def checks(self) -> tuple[CodeCheck, ...]:
        return self.inner.checks

    @property
    def stabilizers(self) -> tuple[Pauli, ...]:
        return self.inner.stabilizers

    @property
    def logical_observables(self) -> tuple[Pauli, ...]:
        return self.inner.logical_observables


@pytest.mark.parametrize(
    ("noise", "shots", "seed", "error", "match"),
    [
        (PhenomenologicalNoise(), 0, None, ValueError, "positive integer"),
        (PhenomenologicalNoise(), 1.5, None, TypeError, "positive integer"),
        (PhenomenologicalNoise(), True, None, TypeError, "positive integer"),
        (PhenomenologicalNoise(), 1, 1.5, TypeError, "integer or None"),
        (PhenomenologicalNoise(), 1, True, TypeError, "integer or None"),
    ],
)
def test_the_shapes_the_sampler_refuses(
    noise: PhenomenologicalNoise,
    shots: object,
    seed: object,
    error: type[Exception],
    match: str,
) -> None:
    """A wrong argument is named rather than sampled."""

    memory = _memory(RepetitionCode(distance=2), 2)

    with pytest.raises(error, match=match):
        sample_memory_circuit(memory, noise=noise, shots=shots, seed=seed)


def test_the_sampler_requires_a_memory_circuit_and_a_noise_record() -> None:
    """The two arguments are checked before anything is lowered."""

    memory = _memory(RepetitionCode(distance=2), 2)

    with pytest.raises(TypeError, match="MemoryCircuit"):
        sample_memory_circuit("not a circuit", noise=PhenomenologicalNoise(), shots=1)
    with pytest.raises(TypeError, match="PhenomenologicalNoise"):
        sample_memory_circuit(memory, noise="not noise", shots=1)


def test_the_sampler_never_calls_the_model_to_produce_a_shot() -> None:
    """The model states the noise; the sampler does not replay its mechanisms.

    The model's own `dem_sampling` answers "what does the model say a shot looks
    like", which is a different question from "what does this circuit read out".
    Reusing it would make the agreement between the two a tautology, so the
    sampler's own execution path has to be reached without it.
    """

    memory = _memory(RepetitionCode(distance=3), 2)
    noise = PhenomenologicalNoise(data_flip=0.05, measurement_flip=0.05)
    sample = sample_memory_circuit(memory, noise=noise, shots=8, seed=_SEED)
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)

    # Two independent draws differ, so the agreement asserted elsewhere at
    # probability one is not an artifact of both paths returning one constant.
    independent = model.dem_sampling(shots=8, seed=_SEED)
    assert not torch.equal(sample.detectors, independent.detectors)
    assert sample.detectors.shape == independent.detectors.shape


# A logical failure rate is read from a sampled circuit and a decoder built from
# the same circuit's model. The shot count is lower than the rate tests' because
# the comparison is between two rates measured the same way, not between a rate
# and a model's prediction.
_DECODE_SHOTS = 8_000


def _decode_logical_failures(
    memory, model: DetectorErrorModel, sample, observable: int = 0
) -> tuple[int, int]:
    """Return the raw and post-decode failure counts of one sampled experiment."""

    decoder = MinimumWeightMatchingDecoder.from_detector_error_model(model)
    detectors = sample.detectors
    actual = sample.observables[:, observable]
    predicted = torch.zeros_like(actual)
    for shot in range(detectors.shape[0]):
        events = tuple(
            int(index)
            for index in (detectors[shot] != 0).nonzero().reshape(-1).tolist()
        )
        if observable in decoder.decode(events).observables:
            predicted[shot] = 1
    return int(actual.sum()), int((predicted != actual).sum())


@pytest.mark.parametrize(
    ("code", "rounds"),
    [
        pytest.param(RepetitionCode(distance=3), 3, id="repetition-d3-r3"),
        pytest.param(RotatedSurfaceCode(distance=3), 3, id="surface-d3-r3"),
    ],
)
def test_a_sampled_experiment_decodes_through_the_model_it_shares(
    code, rounds: int
) -> None:
    """The join's purpose is a decoding claim with no statevector ceiling.

    The detection events come from the circuit sampler and the decoder is built
    from the model derived from the same circuit, which is the route the row
    exists to provide. The rate falling is what says the two halves agree about
    which mechanisms flip which detectors: a decoder fed events from a circuit
    the model does not describe corrects nothing.
    """

    memory = _memory(code, rounds)
    noise = PhenomenologicalNoise(data_flip=_PROBABILITY, measurement_flip=_PROBABILITY)
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    sample = sample_memory_circuit(memory, noise=noise, shots=_DECODE_SHOTS, seed=_SEED)

    raw, decoded = _decode_logical_failures(memory, model, sample)

    assert raw > 0, "the raw observable flips this comparison needs are absent"
    assert decoded < raw / 4, (raw, decoded)


def test_a_larger_distance_decodes_to_a_lower_logical_failure_rate() -> None:
    """The residual falls with distance, which a wrong join would not show.

    If the sampler placed a mechanism where the model did not, the decoder's
    correction would be wrong about as often as the noise fires, and the residual
    would not fall as the code grows.
    """

    noise = PhenomenologicalNoise(data_flip=_PROBABILITY, measurement_flip=_PROBABILITY)
    residuals = {}
    for distance in (3, 5):
        memory = _memory(RepetitionCode(distance=distance), distance)
        model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
        sample = sample_memory_circuit(
            memory, noise=noise, shots=_DECODE_SHOTS, seed=_SEED
        )
        _, residuals[distance] = _decode_logical_failures(memory, model, sample)

    assert residuals[5] < residuals[3], residuals
