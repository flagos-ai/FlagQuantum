"""Unit coverage for the decoder inputs a memory experiment implies.

A detector error model says which mechanisms a decoder can see. It does not say
where in a shot each detector's parity is read, and a matcher needs both halves:
it is handed a syndrome over raw measurements and has to know which measurements
compose each detector and each observable. Upstream cudaq-qec hands both over
together, through ``decoder_context_from_memory_circuit`` and the
``decoder_inputs`` record its components return.

These tests pin the local counterpart: the maps, their two projections, the
component split, and -- the claim the whole module rests on -- that the numbering
this module derives from the circuit's *declaration* is the numbering the
sampler derives from the *lowered program*.
"""

from __future__ import annotations

import pytest
import torch

from flagquantum.qec.circuit import build_memory_circuit
from flagquantum.qec.codes import RepetitionCode
from flagquantum.qec.context import (
    DecoderContext,
    DecoderInputs,
    MeasurementMap,
    decoder_context_from_memory_circuit,
)
from flagquantum.qec.dem import DemError, DetectorErrorModel
from flagquantum.qec.noise import PhenomenologicalNoise
from flagquantum.qec.sampling import _measurement_plan
from flagquantum.qec.surface import RotatedSurfaceCode, ZxxzSurfaceCode

pytestmark = pytest.mark.unit

_NOISE = PhenomenologicalNoise(data_flip=0.01, measurement_flip=0.02)

_CODES = {
    "surface": RotatedSurfaceCode(distance=3),
    "repetition": RepetitionCode(distance=3),
}
_CIRCUITS: dict[tuple[str, int], object] = {}
_CONTEXTS: dict[tuple[str, int], DecoderContext] = {}


def _kind(code) -> str:
    return "repetition" if isinstance(code, RepetitionCode) else "surface"


def _circuit(code=None, *, rounds: int = 3):
    """Return a configured memory circuit, built once per code and round count.

    Building a distance-three surface code's memory circuit and its layouts is
    the expensive half of every test here, and the records are immutable, so one
    build is enough for all of them.
    """

    key = (_kind(code), rounds)
    if key not in _CIRCUITS:
        _CIRCUITS[key] = build_memory_circuit(
            code if code is not None else _CODES["surface"], rounds=rounds
        )
    return _CIRCUITS[key]


def _context(code=None, *, rounds: int = 3) -> DecoderContext:
    key = (_kind(code), rounds)
    if key not in _CONTEXTS:
        _CONTEXTS[key] = decoder_context_from_memory_circuit(
            _circuit(code, rounds=rounds), noise=_NOISE
        )
    return _CONTEXTS[key]


def _sampler_columns(circuit):
    """Return the record column of every measurement reference, as the sampler reads it.

    The plan is private on purpose: this is the one place the two independent
    readings of the record convention are compared, and reaching for the
    sampler's own plan is what makes the comparison a measurement rather than a
    restatement of the construction.
    """

    plan = _measurement_plan(circuit)

    def column(reference):
        if reference.round_index is None:
            return plan.terminal_columns[reference.wire]
        return plan.syndrome_columns[(reference.round_index, reference.wire)]

    return column, plan


# --------------------------------------------------------------------------
# the map record
# --------------------------------------------------------------------------


def test_a_row_is_the_measurements_one_parity_reads() -> None:
    measurement_map = MeasurementMap(num_measurements=6, rows=((3, 1), (), (0,)))
    assert measurement_map.rows == ((1, 3), (), (0,))
    assert measurement_map.num_rows == 3
    assert measurement_map.num_entries == 3


def test_a_row_entry_is_a_measurement_index_and_not_a_quantity() -> None:
    with pytest.raises(TypeError, match="must contain integer indices"):
        MeasurementMap(num_measurements=4, rows=((0.5,),))
    # A bool is an int in Python, and True would silently be the index 1.
    with pytest.raises(TypeError, match="must contain integer indices"):
        MeasurementMap(num_measurements=4, rows=((True,),))
    # A negative index is not a measurement, so it is out of range rather than a
    # second kind of bad value: the buffer has no entries before zero.
    with pytest.raises(ValueError, match="which is outside the 4 measurement"):
        MeasurementMap(num_measurements=4, rows=((-1,),))


def test_a_row_index_outside_the_buffer_is_refused() -> None:
    """A map that names a measurement the experiment never records is not a map."""

    with pytest.raises(ValueError, match="outside the 4 measurement"):
        MeasurementMap(num_measurements=4, rows=((4,),))
    assert MeasurementMap(num_measurements=4, rows=((3,),)).rows == ((3,),)


def test_a_row_cannot_name_one_measurement_twice() -> None:
    """Two reads of one measurement cancel, so a row that repeats one states nothing."""

    with pytest.raises(ValueError, match="names measurement 2 twice"):
        MeasurementMap(num_measurements=4, rows=((2, 2),))


def test_the_measurement_count_is_a_non_negative_integer() -> None:
    with pytest.raises(TypeError, match="measurement count must be an integer"):
        MeasurementMap(num_measurements=2.0, rows=())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="measurement count must be an integer"):
        MeasurementMap(num_measurements=True, rows=())
    with pytest.raises(ValueError, match="measurement count must be non-negative"):
        MeasurementMap(num_measurements=-1, rows=())
    assert MeasurementMap(num_measurements=0, rows=()).num_measurements == 0


def test_the_dense_form_is_one_row_per_parity_and_one_column_per_measurement() -> None:
    measurement_map = MeasurementMap(num_measurements=4, rows=((1, 3), (), (2,)))
    dense = measurement_map.dense()
    assert dense.shape == (3, 4)
    assert dense.dtype == torch.int8
    assert dense.tolist() == [[0, 1, 0, 1], [0, 0, 0, 0], [0, 0, 1, 0]]


def test_the_dense_form_and_the_flattened_form_agree() -> None:
    """Both projections are readings of the rows, so they cannot disagree."""

    measurement_map = MeasurementMap(num_measurements=5, rows=((0, 4), (2,), ()))
    dense = measurement_map.dense()
    sparse = measurement_map.flattened()
    rebuilt = []
    row: list[int] = []
    for entry in sparse:
        if entry == -1:
            rebuilt.append(row)
            row = []
        else:
            row.append(entry)
    assert tuple(tuple(entry) for entry in rebuilt) == measurement_map.rows
    assert rebuilt
    for index, entries in enumerate(rebuilt):
        assert torch.nonzero(dense[index]).flatten().tolist() == entries


def test_the_terminator_is_what_makes_an_empty_row_readable() -> None:
    """Without it a row that reads nothing would be invisible to a sparse reader."""

    measurement_map = MeasurementMap(num_measurements=3, rows=((), (1,), ()))
    assert measurement_map.flattened() == (-1, 1, -1, -1)


# --------------------------------------------------------------------------
# the inputs record
# --------------------------------------------------------------------------


def _inputs(detectors: int = 2, observables: int = 1, *, width: int = 4):
    return DecoderInputs(
        dem=DetectorErrorModel(
            num_detectors=detectors,
            num_observables=observables,
            errors=(DemError(probability=0.1, detectors=(0,), observables=(0,)),),
        ),
        measurement_to_detectors=MeasurementMap(
            num_measurements=width, rows=tuple((index,) for index in range(detectors))
        ),
        measurement_to_observables=MeasurementMap(
            num_measurements=width,
            rows=tuple((detectors + index,) for index in range(observables)),
        ),
    )


def test_inputs_require_a_model_and_two_maps() -> None:
    good = _inputs()
    with pytest.raises(TypeError, match="require a detector error model"):
        DecoderInputs(
            dem="not a model",  # type: ignore[arg-type]
            measurement_to_detectors=good.measurement_to_detectors,
            measurement_to_observables=good.measurement_to_observables,
        )
    with pytest.raises(TypeError, match="detector map must be a MeasurementMap"):
        DecoderInputs(
            dem=good.dem,
            measurement_to_detectors=(),  # type: ignore[arg-type]
            measurement_to_observables=good.measurement_to_observables,
        )
    with pytest.raises(TypeError, match="observable map must be a MeasurementMap"):
        DecoderInputs(
            dem=good.dem,
            measurement_to_detectors=good.measurement_to_detectors,
            measurement_to_observables=(),  # type: ignore[arg-type]
        )
    assert good.num_measurements == 4


def test_inputs_refuse_a_detector_map_with_the_wrong_row_count() -> None:
    good = _inputs()
    with pytest.raises(ValueError, match="one row per detector"):
        DecoderInputs(
            dem=good.dem,
            measurement_to_detectors=MeasurementMap(num_measurements=4, rows=((0,),)),
            measurement_to_observables=good.measurement_to_observables,
        )


def test_inputs_refuse_an_observable_map_with_the_wrong_row_count() -> None:
    good = _inputs()
    with pytest.raises(ValueError, match="one row per observable"):
        DecoderInputs(
            dem=good.dem,
            measurement_to_detectors=good.measurement_to_detectors,
            measurement_to_observables=MeasurementMap(num_measurements=4, rows=()),
        )


def test_inputs_refuse_two_maps_read_from_different_buffers() -> None:
    """One shot records one set of measurements, so the two widths are one width."""

    good = _inputs()
    with pytest.raises(ValueError, match="same measurement buffer"):
        DecoderInputs(
            dem=good.dem,
            measurement_to_detectors=good.measurement_to_detectors,
            measurement_to_observables=MeasurementMap(num_measurements=5, rows=((2,),)),
        )


# --------------------------------------------------------------------------
# the context record
# --------------------------------------------------------------------------


def test_a_context_requires_a_memory_circuit() -> None:
    circuit = _circuit()
    dem = DetectorErrorModel(num_detectors=len(circuit.detectors), num_observables=1)
    with pytest.raises(TypeError, match="requires a MemoryCircuit"):
        DecoderContext(circuit="nope", dem=dem)  # type: ignore[arg-type]


def test_a_context_requires_a_detector_error_model() -> None:
    circuit = _circuit()
    with pytest.raises(TypeError, match="requires a detector error model"):
        DecoderContext(circuit=circuit, dem=())  # type: ignore[arg-type]


def test_a_context_refuses_a_model_for_a_different_experiment() -> None:
    circuit = _circuit()
    with pytest.raises(ValueError, match="detector count must be the circuit's"):
        DecoderContext(
            circuit=circuit,
            dem=DetectorErrorModel(
                num_detectors=len(circuit.detectors) - 1, num_observables=1
            ),
        )
    with pytest.raises(ValueError, match="observable count must be the circuit's"):
        DecoderContext(
            circuit=circuit,
            dem=DetectorErrorModel(
                num_detectors=len(circuit.detectors), num_observables=2
            ),
        )


def test_the_entry_point_requires_a_memory_circuit_and_a_noise_record() -> None:
    with pytest.raises(TypeError, match="circuit must be a MemoryCircuit"):
        decoder_context_from_memory_circuit("nope", noise=_NOISE)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="noise must be a PhenomenologicalNoise"):
        decoder_context_from_memory_circuit(_circuit(), noise="nope")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# the numbering is the sampler's numbering
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rounds", [1, 2])
def test_the_map_numbering_is_the_numbering_the_sampler_records(rounds: int) -> None:
    """The declaration and the lowered program must agree, and here is the check.

    The sampler derives its record columns from the lowered program and this
    module derives them from the circuit's own layouts. If the two readings ever
    drifted, a decoder fed a sampled syndrome would be matching detectors against
    measurements that do not compose them, and nothing else in the package would
    notice.
    """

    circuit = _circuit(rounds=rounds)
    column, plan = _sampler_columns(circuit)
    inputs = _context(rounds=rounds).full_component()
    assert inputs.measurement_to_detectors.rows == tuple(
        tuple(sorted(column(reference) for reference in detector.parity))
        for detector in circuit.detectors.detectors
    )
    assert inputs.measurement_to_observables.rows == tuple(
        tuple(sorted(column(reference) for reference in observable.measurement_parity))
        for observable in circuit.observables.observables
    )
    width = max(
        list(plan.syndrome_columns.values()) + list(plan.terminal_columns.values())
    )
    assert (
        inputs.num_measurements
        == width + 1
        == _context(rounds=rounds).num_measurements()
    )


def test_the_measurement_count_is_one_per_check_per_round_plus_the_readout() -> None:
    circuit = _circuit()
    checks = len(circuit.code.checks)
    assert _context().num_measurements() == checks * circuit.rounds + len(
        circuit.code.data_wires
    )


def test_the_terminal_readout_follows_the_last_syndrome_round() -> None:
    """The readout is appended, so round `r` cannot address it as a syndrome."""

    circuit = _circuit(rounds=3)
    _, plan = _sampler_columns(circuit)
    checks = len(circuit.code.checks)
    assert sorted(plan.syndrome_columns.values()) == list(range(checks * 3))
    assert sorted(plan.terminal_columns.values()) == list(
        range(checks * 3, checks * 3 + len(circuit.code.data_wires))
    )


def test_a_repetition_code_numbers_the_same_way() -> None:
    circuit = _circuit(RepetitionCode(distance=3))
    column, _ = _sampler_columns(circuit)
    inputs = _context(RepetitionCode(distance=3)).full_component()
    assert inputs.measurement_to_detectors.rows == tuple(
        tuple(sorted(column(reference) for reference in detector.parity))
        for detector in circuit.detectors.detectors
    )


# --------------------------------------------------------------------------
# the component split
# --------------------------------------------------------------------------


def _component_detectors(context: DecoderContext, *, x_type: bool) -> list[int]:
    """Return the layout indices of the detectors one basis component keeps.

    A detector belongs to the check whose ancilla its syndrome measurements
    read, so the basis is decided by that check's type. The rule is stated here
    again rather than reached for in the module, so the test would notice a
    change in the split rather than restate it.
    """

    by_ancilla = {check.ancilla_wire: check for check in context.circuit.code.checks}
    kept: list[int] = []
    for detector in context.circuit.detectors.detectors:
        ancillas = {
            reference.wire
            for reference in detector.parity
            if reference.round_index is not None
        }
        assert len(ancillas) == 1
        if bool(by_ancilla[next(iter(ancillas))].stabilizer.x_wires) is x_type:
            kept.append(detector.index)
    return kept


def test_the_full_component_is_the_model_as_built() -> None:
    context = _context()
    inputs = context.full_component()
    assert inputs.dem == context.dem
    assert inputs.dem == DetectorErrorModel.from_memory_circuit(
        context.circuit, noise=_NOISE
    )
    assert inputs.measurement_to_detectors.num_rows == context.dem.num_detectors
    assert inputs.measurement_to_observables.num_rows == context.dem.num_observables


def test_the_two_components_partition_the_detectors() -> None:
    """Every detector is carried by exactly one check, so the split is a partition."""

    context = _context()
    x_kept = _component_detectors(context, x_type=True)
    z_kept = _component_detectors(context, x_type=False)
    assert sorted(x_kept + z_kept) == list(range(context.dem.num_detectors))
    assert not set(x_kept) & set(z_kept)
    assert context.x_component().dem.num_detectors == len(x_kept)
    assert context.z_component().dem.num_detectors == len(z_kept)


def test_the_z_component_holds_the_terminal_detectors() -> None:
    """A terminal detector compares a Z check with the Z-basis data readout."""

    context = _context()
    terminals = [
        detector.index
        for detector in context.circuit.detectors.detectors
        if any(reference.round_index is None for reference in detector.parity)
    ]
    assert terminals
    z_kept = _component_detectors(context, x_type=False)
    assert set(terminals) <= set(z_kept)
    assert not set(terminals) & set(_component_detectors(context, x_type=True))
    z_checks = sum(
        1 for check in context.circuit.code.checks if not check.stabilizer.x_wires
    )
    assert len(terminals) == z_checks
    assert len(z_kept) == z_checks * (context.circuit.rounds + 1)


def test_a_component_keeps_every_detectors_marginal_rate() -> None:
    """The split is a reading of the model, so it may not change what a detector sees."""

    context = _context()
    full_rates = context.dem.detector_rates()
    for x_type, component in (
        (True, context.x_component()),
        (False, context.z_component()),
    ):
        kept = _component_detectors(context, x_type=x_type)
        component_rates = component.dem.detector_rates()
        assert len(component_rates) == len(kept)
        for position, original in enumerate(kept):
            assert float(component_rates[position]) == pytest.approx(
                float(full_rates[original])
            )


def test_a_component_keeps_an_undetectable_logical_fault() -> None:
    """A mechanism with no detector signature is a logical fault, and it survives.

    Projecting onto a basis cannot make an undetectable fault detectable, and
    dropping it would change the observable rate the model states.
    """

    context = _context(RepetitionCode(distance=3))
    full = context.full_component()
    assert full.dem.num_observables == 1
    assert float(context.z_component().dem.observable_rates()[0]) == pytest.approx(
        float(full.dem.observable_rates()[0])
    )


def test_an_empty_component_is_refused() -> None:
    """A repetition code has terminal-only Z detectors, so its X component is empty."""

    with pytest.raises(ValueError, match="no detector carried by an X-type check"):
        _context(RepetitionCode(distance=3)).x_component()
    assert _context(RepetitionCode(distance=3)).z_component().dem.num_detectors > 0


def test_a_component_of_an_id_carrying_model_is_refused() -> None:
    """A projection may not silently drop a correlation or merge two alternatives."""

    circuit = _circuit()
    dem = DetectorErrorModel(
        num_detectors=len(circuit.detectors),
        num_observables=1,
        errors=(
            DemError(probability=0.1, detectors=(0,), error_id=7),
            DemError(probability=0.1, detectors=(0,), error_id=7),
        ),
    )
    context = DecoderContext(circuit=circuit, dem=dem)
    with pytest.raises(ValueError, match="cannot be read from a model that states"):
        context.x_component()
    assert context.full_component().dem == dem
    assert context.full_component().dem.stated_error_ids() == (7,)


def test_a_component_of_a_mixed_check_model_is_refused() -> None:
    """A split is defined only for a code whose checks are pure.

    The two components keep the detectors a check of one basis type carries, so
    the split needs every check to have exactly one type. A ZXXZ patch has none:
    every check is a product of an X factor and a Z factor, its detectors fire for
    faults of both families, and there is no honest way to put one of them in a
    single component. The refusal names the check rather than dropping it, and it
    is answered in both bases, because a caller that asked the other way round
    would otherwise be handed the complement of an arbitrary choice.

    The whole model is still readable, which is what keeps this a deliberate
    limit rather than a lost capability.
    """

    context = decoder_context_from_memory_circuit(
        build_memory_circuit(ZxxzSurfaceCode(distance=3), rounds=2), noise=_NOISE
    )
    with pytest.raises(ValueError, match="mixed X-and-Z stabilizer"):
        context.x_component()
    with pytest.raises(ValueError, match="belong to neither basis component"):
        context.z_component()
    assert context.full_component().dem.num_detectors > 0

    """Two contexts over one circuit agree, and a component is stable across reads."""

    context = _context()
    first = context.z_component()
    second = context.z_component()
    assert first == second
    assert first.dem == second.dem
    assert first.measurement_to_detectors.rows == second.measurement_to_detectors.rows
    assert (
        decoder_context_from_memory_circuit(_circuit(), noise=_NOISE).z_component()
        == first
    )
