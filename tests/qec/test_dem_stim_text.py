"""Unit coverage for the stim text interchange."""

from __future__ import annotations

import pytest

from flagquantum.qec.dem import DemError, DetectorErrorModel

pytestmark = pytest.mark.unit


def _model(
    *errors: DemError, detectors: int = 3, observables: int = 1
) -> DetectorErrorModel:
    return DetectorErrorModel(
        num_detectors=detectors, num_observables=observables, errors=errors
    )


def test_exact_text_for_a_hand_built_model() -> None:
    model = _model(
        DemError(probability=0.05, detectors=(0, 1), observables=(0,)),
        DemError(probability=0.125, detectors=(2,), observables=()),
    )
    assert model.to_stim_text() == (
        "error(0.05) D0 D1 L0\n"
        "error(0.125) D2\n"
        "detector D0\n"
        "detector D1\n"
        "detector D2\n"
        "logical_observable L0\n"
    )


def test_error_free_model_still_declares_its_shape() -> None:
    model = DetectorErrorModel(num_detectors=2, num_observables=1, errors=())
    assert model.to_stim_text() == ("detector D0\ndetector D1\nlogical_observable L0\n")


def test_round_trip_preserves_the_model() -> None:
    model = _model(
        DemError(probability=0.05, detectors=(0, 1), observables=(0,)),
        DemError(probability=1 / 3, detectors=(2,), observables=(0,)),
        detectors=3,
    )
    assert DetectorErrorModel.from_stim_text(model.to_stim_text()) == model


def test_round_trip_preserves_a_shape_no_error_touches() -> None:
    """The declaration lines are what make this lossless."""

    model = DetectorErrorModel(num_detectors=5, num_observables=2, errors=())
    restored = DetectorErrorModel.from_stim_text(model.to_stim_text())
    assert restored.num_detectors == 5
    assert restored.num_observables == 2


def test_round_trip_is_stable() -> None:
    model = _model(
        DemError(probability=1 / 3, detectors=(0,), observables=(0,)), detectors=1
    )
    once = model.to_stim_text()
    assert DetectorErrorModel.from_stim_text(once).to_stim_text() == once


def test_parses_coordinate_declarations() -> None:
    text = (
        "error(0.1) D0 D1 L0\n"
        "error(0.2) D2\n"
        "detector(1, 2) D0\n"
        "detector D1\n"
        "detector D2\n"
        "logical_observable L0\n"
    )
    model = DetectorErrorModel.from_stim_text(text)
    assert model.num_detectors == 3
    assert model.num_observables == 1
    assert model.num_errors == 2


def test_applies_a_shift_detectors_line_to_the_lines_after_it() -> None:
    """Stim reaches an absolute index by adding the shifts accumulated so far.

    The shift reaches both kinds of line: the declaration and the error
    mechanism below both name a detector relative to it, and the declaration
    set stays consecutive from zero either way.
    """

    text = (
        "detector D0\n"
        "detector D1\n"
        "shift_detectors 1\n"
        "detector D1\n"
        "error(0.1) D0\n"
    )
    model = DetectorErrorModel.from_stim_text(text)
    assert model.num_detectors == 3
    assert model.errors == (DemError(probability=0.1, detectors=(1,), observables=()),)


def test_shift_detectors_accumulates_rather_than_replacing() -> None:
    """A second shift adds to the first, which is how a repeated block unrolls."""

    text = (
        "detector D0\n"
        "shift_detectors 1\n"
        "detector D0\n"
        "shift_detectors 1\n"
        "detector D0\n"
    )
    model = DetectorErrorModel.from_stim_text(text)
    assert model.num_detectors == 3


def test_shift_detectors_discards_its_coordinates() -> None:
    """Coordinates carry geometry the model does not hold, as on a detector."""

    text = "detector(1, 0) D0\nshift_detectors(0, 1) 0\ndetector(1, 0) D1\n"
    model = DetectorErrorModel.from_stim_text(text)
    assert model.num_detectors == 2


def test_a_shift_detectors_line_shifts_detectors_and_not_observables() -> None:
    """The format has no observable shift, so an L target stays absolute.

    The shift moves the error's detector target from zero to one. A reader that
    moved the observable target too would name ``L1``, which the single
    declaration does not cover, so the model would be refused instead.
    """

    text = (
        "logical_observable L0\n"
        "detector D0\n"
        "detector D1\n"
        "detector D2\n"
        "detector D3\n"
        "shift_detectors 1\n"
        "detector D3\n"
        "error(0.1) D0 L0\n"
    )
    model = DetectorErrorModel.from_stim_text(text)
    assert model.num_detectors == 5
    assert model.num_observables == 1
    assert model.errors == (
        DemError(probability=0.1, detectors=(1,), observables=(0,)),
    )


@pytest.mark.parametrize(
    ("text", "refusal"),
    [
        ("shift_detectors\n", r"exactly one detector-index shift"),
        ("shift_detectors(0, 1)\n", r"exactly one detector-index shift"),
        ("shift_detectors(0, 1 2\n", r"coordinates must close in parentheses"),
        ("shift_detectors abc\n", r"non-negative integer shift"),
        ("shift_detectors -1\n", r"non-negative integer shift"),
    ],
)
def test_rejects_a_malformed_shift_detectors_line(text: str, refusal: str) -> None:
    """Each guard has its own refusal, so none may fall through to another."""

    with pytest.raises(ValueError, match=refusal):
        DetectorErrorModel.from_stim_text(text + "detector D0\n")


def test_infers_the_observable_count_when_no_declaration_states_it() -> None:
    """Stim declares the observable exactly when no error references it.

    This is the shape rule that differs between the two indices: a detector
    index outside the declared shape is still a fault, because Stim always
    declares every detector it emits.
    """

    text = "detector D0\nerror(0.1) D0 L0\n"
    model = DetectorErrorModel.from_stim_text(text)
    assert model.num_detectors == 1
    assert model.num_observables == 1

    wider = DetectorErrorModel.from_stim_text("detector D0\nerror(0.1) D0 L3\n")
    assert wider.num_observables == 4

    assert DetectorErrorModel.from_stim_text("detector D0\n").num_observables == 0


def test_a_declared_observable_still_governs_the_shape() -> None:
    """Inference is the fallback, not a widening of a declared shape."""

    with pytest.raises(ValueError, match="outside the model shape"):
        DetectorErrorModel.from_stim_text(
            "logical_observable L0\ndetector D0\nerror(0.1) D0 L1\n"
        )


def test_rejects_a_repeat_block() -> None:
    with pytest.raises(ValueError, match=r"repeat blocks are not supported"):
        DetectorErrorModel.from_stim_text("repeat 2 {\n error(0.1) D0\n}\n")


@pytest.mark.parametrize(
    ("text", "refusal"),
    [
        ("error 0.1\n", r"must state its probability in parentheses"),
        ("error(0.1 D0\n", r"must close its probability in parentheses"),
        ("error(abc) D0\n", r"must state a numeric probability"),
    ],
)
def test_rejects_a_malformed_error_line(text: str, refusal: str) -> None:
    """Each guard has its own refusal, so none may fall through to another."""

    with pytest.raises(ValueError, match=refusal):
        DetectorErrorModel.from_stim_text(text)


@pytest.mark.parametrize(
    ("text", "refusal"),
    [
        ("detector(1, 2 D0\n", r"coordinates must close in parentheses"),
        ("detector X0\n", r"must name a D index"),
        ("detector 0\n", r"must name a D index"),
        ("detector D0 D1\n", r"exactly one D index"),
    ],
)
def test_rejects_a_malformed_detector_declaration(text: str, refusal: str) -> None:
    """Each guard names its own fault, so none may fall through to another."""

    with pytest.raises(ValueError, match=refusal):
        DetectorErrorModel.from_stim_text(text)


def test_rejects_an_error_with_no_effect() -> None:
    with pytest.raises(ValueError, match="must flip at least one"):
        DetectorErrorModel.from_stim_text("error(0.1)\n")


@pytest.mark.parametrize("probability", ["1.5", "-0.1"])
def test_rejects_a_probability_outside_the_unit_interval(probability: str) -> None:
    """The probability is the only thing this text can be refused for.

    A text that declares no detector is refused by the model's "at least one
    detector" rule whatever the probability is, so it would pass with the range
    check deleted. Declaring the detector the error names leaves the range check
    as the sole possible ``ValueError``, on both sides of the interval.
    """

    with pytest.raises(ValueError, match="must be between zero and one"):
        DetectorErrorModel.from_stim_text(f"error({probability}) D0\ndetector D0\n")


def test_rejects_a_malformed_target() -> None:
    with pytest.raises(ValueError, match="D or L"):
        DetectorErrorModel.from_stim_text("error(0.1) X0\n")


def test_rejects_a_model_with_no_detectors() -> None:
    with pytest.raises(ValueError, match="at least one detector"):
        DetectorErrorModel.from_stim_text("logical_observable L0\n")


def test_rejects_a_bare_string_without_declarations() -> None:
    with pytest.raises(ValueError, match="at least one detector"):
        DetectorErrorModel.from_stim_text("error(0.1) D0\n")


def test_parses_the_error_content_not_only_the_shape() -> None:
    """A parser that dropped a target would satisfy a shape-only check."""

    text = (
        "error(0.1) D0 D2 L0 L1\n"
        "error(0.2) D1\n"
        "detector D0\n"
        "detector D1\n"
        "detector D2\n"
        "logical_observable L0\n"
        "logical_observable L1\n"
    )
    model = DetectorErrorModel.from_stim_text(text)
    assert model.errors == (
        DemError(probability=0.1, detectors=(0, 2), observables=(0, 1)),
        DemError(probability=0.2, detectors=(1,), observables=()),
    )
    assert model.detector_error_matrix().tolist() == [[1, 0], [0, 1], [1, 0]]
    assert model.observables_flips_matrix().tolist() == [[1, 0], [1, 0]]


def test_shape_comes_from_the_declarations_not_the_largest_index() -> None:
    """The last detectors are declared but no error mentions them."""

    text = "error(0.1) D0\ndetector D0\ndetector D1\ndetector D2\n"
    model = DetectorErrorModel.from_stim_text(text)
    assert model.num_detectors == 3
    assert model.num_observables == 0
    assert model.errors == (DemError(probability=0.1, detectors=(0,), observables=()),)


def test_parses_coordinate_declarations_by_ignoring_the_coordinates() -> None:
    """The declaration's index is the target, never the coordinate."""

    text = "error(0.1) D0\ndetector(7, 8) D0\n"
    model = DetectorErrorModel.from_stim_text(text)
    assert model.num_detectors == 1
    assert model.errors == (DemError(probability=0.1, detectors=(0,), observables=()),)


def test_rejects_an_unrecognized_instruction() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        DetectorErrorModel.from_stim_text("noise 0.1\n")


def test_rejects_coordinates_on_a_logical_observable() -> None:
    """Only a detector declaration carries coordinates."""

    with pytest.raises(ValueError, match="exactly one L index"):
        DetectorErrorModel.from_stim_text("logical_observable(7, 8) L0\n")


def test_rejects_declarations_that_skip_an_index() -> None:
    """A hole in the declarations would silently misstate the shape."""

    with pytest.raises(ValueError, match="consecutive"):
        DetectorErrorModel.from_stim_text("detector D1\n")


def test_rejects_an_error_outside_the_declared_shape() -> None:
    with pytest.raises(ValueError, match="outside the model shape"):
        DetectorErrorModel.from_stim_text("error(0.1) D5\ndetector D0\n")


def test_from_text_orders_errors_that_tie_on_detectors_and_probability() -> None:
    """The canonical key separates errors by their observables.

    Two errors can share a probability and a detector signature, which makes
    the observable component of the sort key the only thing that orders them.
    ``sorted`` is stable, so a key that drops that component leaves the two in
    whatever order the text listed them.
    """

    text = (
        "error(0.25) D1 L1\n"
        "error(0.25) D1 L0\n"
        "error(0.5) D0\n"
        "detector D0\n"
        "detector D1\n"
        "logical_observable L0\n"
        "logical_observable L1\n"
    )
    model = DetectorErrorModel.from_stim_text(text)
    assert model.errors == (
        DemError(probability=0.5, detectors=(0,), observables=()),
        DemError(probability=0.25, detectors=(1,), observables=(0,)),
        DemError(probability=0.25, detectors=(1,), observables=(1,)),
    )


def test_a_decomposition_separator_partitions_without_changing_the_signature() -> None:
    """A shot flips the symmetric difference of the line's targets.

    ``stim`` marks how a composite mechanism decomposes with ``^``. The groups
    are a decoder's business, so the signature is the same however the line
    groups them, and this reader states that signature.
    """

    declarations = "detector D0\ndetector D1\ndetector D2\n"
    expected = DemError(probability=0.1, detectors=(0, 1, 2), observables=())

    for line in ("error(0.1) D0 ^ D1 ^ D2", "error(0.1) D0 D1 ^ D2"):
        model = DetectorErrorModel.from_stim_text(declarations + line + "\n")
        assert model.errors == (expected,)


def test_a_repeated_target_cancels_within_one_group_as_well() -> None:
    """The separator is not the only thing that cancels: a repeat does too.

    ``stim`` reports no detector flip at all for ``error(0.1) D0 D0``, which is
    the same rule as for a target that appears on both sides of a separator.
    """

    declarations = "detector D0\ndetector D1\n"

    model = DetectorErrorModel.from_stim_text(declarations + "error(0.1) D0 D1 D1\n")
    assert model.errors == (DemError(probability=0.1, detectors=(0,), observables=()),)


def test_a_cancelling_decomposition_line_is_refused() -> None:
    """A mechanism that flips nothing cannot be stated, so it is not silently dropped."""

    declarations = "detector D0\ndetector D1\n"
    with pytest.raises(ValueError, match="must flip at least one"):
        DetectorErrorModel.from_stim_text(declarations + "error(0.1) D0 ^ D0\n")


def test_a_separator_can_move_a_detector_flip_onto_an_observable_alone() -> None:
    """Groups may cancel the detectors while leaving the observable."""

    text = "detector D0\n" "logical_observable L0\n" "error(0.1) D0 L0 ^ D0\n"
    model = DetectorErrorModel.from_stim_text(text)
    assert model.errors == (DemError(probability=0.1, detectors=(), observables=(0,)),)
    assert model.detector_rates().tolist() == [0.0]
    assert model.observable_rates().tolist() == pytest.approx([0.1])


def test_the_suggestion_reading_splits_each_group_into_its_own_mechanism() -> None:
    """The other reading of the same line, and the parameter is the only difference.

    Upstream's ``dem_from_stim_text(dem_text, use_decomp_suggestions=True)``
    expands a decomposed instruction into one column per component. This is that
    reading: the groups stop being a hint about how the mechanism decomposes and
    become mechanisms, each stated at the parent instruction's probability.
    """

    text = "detector D0\ndetector D1\ndetector D2\nerror(0.1) D0 D1 ^ D2\n"

    combined = DetectorErrorModel.from_stim_text(text)
    assert combined.errors == (DemError(probability=0.1, detectors=(0, 1, 2)),)

    suggested = DetectorErrorModel.from_stim_text(text, use_decomp_suggestions=True)
    assert suggested.errors == (
        DemError(probability=0.1, detectors=(0, 1), observables=()),
        DemError(probability=0.1, detectors=(2,), observables=()),
    )
    assert suggested.num_detectors == combined.num_detectors
    assert suggested.num_observables == combined.num_observables


def test_every_component_inherits_the_probability_of_the_line() -> None:
    """One probability on the line, three mechanisms out of it."""

    text = "detector D0\ndetector D1\ndetector D2\nerror(0.25) D0 ^ D1 ^ D2\n"
    model = DetectorErrorModel.from_stim_text(text, use_decomp_suggestions=True)
    assert model.errors == (
        DemError(probability=0.25, detectors=(0,), observables=()),
        DemError(probability=0.25, detectors=(1,), observables=()),
        DemError(probability=0.25, detectors=(2,), observables=()),
    )


def test_the_suggestion_reading_keeps_groups_apart_where_the_default_cancels() -> None:
    """The sharpest difference between the two readings of one line.

    The line names ``D0`` twice, so the default reading cancels the detector and
    leaves the observable flipping alone. The suggestion reading keeps the two
    groups apart, so ``D0`` flips in one of its two mechanisms and the detector
    rate stops being zero.
    """

    text = "detector D0\nlogical_observable L0\nerror(0.1) D0 L0 ^ D0\n"

    combined = DetectorErrorModel.from_stim_text(text)
    assert combined.errors == (
        DemError(probability=0.1, detectors=(), observables=(0,)),
    )
    assert combined.detector_rates().tolist() == [0.0]

    suggested = DetectorErrorModel.from_stim_text(text, use_decomp_suggestions=True)
    assert suggested.errors == (
        DemError(probability=0.1, detectors=(0,), observables=()),
        DemError(probability=0.1, detectors=(0,), observables=(0,)),
    )
    assert suggested.observable_rates().tolist() == pytest.approx([0.1])
    assert suggested.detector_rates().tolist() == pytest.approx([0.18])


def test_the_suggestion_reading_is_not_an_equivalent_statement() -> None:
    """Lossy relative to the line, and lossless relative to itself.

    The decomposition a separator suggests is a set of independent mechanisms,
    and that is what this reading returns: a model of its own, which prints and
    re-reads exactly. What it does not do is reproduce the one mechanism the
    line states, at the line's probability, and no amount of care in the reader
    can make it, because the two are different distributions. The default
    reading is the one that states the line.
    """

    text = "detector D0\ndetector D1\nerror(0.1) D0 ^ D1\n"
    suggested = DetectorErrorModel.from_stim_text(text, use_decomp_suggestions=True)

    # Not the line's model, and not the default reading of it either.
    assert suggested != DetectorErrorModel.from_stim_text(text)
    assert suggested.num_errors == 2

    # But a model in its own right: its own text returns it unchanged.
    printed = suggested.to_stim_text()
    assert printed.count("error(") == 2
    assert DetectorErrorModel.from_stim_text(printed) == suggested


def test_a_separator_free_line_reads_the_same_either_way() -> None:
    """The parameter decides about separators and about nothing else."""

    for line in ("error(0.1) D0 D1", "error(0.1) D0", "error(0.1) D0 D0 D1"):
        text = f"detector D0\ndetector D1\n{line}\n"
        assert (
            DetectorErrorModel.from_stim_text(text).errors
            == DetectorErrorModel.from_stim_text(
                text, use_decomp_suggestions=True
            ).errors
        )


def test_the_suggestion_reading_refuses_a_component_that_flips_nothing() -> None:
    """A component with no targets has no mechanism to become.

    ``stim`` accepts the text, and the default reading states it as the
    symmetric difference ``D1``. Expanding it would have to emit a mechanism
    that flips nothing, which no model can hold, so the line is refused and the
    combined reading is named as the route that states it.
    """

    text = "detector D0\ndetector D1\nerror(0.1) D0 D0 ^ D1\n"

    combined = DetectorErrorModel.from_stim_text(text)
    assert combined.errors == (DemError(probability=0.1, detectors=(1,)),)

    with pytest.raises(ValueError, match="cannot be a mechanism of its own"):
        DetectorErrorModel.from_stim_text(text, use_decomp_suggestions=True)


def test_the_suggestion_reading_splits_a_duplicated_component_into_two() -> None:
    """The groups are not deduplicated: two groups mean two mechanisms.

    The default reading of this line is refused outright, because the two groups
    cancel and the line then flips nothing at all.
    """

    text = "detector D0\nerror(0.1) D0 ^ D0\n"

    with pytest.raises(ValueError, match="must flip at least one"):
        DetectorErrorModel.from_stim_text(text)

    model = DetectorErrorModel.from_stim_text(text, use_decomp_suggestions=True)
    assert model.errors == (
        DemError(probability=0.1, detectors=(0,), observables=()),
        DemError(probability=0.1, detectors=(0,), observables=()),
    )


@pytest.mark.parametrize("use_decomp_suggestions", [False, True])
@pytest.mark.parametrize(
    ("targets", "refusal"),
    [
        ("^ D0", r"between two groups of targets"),
        ("D0 ^", r"between two groups of targets"),
        ("D0 ^ ^ D1", r"between two groups of targets"),
        ("^", r"between two groups of targets"),
        ("D0^D1", r"separated by spacing"),
        ("D0 ^D1", r"separated by spacing"),
    ],
)
def test_rejects_a_malformed_separator(
    targets: str, refusal: str, use_decomp_suggestions: bool
) -> None:
    """The independent reader refuses each of these too, so the shapes agree.

    A malformed separator is malformed under either reading of a well-formed
    one, so the parameter is crossed with the cases rather than left out.
    """

    with pytest.raises(ValueError, match=refusal):
        DetectorErrorModel.from_stim_text(
            f"detector D0\ndetector D1\nerror(0.1) {targets}\n",
            use_decomp_suggestions=use_decomp_suggestions,
        )
