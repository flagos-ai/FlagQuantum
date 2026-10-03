"""Unit coverage for the phenomenological noise record.

The record has two levels: a rate per fault family and, optionally, a rate per
element. The first level is four scalars and the second is the same four as
vectors, which is upstream `CssNoise`'s `px`/`pz`/`py`/`pm` and its
`px_per_qubit`/`pz_per_qubit`/`py_per_qubit`/`pm_per_check`. What is pinned here
is the record itself -- which fields exist, what they default to, what they
refuse, and how a vector is resolved against a count. What the construction
routes and the sampler then read out of it is pinned in
``tests/qec/test_dem_rate_overrides.py``.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from flagquantum.qec.noise import PhenomenologicalNoise, RepetitionNoiseProfile

pytestmark = pytest.mark.unit

#: The four families, as ``(scalar, vector, count argument, element name)``.
#: The element name is the one the refusal uses, which is why it is "data qubit"
#: for the three data families and "check" for the measurement one.
_FAMILIES = (
    ("data_flip", "data_flip_per_qubit", "num_qubits", "data qubit"),
    ("phase_flip", "phase_flip_per_qubit", "num_qubits", "data qubit"),
    ("both_flip", "both_flip_per_qubit", "num_qubits", "data qubit"),
    ("measurement_flip", "measurement_flip_per_check", "num_checks", "check"),
)


def test_defaults_are_noiseless() -> None:
    noise = PhenomenologicalNoise()
    assert noise.data_flip == 0.0
    assert noise.measurement_flip == 0.0


def test_fields_round_trip() -> None:
    noise = PhenomenologicalNoise(data_flip=0.05, measurement_flip=0.02)
    assert noise.data_flip == 0.05
    assert noise.measurement_flip == 0.02


def test_record_is_frozen() -> None:
    noise = PhenomenologicalNoise(data_flip=0.05)
    with pytest.raises(dataclasses.FrozenInstanceError):
        noise.data_flip = 0.1  # type: ignore[misc]


@pytest.mark.parametrize("field", ["data_flip", "measurement_flip"])
def test_probability_bounds_are_enforced(field: str) -> None:
    with pytest.raises(ValueError):
        PhenomenologicalNoise(**{field: -0.01})
    with pytest.raises(ValueError):
        PhenomenologicalNoise(**{field: 1.01})


@pytest.mark.parametrize("field", ["data_flip", "measurement_flip"])
def test_bool_is_rejected_as_a_probability(field: str) -> None:
    with pytest.raises(TypeError):
        PhenomenologicalNoise(**{field: True})


def test_boundary_values_are_accepted() -> None:
    assert (
        PhenomenologicalNoise(data_flip=0.0, measurement_flip=1.0).measurement_flip
        == 1.0
    )


def test_frozen_profile_is_unchanged() -> None:
    """The additive rule: the frozen profile keeps its fields and defaults."""

    profile = RepetitionNoiseProfile()
    assert profile.data_bit_flip_probability == 0.0
    assert profile.syndrome_readout_error_probability == 0.0
    assert profile.final_readout_error_probability == 0.0


# --------------------------------------------------------------------------
# the two levels of the record
# --------------------------------------------------------------------------


def test_every_family_has_a_scalar_and_a_vector_and_each_defaults_to_neutral() -> None:
    """Four families, four scalars, four vectors, and the empty state of each.

    A scalar defaults to zero, which is no fault at all, and a vector defaults to
    empty, which is the unstated state rather than a vector of no-ops: the scalar
    is what applies then.
    """

    noise = PhenomenologicalNoise()

    for scalar, vector, _, _ in _FAMILIES:
        assert getattr(noise, scalar) == 0.0
        assert getattr(noise, vector) == ()


def test_the_four_scalars_are_all_read_written() -> None:
    """Upstream states px, pz, py and pm; each is a field here in its own right.

    The mapping is the published one and is asserted rather than assumed: the X
    fault is `data_flip`, the Z fault `phase_flip`, the Y fault `both_flip` and
    the measurement fault `measurement_flip`.
    """

    noise = PhenomenologicalNoise(
        data_flip=0.01, phase_flip=0.02, both_flip=0.03, measurement_flip=0.04
    )

    assert noise.data_flip == 0.01
    assert noise.phase_flip == 0.02
    assert noise.both_flip == 0.03
    assert noise.measurement_flip == 0.04


def test_the_four_vectors_are_all_read_written() -> None:
    noise = PhenomenologicalNoise(
        data_flip_per_qubit=(0.01, 0.02),
        phase_flip_per_qubit=(0.03, 0.04),
        both_flip_per_qubit=(0.05, 0.06),
        measurement_flip_per_check=(0.07, 0.08),
    )

    assert noise.data_flip_per_qubit == (0.01, 0.02)
    assert noise.phase_flip_per_qubit == (0.03, 0.04)
    assert noise.both_flip_per_qubit == (0.05, 0.06)
    assert noise.measurement_flip_per_check == (0.07, 0.08)


@pytest.mark.parametrize("vector", [field for _, field, _, _ in _FAMILIES])
def test_a_vector_field_is_frozen_too(vector: str) -> None:
    noise = PhenomenologicalNoise(**{vector: (0.01, 0.02)})
    with pytest.raises(dataclasses.FrozenInstanceError):
        noise.__setattr__(vector, (0.03,))


def test_two_records_with_the_same_rates_are_equal_and_hash_alike() -> None:
    """The record is a value, so it can be a default or a dictionary key."""

    first = PhenomenologicalNoise(data_flip=0.05, data_flip_per_qubit=(0.01, 0.02))
    second = PhenomenologicalNoise(data_flip=0.05, data_flip_per_qubit=(0.01, 0.02))
    third = PhenomenologicalNoise(data_flip=0.05, data_flip_per_qubit=(0.02, 0.01))

    assert first == second
    assert hash(first) == hash(second)
    assert first != third
    assert len({first, second, third}) == 2


def test_a_vector_is_stored_as_a_tuple_whatever_iterable_states_it() -> None:
    """The record is frozen, so the iterable it is handed is read once on entry."""

    from_list = PhenomenologicalNoise(data_flip_per_qubit=[0.01, 0.02])
    from_generator = PhenomenologicalNoise(
        data_flip_per_qubit=(value for value in (0.01, 0.02))
    )
    from_scalars = PhenomenologicalNoise(data_flip_per_qubit=(0.01, 0.02))

    assert from_list.data_flip_per_qubit == (0.01, 0.02)
    assert from_generator.data_flip_per_qubit == (0.01, 0.02)
    assert from_list == from_generator == from_scalars


# --------------------------------------------------------------------------
# resolving a vector against a count
# --------------------------------------------------------------------------


def test_an_empty_vector_resolves_to_the_scalar_for_every_element() -> None:
    """The stated state of "no per-element override" is the scalar everywhere."""

    noise = PhenomenologicalNoise(data_flip=0.05, measurement_flip=0.02)

    assert noise.data_flip_rates(num_qubits=0) == ()
    assert noise.data_flip_rates(num_qubits=3) == (0.05, 0.05, 0.05)
    assert noise.measurement_flip_rates(num_checks=2) == (0.02, 0.02)


def test_an_empty_vector_resolves_at_every_count_including_zero() -> None:
    """Empty is the unstated state, so a count of zero is a shape rather than a mismatch."""

    noise = PhenomenologicalNoise(data_flip=0.05)

    for count in range(5):
        assert noise.data_flip_rates(num_qubits=count) == (0.05,) * count


def test_a_stated_vector_resolves_to_itself_and_ignores_the_scalar() -> None:
    """The override is wholesale: the scalar reaches no element at all.

    If the scalar reached an element the result would mix a `0.5` in, and a
    caller who stated three rates would silently be read as stating a fourth.
    """

    noise = PhenomenologicalNoise(data_flip=0.5, data_flip_per_qubit=(0.1, 0.2, 0.3))

    assert noise.data_flip_rates(num_qubits=3) == (0.1, 0.2, 0.3)
    assert 0.5 not in noise.data_flip_rates(num_qubits=3)


@pytest.mark.parametrize(("scalar", "vector", "argument", "element"), _FAMILIES)
def test_every_family_resolves_its_own_vector(
    scalar: str, vector: str, argument: str, element: str
) -> None:
    """Each family has its own accessor, and each reads its own vector.

    A family whose accessor read another family's vector would answer with a rate
    the caller never stated. Each is resolved against a distinct vector of its
    own, and the four results are compared to each other rather than to a
    restatement of themselves.
    """

    rates = (0.11, 0.22, 0.33)
    noise = PhenomenologicalNoise(**{vector: rates})  # type: ignore[arg-type]
    accessor = f"{scalar}_rates"

    assert getattr(noise, accessor)(**{argument: 3}) == rates
    with pytest.raises(
        ValueError, match=rf"{vector} states 3 rate\(s\) but the code declares 0"
    ):
        getattr(noise, accessor)(**{argument: 0})
    # The count is this accessor's own argument and is not optional: the other
    # family's count name is not accepted here.
    with pytest.raises(TypeError):
        getattr(noise, accessor)()
    assert element in {"data qubit", "check"}


def test_the_accessor_is_the_only_reader_of_the_override_rule() -> None:
    """Both levels are readable through one call per family, and they agree.

    The scalar is what the accessor answers with when no vector is stated and the
    vector is what it answers with when one is, so a caller who wants the
    effective rate of an element never has to test the vector itself.
    """

    explicit = PhenomenologicalNoise(data_flip=0.05)
    asserted = PhenomenologicalNoise(data_flip=0.0, data_flip_per_qubit=(0.05,) * 3)

    assert explicit.data_flip_rates(num_qubits=3) == asserted.data_flip_rates(
        num_qubits=3
    )


# --------------------------------------------------------------------------
# what the record refuses
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("scalar", "vector", "argument", "element"), _FAMILIES)
def test_a_vector_that_does_not_name_every_element_is_refused(
    scalar: str, vector: str, argument: str, element: str
) -> None:
    """A short vector would put one element's rate on another, so it is refused.

    The message names the vector, the number of rates it states and the number of
    elements the reader declares, so a caller can tell which of the two counts is
    the wrong one rather than only that they differ. It also states the
    consequence, because that is the whole reason a partial read is not offered:
    the elements a short vector left out would silently take their neighbours'
    rates.
    """

    count = 3
    noise = PhenomenologicalNoise(**{vector: (0.1, 0.2)})  # type: ignore[arg-type]

    with pytest.raises(ValueError) as caught:
        getattr(noise, f"{scalar}_rates")(**{argument: count})

    message = str(caught.value)
    assert message.startswith(f"{vector} states 2 rate(s)")
    assert f"the code declares {count} {element}(s)" in message
    assert f"one {element}'s rate would be read as another's" in message


def test_a_vector_of_the_right_length_resolves_at_that_length() -> None:
    """The refusal is about the length, not about stating a vector at all."""

    noise = PhenomenologicalNoise(data_flip_per_qubit=(0.1, 0.2, 0.3))

    assert noise.data_flip_rates(num_qubits=3) == (0.1, 0.2, 0.3)
    with pytest.raises(ValueError, match=r"states 3 rate\(s\) but the code declares 4"):
        noise.data_flip_rates(num_qubits=4)


@pytest.mark.parametrize(
    ("method", "argument"),
    (("data_flip_rates", "num_qubits"), ("measurement_flip_rates", "num_checks")),
)
@pytest.mark.parametrize("count", (True, False, 3.0, "3", None), ids=repr)
def test_a_count_that_is_not_an_integer_is_refused(
    method: str, argument: str, count: object
) -> None:
    """The count is a code's own length, and a bool is not one even though it is an int."""

    noise = PhenomenologicalNoise()
    with pytest.raises(TypeError, match="must be an integer"):
        getattr(noise, method)(**{argument: count})


@pytest.mark.parametrize(
    ("method", "argument"),
    (("data_flip_rates", "num_qubits"), ("measurement_flip_rates", "num_checks")),
)
def test_a_negative_count_is_refused(method: str, argument: str) -> None:
    """A negative count is a shape that cannot be enumerated, not a mismatch."""

    noise = PhenomenologicalNoise()
    with pytest.raises(ValueError, match="cannot be negative"):
        getattr(noise, method)(**{argument: -1})


@pytest.mark.parametrize(
    ("vector", "value", "index"),
    (
        ("data_flip_per_qubit", 1.5, 0),
        ("data_flip_per_qubit", 2.0, 0),
        ("data_flip_per_qubit", -0.0 - 0.1, 1),
        ("phase_flip_per_qubit", math.inf, 1),
        ("both_flip_per_qubit", math.nan, 1),
        ("measurement_flip_per_check", 1.0000001, 2),
    ),
)
def test_an_element_outside_zero_and_one_is_refused_by_index(
    vector: str, value: float, index: int
) -> None:
    """The offending index is named, because a vector is read as a list.

    ``nan`` is refused here rather than accepted as an unorderable value: a rate
    that is not a number cannot be read as a probability, so the comparison is
    written to catch it instead of passing it through.
    """

    rates = [0.1, 0.2, 0.3]
    rates[index] = value

    with pytest.raises(
        ValueError, match=rf"{vector}\[{index}\] must be between zero and one"
    ):
        PhenomenologicalNoise(**{vector: tuple(rates)})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("vector", "value", "index"),
    (
        ("data_flip_per_qubit", True, 1),
        ("data_flip_per_qubit", None, 1),
        ("both_flip_per_qubit", "0.1", 0),
        ("measurement_flip_per_check", 1 + 0j, 2),
    ),
)
def test_an_element_that_is_not_a_real_probability_is_refused_by_index(
    vector: str, value: object, index: int
) -> None:
    """A bool is an int and a string is iterable, so both are refused by hand."""

    rates: list[object] = [0.1, 0.2, 0.3]
    rates[index] = value

    with pytest.raises(
        TypeError, match=rf"{vector}\[{index}\] must be a real probability"
    ):
        PhenomenologicalNoise(**{vector: tuple(rates)})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "kind"),
    (
        (0.1, "a scalar"),
        ("0.1", "a string"),
        (b"\x00\x01", "bytes"),
        (None, "none"),
    ),
    ids=("float", "string", "bytes", "none"),
)
def test_a_vector_must_be_an_iterable_and_not_a_scalar(
    value: object, kind: str
) -> None:
    """A rate belongs in the scalar field, and a string is not a list of rates.

    Without the check a string would be iterated into its characters and the
    refusal would arrive from the element check instead of naming the field.
    """

    with pytest.raises(TypeError, match="must be an iterable of probabilities"):
        PhenomenologicalNoise(data_flip_per_qubit=value)  # type: ignore[arg-type]


def test_the_two_levels_are_validated_in_the_same_pass() -> None:
    """A record states both levels at once, and either one can refuse it.

    Validation happens on construction rather than at read time, so a record that
    exists is a record whose every element was already read as a probability.
    """

    with pytest.raises(TypeError):
        PhenomenologicalNoise(data_flip=True)
    with pytest.raises(TypeError):
        PhenomenologicalNoise(data_flip_per_qubit=(True,))
    with pytest.raises(ValueError):
        PhenomenologicalNoise(data_flip=1.5, data_flip_per_qubit=(0.1,))
