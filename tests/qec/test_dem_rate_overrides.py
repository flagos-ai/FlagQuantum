"""Unit coverage for per-element noise rates.

A phenomenological rate record has two levels: a rate per fault family and,
optionally, a rate per element. The second level is upstream `CssNoise`'s
`*_per_qubit` and `pm_per_check` vectors, and what these tests pin is that the
override is read the way it is stated rather than the way it is convenient.

Three properties carry most of the weight. A stated vector replaces its scalar
for every element instead of mixing with it, which is why a vector has to name
every data qubit or every check and a short one is refused rather than partially
applied. An element whose effective rate is zero names a location that cannot
fire, so it enumerates no mechanism and costs nothing to sample. And the
per-check vector is indexed by matrix row -- Z-type checks first -- while a code
declares its checks in its own order, so the two orders are related by one
translation and the tests pin that translation with a code whose two orders
differ.

The routes are compared to each other rather than to a restatement of
themselves: the circuit route, the matrix route and the sampler all resolve an
element through the same accessor, and the tests read that back as the same
rates rather than as the same prose.
"""

from __future__ import annotations

import pytest

from flagquantum.qec import (
    DetectorErrorModel,
    PhenomenologicalNoise,
    RepetitionCode,
    RotatedSurfaceCode,
    SteaneCode,
    build_memory_circuit,
    css_code_matrices,
)
from flagquantum.qec.dem_construction import (
    _check_rate_order,
    _code_matrix_entries,
    _mechanisms,
)

pytestmark = pytest.mark.unit

_ROUNDS = 2


def _entries(code, noise: PhenomenologicalNoise, *, num_rounds: int = 1):
    """Return the matrix route's mechanism entries for ``code``."""

    return _code_matrix_entries(css_code_matrices(code), noise, num_rounds=num_rounds)[
        2
    ]


def _sources(matrix, qubit: int) -> tuple[int, ...]:
    """Return the rows of ``matrix`` that select ``qubit``, in ascending order."""

    assert matrix is not None
    return tuple(int(row) for row in matrix[:, qubit].nonzero().flatten())


def _detectors_of_an_x_fault(code, qubit: int) -> tuple[int, ...]:
    """Return the detector band an X fault on ``qubit`` reaches, one round in."""

    matrices = css_code_matrices(code)
    return _sources(matrices.hz, qubit)


def _detectors_of_a_z_fault(code, qubit: int) -> tuple[int, ...]:
    """Return the detector band a Z fault on ``qubit`` reaches, one round in.

    The X-type checks are declared after the Z-type ones in the matrix, so their
    detector indices are offset by the number of Z-type rows.
    """

    matrices = css_code_matrices(code)
    offset = 0 if matrices.hx is None else matrices.hz.shape[0]
    return tuple(offset + row for row in _sources(matrices.hx, qubit))


# --------------------------------------------------------------------------
# the override rule
# --------------------------------------------------------------------------


def test_a_stated_vector_replaces_the_scalar_for_every_element() -> None:
    """Every element is read from the vector, and the scalar reaches none of them.

    The scalar here is a rate the model must not contain: if any mechanism came
    back at ``0.5`` the vector would have been mixed with the scalar rather than
    replacing it.
    """

    noise = PhenomenologicalNoise(data_flip=0.5, data_flip_per_qubit=(0.1, 0.2, 0.3))
    entries = _entries(RepetitionCode(distance=3), noise)

    assert sorted(entry[0] for entry in entries) == [0.1, 0.2, 0.3]
    assert 0.5 not in {entry[0] for entry in entries}


def test_the_scalar_applies_to_every_element_when_no_vector_is_stated() -> None:
    """An empty vector is the unstated state, not a vector of no-ops."""

    stated = PhenomenologicalNoise(data_flip=0.05)
    explicit = PhenomenologicalNoise(data_flip=0.05, data_flip_per_qubit=())
    code = RepetitionCode(distance=3)

    assert stated.data_flip_rates(num_qubits=3) == (0.05, 0.05, 0.05)
    assert _entries(code, stated) == _entries(code, explicit)


def test_a_zero_element_states_a_location_that_cannot_fire() -> None:
    """One quiet qubit between two noisy ones contributes no mechanism.

    The circuit route states which wire each mechanism belongs to, so the quiet
    element is identified rather than only counted, and it holds for every round
    rather than for the first one.
    """

    memory = build_memory_circuit(RepetitionCode(distance=3), rounds=_ROUNDS)
    noise = PhenomenologicalNoise(data_flip=0.5, data_flip_per_qubit=(0.0, 0.2, 0.0))
    mechanisms = _mechanisms(memory, noise)

    assert [mechanism.wire for mechanism in mechanisms] == [1] * _ROUNDS
    assert {mechanism.probability for mechanism in mechanisms} == {0.2}
    assert {mechanism.round_index for mechanism in mechanisms} == set(range(_ROUNDS))


def test_a_family_whose_elements_are_all_zero_contributes_nothing() -> None:
    """A vector of zeros silences a family even when its scalar is loud."""

    noise = PhenomenologicalNoise(
        data_flip=0.0, phase_flip=0.5, phase_flip_per_qubit=(0.0, 0.0, 0.0)
    )
    model = DetectorErrorModel.from_code(RepetitionCode(distance=3), noise=noise)

    assert model.num_errors == 0
    assert model.detector_rates().tolist() == [0.0, 0.0]


def test_a_family_the_code_has_no_block_for_contributes_nothing() -> None:
    """A stated vector for a block a code does not have is still no mechanism.

    The repetition code declares no X-type check, so its ``hx`` block is empty
    and a Z fault reaches nothing. The family is still enumerated once per
    element, and every one of those entries flips nothing, so the model drops all
    of them: the rate is read and the location simply reaches nothing.
    """

    code = RepetitionCode(distance=3)
    noise = PhenomenologicalNoise(phase_flip_per_qubit=(0.1, 0.2, 0.3))
    entries = _entries(code, noise)

    assert len(entries) == len(code.data_qubits)
    assert all(entry[1] == () and entry[2] == () for entry in entries)
    assert DetectorErrorModel.from_code(code, noise=noise).num_errors == 0


def test_the_model_is_the_union_of_the_one_hot_models_its_vector_states() -> None:
    """A vector is read element by element, not at its first or last entry.

    Each qubit's contribution is read on its own by silencing the others, and the
    vector's model must be exactly the union of those. A vector read at one entry
    for every element would agree with one of them and disagree with the rest.
    """

    code = RepetitionCode(distance=3)
    matrices = css_code_matrices(code)

    def one_hot(index: int, rate: float) -> DetectorErrorModel:
        vector = [0.0, 0.0, 0.0]
        vector[index] = rate
        return DetectorErrorModel.from_code_matrices(
            matrices,
            noise=PhenomenologicalNoise(
                data_flip=0.5, data_flip_per_qubit=tuple(vector)
            ),
        )

    model = DetectorErrorModel.from_code_matrices(
        matrices,
        noise=PhenomenologicalNoise(data_flip=0.5, data_flip_per_qubit=(0.1, 0.2, 0.3)),
    )
    union = sorted(
        (
            error.probability,
            tuple(error.detectors),
            tuple(error.observables),
        )
        for part in (one_hot(0, 0.1), one_hot(1, 0.2), one_hot(2, 0.3))
        for error in part.errors
    )

    assert (
        sorted(
            (error.probability, tuple(error.detectors), tuple(error.observables))
            for error in model.errors
        )
        == union
    )
    assert len(model.errors) == 3


# --------------------------------------------------------------------------
# every family reaches its own block
# --------------------------------------------------------------------------


def test_the_x_vector_reaches_the_z_type_checks() -> None:
    """The X fault family's vector is read against ``hz`` and ``lz``."""

    code = RotatedSurfaceCode(distance=3)
    matrices = css_code_matrices(code)
    vector = tuple(0.001 * (qubit + 1) for qubit in range(len(code.data_qubits)))
    entries = _entries(
        code, PhenomenologicalNoise(data_flip_per_qubit=vector), num_rounds=1
    )

    assert sorted(entries) == sorted(
        (rate, _detectors_of_an_x_fault(code, qubit), _sources(matrices.lz, qubit))
        for qubit, rate in enumerate(vector)
    )


def test_the_z_vector_reaches_the_x_type_checks() -> None:
    """The Z fault family's vector is read against ``hx`` and ``lx``."""

    code = RotatedSurfaceCode(distance=3)
    matrices = css_code_matrices(code)
    vector = tuple(0.001 * (qubit + 1) for qubit in range(len(code.data_qubits)))
    entries = _entries(
        code, PhenomenologicalNoise(phase_flip_per_qubit=vector), num_rounds=1
    )

    assert sorted(entries) == sorted(
        (
            rate,
            _detectors_of_a_z_fault(code, qubit),
            tuple(
                matrices.num_z_logicals + row for row in _sources(matrices.lx, qubit)
            ),
        )
        for qubit, rate in enumerate(vector)
    )


def test_the_y_vector_reaches_both_blocks() -> None:
    """A Y fault is the X fault and the Z fault at once, so it reaches both.

    Its detectors are the union of the two bands and its observables the union of
    the two logical blocks, which is what makes the Y rate a separate family
    rather than a second reading of either one.
    """

    code = RotatedSurfaceCode(distance=3)
    matrices = css_code_matrices(code)
    vector = tuple(0.001 * (qubit + 1) for qubit in range(len(code.data_qubits)))
    entries = _entries(
        code, PhenomenologicalNoise(both_flip_per_qubit=vector), num_rounds=1
    )

    assert sorted(entries) == sorted(
        (
            rate,
            tuple(
                sorted(
                    set(_detectors_of_an_x_fault(code, qubit))
                    | set(_detectors_of_a_z_fault(code, qubit))
                )
            ),
            tuple(
                sorted(
                    set(_sources(matrices.lz, qubit))
                    | set(
                        matrices.num_z_logicals + row
                        for row in _sources(matrices.lx, qubit)
                    )
                )
            ),
        )
        for qubit, rate in enumerate(vector)
    )


# --------------------------------------------------------------------------
# the per-check vector's order
# --------------------------------------------------------------------------


def test_the_per_check_vector_is_indexed_in_the_matrix_row_order() -> None:
    """Detector ``r`` carries the vector's ``r``-th element, not its own position.

    The rotated surface code declares its checks in lattice order, which
    interleaves the two types, so the declaration order and the matrix row order
    are different permutations. The rates are chosen distinct so that the
    difference is observable: read in the declaration order instead, at least one
    detector would carry another check's rate.
    """

    code = RotatedSurfaceCode(distance=3)
    matrices = css_code_matrices(code)
    order = _check_rate_order(code.checks)
    rates = tuple(0.001 * (row + 1) for row in range(matrices.num_checks))
    model = DetectorErrorModel.from_code_matrices(
        matrices,
        noise=PhenomenologicalNoise(measurement_flip_per_check=rates),
    )

    assert order != tuple(range(len(code.checks)))
    assert model.num_errors == matrices.num_checks
    assert model.detector_rates().tolist() == pytest.approx(list(rates))
    declaration_order = [rates[order.index(row)] for row in range(len(rates))]
    assert declaration_order != list(rates)


def test_a_code_whose_types_are_already_grouped_sees_one_order() -> None:
    """A code declaring every Z-type check before every X-type check needs no move."""

    repetition = RepetitionCode(distance=5)
    steane = SteaneCode()

    assert _check_rate_order(repetition.checks) == tuple(range(len(repetition.checks)))
    assert _check_rate_order(steane.checks) == tuple(range(len(steane.checks)))


def test_the_order_translation_is_a_permutation() -> None:
    """Every check has exactly one row and every row exactly one check."""

    for code in (
        RepetitionCode(distance=5),
        RotatedSurfaceCode(distance=3),
        SteaneCode(),
    ):
        order = _check_rate_order(code.checks)
        assert sorted(order) == list(range(len(code.checks)))


# --------------------------------------------------------------------------
# the two routes read one vector the same way
# --------------------------------------------------------------------------


def test_both_construction_routes_resolve_one_vector_to_the_same_rates() -> None:
    """The circuit route and the matrix route read the same elements.

    The two routes describe different experiments, so their geometries are not
    compared here; what is compared is the rate of each data fault they share,
    which is the part a vector could be read two ways about.
    """

    code = RepetitionCode(distance=3)
    memory = build_memory_circuit(code, rounds=_ROUNDS)
    noise = PhenomenologicalNoise(data_flip=0.5, data_flip_per_qubit=(0.0, 0.11, 0.22))
    circuit_rates = sorted(
        mechanism.probability for mechanism in _mechanisms(memory, noise)
    )
    matrix_rates = sorted(
        entry[0] for entry in _entries(code, noise, num_rounds=_ROUNDS)
    )

    assert circuit_rates == pytest.approx([0.11, 0.11, 0.22, 0.22])
    assert matrix_rates == pytest.approx(circuit_rates)


def test_the_circuit_route_reads_the_per_check_vector_in_the_matrix_order() -> None:
    """The sampler's translation and the construction route's are the same one.

    Both read the rate of a check at its declaration position out of a vector
    indexed by matrix row, so the rates the mechanisms carry are the ones the row
    order names. A second translation that disagreed would put one check's rate
    on another check.
    """

    code = RotatedSurfaceCode(distance=3)
    memory = build_memory_circuit(code, rounds=1)
    rates = tuple(0.001 * (row + 1) for row in range(len(code.checks)))
    noise = PhenomenologicalNoise(measurement_flip_per_check=rates)
    mechanisms = _mechanisms(memory, noise)
    order = _check_rate_order(code.checks)

    assert len(mechanisms) == len(code.checks)
    for position, check in enumerate(code.checks):
        matching = [
            mechanism
            for mechanism in mechanisms
            if mechanism.wire == check.ancilla_qubit
        ]
        assert len(matching) == 1
        assert matching[0].probability == pytest.approx(rates[order[position]])


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "vector"),
    (
        ("data_flip_per_qubit", (0.1, 0.2)),
        ("phase_flip_per_qubit", (0.1, 0.2)),
        ("both_flip_per_qubit", (0.1, 0.2)),
    ),
)
def test_a_data_vector_that_does_not_name_every_qubit_is_refused(
    field: str, vector: tuple[float, ...]
) -> None:
    """A short vector would read one qubit's rate as another's, so it is refused.

    The message names the field, the count stated and the count the code has, so
    a caller can see which of the two is wrong rather than only that they differ.
    """

    noise = PhenomenologicalNoise(data_flip=0.05, **{field: vector})
    with pytest.raises(
        ValueError,
        match=rf"{field} states 2 rate\(s\) but the code declares 3 data qubit\(s\)",
    ):
        DetectorErrorModel.from_code(RepetitionCode(distance=3), noise=noise)


def test_a_per_check_vector_that_does_not_name_every_check_is_refused() -> None:
    """The check count is the code's, and the order is the matrix row order."""

    noise = PhenomenologicalNoise(measurement_flip_per_check=(0.02,) * 7)
    with pytest.raises(
        ValueError,
        match=(
            r"measurement_flip_per_check states 7 rate\(s\) but the code declares "
            r"8 check\(s\)"
        ),
    ):
        DetectorErrorModel.from_code(RotatedSurfaceCode(distance=3), noise=noise)


def test_the_circuit_route_refuses_the_same_two_lengths() -> None:
    """The circuit route resolves the vectors against its own code record.

    A vector that named fewer wires than the circuit has would be read as a rate
    for the wrong wire here too, so the same refusal is reached before any
    mechanism is enumerated and before anything is executed.
    """

    memory = build_memory_circuit(RepetitionCode(distance=3), rounds=_ROUNDS)

    with pytest.raises(ValueError, match=r"states 2 rate\(s\) but the code declares"):
        DetectorErrorModel.from_memory_circuit(
            memory, noise=PhenomenologicalNoise(data_flip_per_qubit=(0.1, 0.2))
        )
    with pytest.raises(
        ValueError, match=r"measurement_flip_per_check states 1 rate\(s\)"
    ):
        DetectorErrorModel.from_memory_circuit(
            memory,
            noise=PhenomenologicalNoise(measurement_flip_per_check=(0.1,)),
        )


def test_the_matrix_route_refuses_a_short_vector_before_it_reads_anything() -> None:
    """The matrix route's read is refused at the same point and for the same reason.

    The refusal arrives from the resolution rather than from the matrices, so a
    vector is rejected even when the code's blocks would have made the mismatch
    invisible -- a code with no row of a type still declares its data qubits.
    """

    matrices = css_code_matrices(RotatedSurfaceCode(distance=3))

    with pytest.raises(ValueError, match=r"states 2 rate\(s\) but the code declares"):
        DetectorErrorModel.from_code_matrices(
            matrices,
            noise=PhenomenologicalNoise(both_flip_per_qubit=(0.1, 0.2)),
        )
    with pytest.raises(ValueError, match=r"states 4 rate\(s\) but the code declares"):
        DetectorErrorModel.from_code_matrices(
            matrices,
            noise=PhenomenologicalNoise(phase_flip_per_qubit=(0.1,) * 4),
        )


def test_the_three_readers_agree_on_the_length_they_refuse() -> None:
    """One code, one vector, one refusal, three entry points.

    A caller who states a vector of the wrong length is told the same thing
    whichever route reads it, which is the point of resolving the vector in one
    place: a route that decided the length for itself could accept what another
    refused.
    """

    code = RepetitionCode(distance=3)
    memory = build_memory_circuit(code, rounds=_ROUNDS)
    noise = PhenomenologicalNoise(measurement_flip_per_check=(0.1, 0.2, 0.3))
    message = (
        r"measurement_flip_per_check states 3 rate\(s\) but the code declares "
        r"2 check\(s\)"
    )

    with pytest.raises(ValueError, match=message):
        DetectorErrorModel.from_code(code, noise=noise)
    with pytest.raises(ValueError, match=message):
        DetectorErrorModel.from_code_matrices(css_code_matrices(code), noise=noise)
    with pytest.raises(ValueError, match=message):
        DetectorErrorModel.from_memory_circuit(memory, noise=noise)
