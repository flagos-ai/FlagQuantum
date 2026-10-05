"""A code record read off parity-check matrices rather than declared by a class.

The records the package ships are declared: a class states its qubit layout and
builds its checks from it. This file pins the other direction of that seam. A
caller who has the four Calderbank-Shor-Steane matrices and no class declares a
`~flagquantum.qec.CssCode`, and every count and layout a code record owes is read
off those matrices.

Two things are worth stating where the tests can be read against them. The
distance is passed in rather than derived, because computing a quantum code's
distance is exponential in general and a record that derived it would refuse the
large matrices this route exists for; what the record does instead is refuse a
distance its own matrices contradict, which is the one direction a stated logical
operator can prove. And the matrices are held to the algebra the four blocks do
not carry, so matrices that do not describe a code are refused at construction
rather than decoded against a model of nothing.

The comparison with the declared records is deliberately not a comparison of
detector error model text. A matrix carries no qubit labels, so a record rebuilt
from another record's matrices states the same checks in the order the two blocks
list them -- Z-type first, then X-type -- while the declared surface code
interleaves its two bases in lattice order. A detector's identity is the check
and the round it belongs to, so the two experiments are compared under exactly
that relabelling, which is stated here rather than assumed.
"""

from __future__ import annotations

import pytest
import torch

from flagquantum.qec import (
    CssCode,
    CssCodeMatrices,
    DetectorErrorModel,
    MemoryCircuit,
    PhenomenologicalNoise,
    RepetitionCode,
    RotatedSurfaceCode,
    StabilizerCode,
    SteaneCode,
    ancilla_bands,
    build_memory_circuit,
    code_names,
    css_code_matrices,
    get_code,
    sample_memory_circuit,
)

pytestmark = pytest.mark.integration

_ROUNDS = 3
_NOISE = PhenomenologicalNoise(
    data_flip=0.02,
    phase_flip=0.01,
    both_flip=0.005,
    measurement_flip=0.01,
)


def _bold(qubits: tuple[int, ...], *, width: int) -> list[int]:
    """One row of a parity-check block over ``width`` qubits."""

    row = [0] * width
    for qubit in qubits:
        row[qubit] = 1
    return row


def _shor_matrices() -> CssCodeMatrices:
    """The nine-qubit Shor code, whose check blocks no class here declares.

    The code is three three-qubit bit-flip blocks with phase-flip checks across
    them: ``Z_i Z_{i+1}`` inside each block, and ``X`` over the first six data
    qubits and over the last six. Its distance is three in both bases, and the
    lightest logical operator it states is the weight-three product of one qubit
    from each block.
    """

    width = 9
    z_rows = ((0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8))
    x_rows = ((0, 1, 2, 3, 4, 5), (3, 4, 5, 6, 7, 8))
    return CssCodeMatrices(
        hz=torch.tensor([_bold(row, width=width) for row in z_rows]),
        hx=torch.tensor([_bold(row, width=width) for row in x_rows]),
        lz=torch.tensor([_bold((0, 3, 6), width=width)]),
    )


def _rebuilt(code: StabilizerCode) -> CssCode:
    return CssCode(matrices=css_code_matrices(code), distance=code.distance)


def _check_keys(checks: tuple[object, ...]) -> tuple[tuple[str, int], ...]:
    """Name each check by its basis and its position within that basis.

    That pair is what survives a rebuild: the matrices keep the order of each
    block, and a matrix carries nothing about which wire a check was laid out on.
    """

    seen = {"z": 0, "x": 0}
    keys: list[tuple[str, int]] = []
    for check in checks:
        basis = "x" if check.stabilizer.x_qubits else "z"
        keys.append((basis, seen[basis]))
        seen[basis] += 1
    return tuple(keys)


def _detector_addresses(circuit: MemoryCircuit) -> dict[tuple[object, ...], int]:
    """Map every detector to the check and round that placed it.

    This restates the layout rule ``build_memory_circuit`` uses -- round zero
    names no X-type check, every round after it compares against the round before
    it, and a Z-type check ends in a terminal detector against the data readout --
    so that two circuits can be compared by the experiment they describe rather
    than by the order their checks happen to be listed in.
    """

    code = circuit.code
    keys = _check_keys(code.checks)
    addresses: dict[tuple[object, ...], int] = {}
    index = 0
    for round_index in range(circuit.rounds):
        for check, key in zip(code.checks, keys, strict=True):
            if round_index == 0 and check.stabilizer.x_qubits:
                continue
            addresses[key + (round_index,)] = index
            index += 1
    for check, key in zip(code.checks, keys, strict=True):
        if check.stabilizer.x_qubits:
            continue
        addresses[key + ("terminal",)] = index
        index += 1
    assert index == len(
        circuit.detectors
    ), "the restated layout disagrees with the built one"
    return addresses


def _mechanisms(
    model: DetectorErrorModel,
    permutation: dict[int, int] | None = None,
) -> list[tuple[object, ...]]:
    """Every mechanism as a comparable record, optionally under a detector relabelling."""

    records: list[tuple[object, ...]] = []
    for error in model.errors:
        detectors = error.detectors
        if permutation is not None:
            detectors = tuple(permutation[index] for index in detectors)
        records.append(
            (error.probability, tuple(sorted(detectors)), tuple(error.observables))
        )
    return sorted(records, key=repr)


def _relabelling(declared: MemoryCircuit, rebuilt: MemoryCircuit) -> dict[int, int]:
    """Rename a rebuilt detector as the declared detector naming the same check.

    The detector order of a rebuild is the order of its own check list; the two
    address maps are keyed by ``(basis, position, round)``, which is what both
    records agree on, so the map from one to the other is what makes the two
    models comparable.
    """

    left = _detector_addresses(declared)
    right = _detector_addresses(rebuilt)
    assert (
        left.keys() == right.keys()
    ), "the two circuits do not describe the same checks"
    return {right[address]: left[address] for address in left}


# --- a declared record rebuilds itself from its own matrices ----------------


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (RepetitionCode(distance=3), (3, 2, 0, 2)),
        (RepetitionCode(distance=5), (5, 4, 0, 4)),
        (RotatedSurfaceCode(distance=3), (9, 8, 4, 4)),
        (RotatedSurfaceCode(distance=5), (25, 24, 12, 12)),
        (SteaneCode(), (7, 6, 3, 3)),
    ],
)
def test_a_shipped_record_rebuilds_from_its_own_matrices(
    code: StabilizerCode,
    expected: tuple[int, int, int, int],
) -> None:
    """The two routes into a record agree on every count a record states."""

    record = _rebuilt(code)
    data, ancillas, x_count, z_count = expected

    assert isinstance(record, StabilizerCode)
    assert record.num_data_qubits == data == code.num_data_qubits
    assert record.num_ancilla_qubits == ancillas == code.num_ancilla_qubits
    assert record.num_ancilla_x_qubits == x_count == code.num_ancilla_x_qubits
    assert record.num_ancilla_z_qubits == z_count == code.num_ancilla_z_qubits
    assert record.num_x_stabilizers == code.num_x_stabilizers
    assert record.num_z_stabilizers == code.num_z_stabilizers
    assert record.distance == code.distance
    assert record.data_qubits == code.data_qubits
    assert record.ancilla_qubits == code.ancilla_qubits


@pytest.mark.parametrize(
    "code",
    [
        RepetitionCode(distance=3),
        RotatedSurfaceCode(distance=3),
        RotatedSurfaceCode(distance=5),
        SteaneCode(),
    ],
)
def test_a_shipped_record_rebuilds_the_same_checks_and_logicals(
    code: StabilizerCode,
) -> None:
    """A rebuild carries one check per check and one observable per observable."""

    record = _rebuilt(code)

    assert len(record.checks) == len(code.checks)
    assert {check.stabilizer for check in record.checks} == {
        check.stabilizer for check in code.checks
    }
    assert record.stabilizers == tuple(check.stabilizer for check in record.checks)
    assert list(record.logical_observables) == list(code.logical_observables)


def test_a_rebuild_states_the_z_checks_before_the_x_checks() -> None:
    """The ancilla order is the one the two blocks are read in, not a second fact."""

    record = _rebuilt(RotatedSurfaceCode(distance=3))
    bases = ["x" if check.stabilizer.x_qubits else "z" for check in record.checks]

    assert bases == ["z"] * 4 + ["x"] * 4
    assert record.ancilla_qubits == tuple(range(9, 17))
    assert [check.ancilla_qubit for check in record.checks] == list(range(9, 17))
    assert [check.index for check in record.checks] == list(range(8))


def test_a_z_check_couples_the_data_into_the_ancilla_and_the_x_check_mirrors_it() -> (
    None
):
    """The CNOT direction is the check type's, exactly as a declared check states it."""

    record = _rebuilt(RotatedSurfaceCode(distance=3))

    for check in record.checks:
        ancilla = check.ancilla_qubit
        if check.stabilizer.z_qubits:
            assert check.cnot_qubits == tuple(
                (qubit, ancilla) for qubit in check.stabilizer.z_qubits
            )
        else:
            assert check.cnot_qubits == tuple(
                (ancilla, qubit) for qubit in check.stabilizer.x_qubits
            )


def test_a_rebuild_derives_its_bands_from_the_checks_it_built() -> None:
    """The two bands are read off the checks rather than stated beside them."""

    record = _rebuilt(RotatedSurfaceCode(distance=3))
    x_qubits, z_qubits = ancilla_bands(record.checks)

    assert record.num_ancilla_x_qubits == len(x_qubits)
    assert record.num_ancilla_z_qubits == len(z_qubits)
    assert x_qubits == (13, 14, 15, 16)
    assert z_qubits == (9, 10, 11, 12)


# --- the rebuilt record describes the same experiment -----------------------


@pytest.mark.parametrize(
    "code",
    [
        RepetitionCode(distance=3),
        RepetitionCode(distance=5),
        # The rotated surface code is exercised at distance three and no further,
        # because a detector error model built from a memory *circuit* is read by
        # executing the source, so its size is bounded by the statevector: the
        # surface code stops being executable here at distance three. That ceiling
        # belongs to the circuit route and not to this record, and the matrix
        # route below is exercised at distance five to say so.
        RotatedSurfaceCode(distance=3),
    ],
)
def test_a_rebuild_describes_the_same_experiment_as_the_declared_record(
    code: StabilizerCode,
) -> None:
    """The two routes produce one experiment, detector naming aside.

    The relabelling is the check order, which is the only thing a matrix cannot
    carry: the declared surface code interleaves its two bases and a rebuild
    groups them. Rates, detector flip sets and observable flip sets must agree
    exactly under it, with no tolerance, because both routes enumerate the same
    faults over the same supports.
    """

    declared = build_memory_circuit(code, rounds=_ROUNDS)
    rebuilt_circuit = build_memory_circuit(_rebuilt(code), rounds=_ROUNDS)
    relabelling = _relabelling(declared, rebuilt_circuit)

    declared_model = DetectorErrorModel.from_memory_circuit(declared, noise=_NOISE)
    rebuilt_model = DetectorErrorModel.from_memory_circuit(
        rebuilt_circuit, noise=_NOISE
    )

    assert rebuilt_model.num_detectors == declared_model.num_detectors
    assert rebuilt_model.num_observables == declared_model.num_observables
    assert rebuilt_model.num_errors == declared_model.num_errors
    assert _mechanisms(rebuilt_model, relabelling) == _mechanisms(declared_model)


def test_a_repetition_rebuild_lands_on_the_same_detector_text() -> None:
    """Where the two records list their checks in one order, even the text agrees."""

    code = RepetitionCode(distance=3)
    declared = DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(code, rounds=_ROUNDS), noise=_NOISE
    )
    rebuilt = DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(_rebuilt(code), rounds=_ROUNDS), noise=_NOISE
    )

    assert rebuilt.to_stim_text() == declared.to_stim_text()


def test_a_x_type_check_is_the_only_reason_the_surface_text_differs() -> None:
    """The surface rebuild is the same model under a relabelling, and says so."""

    code = RotatedSurfaceCode(distance=3)
    declared = DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(code, rounds=_ROUNDS), noise=_NOISE
    )
    rebuilt = DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(_rebuilt(code), rounds=_ROUNDS), noise=_NOISE
    )

    assert rebuilt.to_stim_text() != declared.to_stim_text()
    assert _mechanisms(rebuilt) != _mechanisms(declared)


def test_a_rebuilt_record_reaches_the_matrix_route_in_one_call() -> None:
    """``from_code`` reads the matrices the record was built from, not a re-derivation."""

    record = _rebuilt(RotatedSurfaceCode(distance=3))
    through_record = DetectorErrorModel.from_code(record, noise=_NOISE, num_rounds=2)
    through_matrices = DetectorErrorModel.from_code_matrices(
        record.matrices, noise=_NOISE, num_rounds=2
    )

    assert through_record.to_stim_text() == through_matrices.to_stim_text()


def test_the_matrix_route_has_no_statevector_ceiling() -> None:
    """A record too large to execute still reaches a model by reading its matrices.

    This is where the two routes of one record genuinely part: a memory circuit is
    a program, so its model is read by executing it and stops where the
    statevector stops, while the matrix route reads a support and has no ceiling
    to stop at. The record itself is built either way.
    """

    num_rounds = 2
    record = _rebuilt(RotatedSurfaceCode(distance=5))
    model = DetectorErrorModel.from_code(record, noise=_NOISE, num_rounds=num_rounds)

    assert record.num_data_qubits == 25
    assert model.num_detectors == num_rounds * len(record.checks)
    assert model.num_errors > 0


def test_a_rebuilt_record_runs_as_a_memory_experiment() -> None:
    """A record that was never declared by a class still samples a memory circuit."""

    record = _rebuilt(RotatedSurfaceCode(distance=3))
    circuit = build_memory_circuit(record, rounds=_ROUNDS)
    samples = sample_memory_circuit(circuit, noise=_NOISE, shots=64, seed=5)

    assert samples.detectors.shape == (64, len(circuit.detectors))
    assert samples.observables.shape == (64, len(record.logical_observables))


# --- a code no class here declares -----------------------------------------


def test_a_code_this_package_does_not_ship_is_declared_by_its_matrices() -> None:
    """The Shor code has no class, and the matrices are enough to state it."""

    record = CssCode(matrices=_shor_matrices(), distance=3)

    assert isinstance(record, StabilizerCode)
    assert record.num_data_qubits == 9
    assert record.num_ancilla_qubits == 8
    assert record.num_ancilla_x_qubits == 2
    assert record.num_ancilla_z_qubits == 6
    assert record.num_x_stabilizers == 2
    assert record.num_z_stabilizers == 6
    assert [check.stabilizer.z_qubits != () for check in record.checks] == [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    assert record.logical_observables[0].z_qubits == (0, 3, 6)


def test_a_code_no_class_declares_builds_a_memory_circuit_and_a_model() -> None:
    """The whole route works on a record whose only source is four blocks."""

    record = CssCode(matrices=_shor_matrices(), distance=3)
    circuit = build_memory_circuit(record, rounds=_ROUNDS)
    model = DetectorErrorModel.from_memory_circuit(circuit, noise=_NOISE)

    assert len(circuit.detectors) == 6 * (_ROUNDS + 1) + 2 * (_ROUNDS - 1)
    assert model.num_detectors == len(circuit.detectors)
    assert model.num_observables == 1
    assert model.num_errors > 0


def test_a_code_is_not_a_second_registry() -> None:
    """A matrices record is reached by its matrices, so no name is reserved for it."""

    before = code_names()

    CssCode(matrices=_shor_matrices(), distance=3)

    assert code_names() == before == ("repetition", "rotated_surface", "steane")
    with pytest.raises(ValueError, match="no code is registered as 'shor'"):
        get_code("shor")


def test_a_code_stating_both_bases_reads_one_observable_per_row() -> None:
    """Two families are two observable blocks, Z-type first, in row order."""

    record = CssCode(matrices=css_code_matrices(SteaneCode()), distance=3)

    assert len(record.logical_observables) == 2
    assert record.logical_observables[0].z_qubits == (0, 1, 2)
    assert record.logical_observables[1].x_qubits == (0, 1, 2)


def test_the_circuit_route_refuses_an_x_type_logical_a_matrix_record_admits() -> None:
    """The matrix route is wider than the memory-circuit route, and not silently.

    A memory circuit starts and ends in the Z basis, so it can only read out a
    Z-type logical observable. A record with an X-type observable is therefore
    perfectly well formed and still has no circuit here, which is the boundary
    this route does not move.
    """

    record = CssCode(matrices=css_code_matrices(SteaneCode()), distance=3)

    with pytest.raises(ValueError, match="Z-type logical observable"):
        build_memory_circuit(record, rounds=_ROUNDS)


# --- every refusal by name --------------------------------------------------


def test_a_record_is_built_from_the_matrix_record_and_not_from_a_tensor() -> None:
    with pytest.raises(TypeError, match="built from a CssCodeMatrices"):
        CssCode(matrices=torch.eye(3), distance=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("distance", [True, 2.0, "3", None])
def test_a_distance_that_is_not_an_integer_is_refused(distance: object) -> None:
    with pytest.raises(TypeError, match="distance must be an integer"):
        CssCode(matrices=_shor_matrices(), distance=distance)  # type: ignore[arg-type]


@pytest.mark.parametrize("distance", [0, -1])
def test_a_distance_below_one_is_refused(distance: int) -> None:
    with pytest.raises(ValueError, match="at least one"):
        CssCode(matrices=_shor_matrices(), distance=distance)


def test_matrices_with_no_check_are_not_a_code() -> None:
    width = 3
    with pytest.raises(ValueError, match="state no check"):
        CssCode(
            matrices=CssCodeMatrices(
                hz=torch.zeros((0, width), dtype=torch.int64),
                lz=torch.tensor([_bold((0, 1, 2), width=width)]),
            ),
            distance=1,
        )


def test_matrices_with_no_logical_operator_are_not_an_error_correcting_code() -> None:
    width = 3
    with pytest.raises(ValueError, match="state no logical observable"):
        CssCode(
            matrices=CssCodeMatrices(
                hz=torch.tensor([_bold((0, 1), width=width)]),
            ),
            distance=1,
        )


def test_a_check_row_that_is_empty_is_refused() -> None:
    width = 3
    with pytest.raises(ValueError, match="is empty"):
        CssCode(
            matrices=CssCodeMatrices(
                hz=torch.tensor([_bold((0, 1), width=width), [0, 0, 0]]),
                lz=torch.tensor([_bold((0, 1, 2), width=width)]),
            ),
            distance=1,
        )


def test_a_logical_operator_that_anticommutes_with_a_check_is_refused() -> None:
    """A stated logical operator is read against the opposite basis' checks."""

    width = 9
    matrices = CssCodeMatrices(
        hz=torch.tensor([_bold(row, width=width) for row in ((0, 1), (1, 2))]),
        hx=torch.tensor([_bold((0, 1, 2), width=width)]),
        lz=torch.tensor([_bold((0,), width=width)]),
    )

    with pytest.raises(ValueError, match="anticommute"):
        CssCode(matrices=matrices, distance=1)


def test_a_logical_operator_that_is_a_product_of_checks_is_refused() -> None:
    """A row inside the check span is a stabilizer, not a logical operator."""

    width = 3
    matrices = CssCodeMatrices(
        hz=torch.tensor([_bold(row, width=width) for row in ((0, 1), (1, 2))]),
        lz=torch.tensor([_bold((0, 2), width=width)]),
    )

    with pytest.raises(ValueError, match="stabilizer rather than a logical operator"):
        CssCode(matrices=matrices, distance=1)


def test_two_dependent_logical_operators_of_one_basis_are_refused() -> None:
    """Each row becomes one observable, so two rows for one operator are refused.

    The two rows here are the same weight-one operator stated twice, which is the
    shape a caller reaches by concatenating one code's logical rows with another's
    without reducing them. Both rows pass the reading a logical operator gets; it
    is the pair that cannot become two observables.
    """

    width = 3
    matrices = CssCodeMatrices(
        hz=torch.tensor([_bold((0, 1), width=width)]),
        lz=torch.tensor([_bold((2,), width=width), _bold((2,), width=width)]),
    )

    with pytest.raises(ValueError, match="not independent"):
        CssCode(matrices=matrices, distance=1)


def test_a_distance_the_matrices_contradict_is_refused() -> None:
    """A stated logical operator proves the distance is at most its own weight."""

    record = CssCode(matrices=_shor_matrices(), distance=3)

    assert record.distance == 3
    with pytest.raises(ValueError, match="exceeds the weight"):
        CssCode(matrices=_shor_matrices(), distance=4)


def test_a_distance_below_the_stated_logical_weight_is_accepted() -> None:
    """The check is one-directional, and the direction that was dropped is stated."""

    record = CssCode(matrices=_shor_matrices(), distance=1)

    assert record.distance == 1


# --- the record is a value --------------------------------------------------


def test_a_record_compares_by_its_matrices_and_its_distance() -> None:
    record = CssCode(matrices=_shor_matrices(), distance=3)

    assert record == CssCode(matrices=_shor_matrices(), distance=3)
    assert record != CssCode(matrices=_shor_matrices(), distance=1)
    assert record != CssCode(matrices=css_code_matrices(SteaneCode()), distance=3)
    assert record != "a shor code"
    assert record.__eq__(object()) is NotImplemented


def test_a_record_is_unhashable_because_its_content_is_a_tensor() -> None:
    record = CssCode(matrices=_shor_matrices(), distance=3)

    with pytest.raises(TypeError, match="unhashable"):
        hash(record)


def test_the_record_is_published_from_the_package_root() -> None:
    import flagquantum.qec as qec

    assert "CssCode" in qec.__all__
    assert qec.CssCode is CssCode
