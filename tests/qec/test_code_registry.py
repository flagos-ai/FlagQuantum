"""Unit coverage for the per-basis ancilla bands and the code registry.

Two things are pinned here. The first is that a code's X-type and Z-type
ancilla bands are a function of its checks rather than a second declaration
beside them, which is the shape upstream's code record has: it carries
``get_num_ancilla_x_qubits`` and ``get_num_ancilla_z_qubits`` next to
``get_num_ancilla_qubits``, and a syndrome-extraction round is laid out against
the split. Beside those it carries ``get_num_x_stabilizers`` and
``get_num_z_stabilizers``, and each code class it ships answers the same number
to a stabilizer count as to the band count of that basis, so the two are one
quantity here. The second is that a record can be reached by name, which is what
upstream's ``get_code(name, options)`` and ``get_available_codes()`` do.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from flagquantum.qec.codes import (
    CodeCheck,
    RepetitionCode,
    RotatedSurfaceCode,
    StabilizerCode,
    SteaneCode,
    ancilla_bands,
    code_names,
    get_code,
    register_code,
)
from flagquantum.qec.pauli import Pauli

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _Bands:
    """A check-free statement of one code's two bands, for the parametrised cases."""

    code: object
    x_count: int
    z_count: int


def _repetition_checks(distance: int) -> tuple[CodeCheck, ...]:
    return RepetitionCode(distance).checks


# --- the bands are derived from the checks ---------------------------------


def test_bands_split_the_repetition_ancillas_by_check_type() -> None:
    """The repetition code measures Z only, so its X band is empty."""

    x_qubits, z_qubits = ancilla_bands(_repetition_checks(5))

    assert x_qubits == ()
    assert z_qubits == (5, 6, 7, 8)


def test_bands_are_sorted_and_carry_no_duplicate_wire() -> None:
    """A wire is listed once however many checks are read off it.

    No shipped code reads one ancilla for two checks, so the deduplication is
    exercised on a record that does: the two checks share wire 5.
    """

    checks = (
        CodeCheck(
            index=0,
            stabilizer=Pauli(z_qubits=(0, 1)),
            ancilla_qubit=5,
            cnot_qubits=((0, 5), (1, 5)),
        ),
        CodeCheck(
            index=1,
            stabilizer=Pauli(z_qubits=(1, 2)),
            ancilla_qubit=5,
            cnot_qubits=((1, 5), (2, 5)),
        ),
    )

    x_qubits, z_qubits = ancilla_bands(checks)

    assert x_qubits == ()
    assert z_qubits == (5,)


def test_bands_keep_the_two_bases_apart_on_a_mixed_code() -> None:
    """A code with both kinds of check reports one wire in each band."""

    x_qubits, z_qubits = ancilla_bands(SteaneCode().checks)

    assert x_qubits == (10, 11, 12)
    assert z_qubits == (7, 8, 9)


def test_bands_refuse_an_ancilla_that_measures_both_bases() -> None:
    """One ancilla measures one basis, so it cannot be in both bands.

    Nothing downstream could read such a record: a detector band is a set of
    handles, and a handle in both bands would be compared against two known
    parities. The refusal names the wire so the caller can find it.
    """

    checks = (
        CodeCheck(
            index=0,
            stabilizer=Pauli(z_qubits=(0, 1)),
            ancilla_qubit=5,
            cnot_qubits=((0, 5), (1, 5)),
        ),
        CodeCheck(
            index=1,
            stabilizer=Pauli(x_qubits=(0, 1)),
            ancilla_qubit=5,
            cnot_qubits=((5, 0), (5, 1)),
        ),
    )

    with pytest.raises(ValueError, match="ancillas 5 measure a check of each basis"):
        ancilla_bands(checks)


def test_bands_leave_a_flag_ancilla_out_of_both() -> None:
    """A wire that measures neither basis is in neither band.

    The bands are not a partition of the declared ancillas, and this is why the
    consistency guard compares counts against bands rather than against the
    total: a code that carries an idle or flag ancilla still has a well defined
    split.
    """

    x_qubits, z_qubits = ancilla_bands(_repetition_checks(3))

    assert x_qubits == ()
    assert z_qubits == (3, 4)
    assert len(x_qubits) + len(z_qubits) < RepetitionCode(3).num_ancilla_qubits + 1


@pytest.mark.parametrize(
    ("record", "x_count", "z_count"),
    [
        (RepetitionCode(2), 0, 1),
        (RepetitionCode(7), 0, 6),
        (RotatedSurfaceCode(3), 4, 4),
        (RotatedSurfaceCode(5), 12, 12),
        (SteaneCode(), 3, 3),
    ],
)
def test_each_shipped_record_reports_the_bands_its_checks_define(
    record: StabilizerCode, x_count: int, z_count: int
) -> None:
    """The stated per-basis counts and the derived bands agree, code by code."""

    x_qubits, z_qubits = ancilla_bands(record.checks)

    assert (record.num_ancilla_x_qubits, record.num_ancilla_z_qubits) == (
        x_count,
        z_count,
    )
    assert (len(x_qubits), len(z_qubits)) == (x_count, z_count)


@pytest.mark.parametrize(
    "record",
    [RepetitionCode(3), RotatedSurfaceCode(3), SteaneCode()],
)
def test_the_per_basis_counts_never_exceed_the_declared_total(
    record: StabilizerCode,
) -> None:
    """The split is a partition of what the checks use, so it cannot overrun it."""

    assert record.num_ancilla_x_qubits + record.num_ancilla_z_qubits <= (
        record.num_ancilla_qubits
    )


def test_the_protocol_declares_the_per_basis_split() -> None:
    """The split is part of the protocol, not a property of three classes.

    A consumer is typed against :class:`StabilizerCode`, so a member the
    consumers need has to be on the protocol; a member only the shipped records
    carry would leave the next record free to omit it and be accepted.
    """

    for member in ("num_ancilla_x_qubits", "num_ancilla_z_qubits"):
        assert member in vars(StabilizerCode), f"{member} is not on the protocol"


def test_the_protocol_declares_the_stabilizer_counts() -> None:
    """The two further counts upstream's record declares are on the protocol.

    ``code.h`` declares ``get_num_x_stabilizers`` and ``get_num_z_stabilizers``
    as pure virtuals beside the three ancilla counts, so a caller reading
    upstream's accessor list reads five counts. A member only the shipped
    records carry would leave the next record free to omit it and be accepted.
    """

    for member in ("num_x_stabilizers", "num_z_stabilizers"):
        assert member in vars(StabilizerCode), f"{member} is not on the protocol"


@pytest.mark.parametrize(
    ("record", "x_count", "z_count"),
    [
        (RepetitionCode(2), 0, 1),
        (RepetitionCode(7), 0, 6),
        (RotatedSurfaceCode(3), 4, 4),
        (RotatedSurfaceCode(5), 12, 12),
        (SteaneCode(), 3, 3),
    ],
)
def test_each_shipped_record_answers_the_stabilizer_counts_from_its_checks(
    record: StabilizerCode, x_count: int, z_count: int
) -> None:
    """The two stabilizer counts are the two bands rather than a second count.

    One ancilla measures one stabilizer here exactly as it does in each of the
    three code classes upstream ships, and every one of those answers the same
    number to a stabilizer count as to the band count of that basis. The two are
    therefore one quantity: the accessors read the bands, and the number of
    checks of each basis, which is the definition upstream counts, agrees.
    """

    checks = tuple(record.checks)
    assert (record.num_x_stabilizers, record.num_z_stabilizers) == (x_count, z_count)
    assert (
        sum(1 for check in checks if check.stabilizer.x_qubits),
        sum(1 for check in checks if check.stabilizer.z_qubits),
    ) == (x_count, z_count)
    assert record.num_x_stabilizers == record.num_ancilla_x_qubits
    assert record.num_z_stabilizers == record.num_ancilla_z_qubits


# --- the registry ----------------------------------------------------------


@dataclass(frozen=True)
class _Registered:
    """A minimal record, so the registry is exercised off the shipped classes."""

    distance: int = 3

    num_data_qubits = 3
    num_ancilla_qubits = 2
    num_ancilla_x_qubits = 0
    num_ancilla_z_qubits = 2
    num_x_stabilizers = 0
    num_z_stabilizers = 2
    data_qubits = (0, 1, 2)
    ancilla_qubits = (3, 4)
    checks = RepetitionCode(3).checks
    stabilizers = RepetitionCode(3).stabilizers
    logical_observables = (Pauli(z_qubits=(0, 1, 2)),)


def test_the_registry_holds_the_three_shipped_families() -> None:
    assert code_names() == ("repetition", "rotated_surface", "steane")


@pytest.mark.parametrize(
    ("name", "options", "data_qubits", "ancillas"),
    [
        ("repetition", {"distance": 5}, 5, 4),
        ("rotated_surface", {"distance": 3}, 9, 8),
        ("steane", {}, 7, 6),
    ],
)
def test_a_registered_name_builds_the_record_it_names(
    name: str, options: dict[str, int], data_qubits: int, ancillas: int
) -> None:
    """A name reaches the same record a direct construction reaches.

    The options are the record's own fields, so this compares the two routes
    field for field rather than only checking that something was returned.
    """

    built = get_code(name, **options)
    direct = {
        "repetition": RepetitionCode,
        "rotated_surface": RotatedSurfaceCode,
        "steane": SteaneCode,
    }[name](**options)

    assert built == direct
    assert built.num_data_qubits == data_qubits
    assert built.num_ancilla_qubits == ancillas


def test_a_registered_name_builds_a_record_with_no_options() -> None:
    """A record whose fields all default is built by name alone."""

    assert get_code("repetition") == RepetitionCode()


def test_an_unregistered_name_lists_the_registered_ones() -> None:
    with pytest.raises(ValueError, match="no code is registered as 'shor'"):
        get_code("shor")


def test_an_unknown_option_names_the_fields_the_record_takes() -> None:
    """The refusal lists what the record takes, not what the caller passed.

    A record built from a name is still configured by its own fields, so the
    check happens here rather than inside the constructor, where the traceback
    would name a parameter the caller never saw.
    """

    with pytest.raises(TypeError, match="takes no option 'rounds'"):
        get_code("repetition", rounds=3)  # type: ignore[call-arg]


def test_a_record_with_no_fields_lists_the_empty_option_set() -> None:
    """The option refusal reads as a sentence when the record takes nothing.

    The stand-in below declares every protocol member and no field at all, which
    is the only shape that reaches this branch.
    """

    class _Fieldless:
        distance = 2
        num_data_qubits = 2
        num_ancilla_qubits = 1
        num_ancilla_x_qubits = 0
        num_ancilla_z_qubits = 1
        num_x_stabilizers = 0
        num_z_stabilizers = 1
        data_qubits = (0, 1)
        ancilla_qubits = (2,)
        checks = _repetition_checks(2)
        stabilizers = (Pauli(z_qubits=(0, 1)),)
        logical_observables = (Pauli(z_qubits=(0, 1)),)

    register_code("_fieldless")(_Fieldless)
    try:
        with pytest.raises(TypeError, match="it takes no options"):
            get_code("_fieldless", distance=3)
    finally:
        _drop("_fieldless")


def test_a_name_is_registered_once_so_a_second_registration_is_refused() -> None:
    """Two records answering to one name is a choice the registry cannot make."""

    with pytest.raises(ValueError, match="is already registered against"):
        register_code("steane")(_Registered)


def test_replace_overwrites_a_registration_and_restores_the_original() -> None:
    """``replace=True`` is how a caller rebinds a name deliberately."""

    original = get_code("steane")
    register_code("steane", replace=True)(_Registered)
    try:
        assert get_code("steane") == _Registered()
    finally:
        register_code("steane", replace=True)(SteaneCode)
    assert get_code("steane") == original


@pytest.mark.parametrize("name", ["", "   ", "two words", "trailing "])
def test_a_name_that_is_not_one_token_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="no whitespace"):
        register_code(name)(_Registered)


@pytest.mark.parametrize("name", [3, None, b"steane"])
def test_a_name_that_is_not_a_string_is_refused(name: object) -> None:
    with pytest.raises(TypeError, match="must be a string"):
        register_code(name)(_Registered)  # type: ignore[arg-type]


def test_registering_a_class_that_misses_protocol_members_names_them() -> None:
    """Registration checks the members rather than trusting the class.

    A registered class is reached through the same consumers as a shipped
    record, so a class that cannot answer one of them has to be refused where it
    enters the registry, with the missing members named.
    """

    class _Partial:
        distance = 3
        num_data_qubits = 3
        num_ancilla_qubits = 2
        data_qubits = (0, 1, 2)

    with pytest.raises(TypeError) as excinfo:
        register_code("_partial")(_Partial)

    message = str(excinfo.value)
    assert "'_Partial' cannot be registered as '_partial'" in message
    assert "num_ancilla_x_qubits" in message
    assert "num_ancilla_z_qubits" in message
    assert "num_x_stabilizers" in message
    assert "num_z_stabilizers" in message
    assert "_partial" not in code_names()


def test_registration_returns_the_class_unchanged() -> None:
    """A registered record keeps its name and stays directly constructible."""

    registered = register_code("_unchanged")(_Registered)
    try:
        assert registered is _Registered
        assert registered.__name__ == "_Registered"
        assert registered(distance=4).distance == 4
    finally:
        _drop("_unchanged")


def _drop(name: str) -> None:
    """Remove a name a test added, so the registry is the shipped set again."""

    from flagquantum.qec import codes

    codes._CODES.pop(name, None)


# --- the builder refuses a record whose bands contradict its checks ---------


def test_builder_rejects_a_record_whose_stated_bands_disagree_with_its_checks() -> None:
    """The split is stated once, and this is where the second statement is read.

    The record below carries the repetition code's checks -- four Z-type
    checks, so four Z ancillas and no X ancilla -- while reporting one of each.
    A consumer that asked the record and a consumer that read the checks would
    disagree, and nothing would say so.
    """

    @dataclass(frozen=True)
    class _Contradictory:
        num_data_qubits = 5
        num_ancilla_qubits = 4
        num_ancilla_x_qubits = 1
        num_ancilla_z_qubits = 1
        num_x_stabilizers = 1
        num_z_stabilizers = 1
        data_qubits = (0, 1, 2, 3, 4)
        ancilla_qubits = (5, 6, 7, 8)
        checks = _repetition_checks(5)
        stabilizers = RepetitionCode(5).stabilizers
        logical_observables = (Pauli(z_qubits=(0, 1, 2, 3, 4)),)

        @property
        def distance(self) -> int:
            return self.num_data_qubits

    from flagquantum.qec.circuit import build_memory_circuit

    with pytest.raises(ValueError, match="the two bands it states and the two bands"):
        build_memory_circuit(_Contradictory(), rounds=2)


def test_builder_accepts_a_record_whose_bands_its_checks_define() -> None:
    """The guard passes exactly the records ``ancilla_bands`` agrees with.

    The Steane code is among them: it declares a logical observable in each
    basis, and the default readout basis is the Z one, so it builds. The refusal
    that remains is per basis rather than per record -- a code that declares no
    logical observable in the requested basis has no memory experiment there --
    and the two Z-memory records are the ones that name it, because neither
    declares an X-type observable.
    """

    from flagquantum.qec.circuit import build_memory_circuit

    for record in (RepetitionCode(3), RotatedSurfaceCode(3), SteaneCode()):
        built = build_memory_circuit(record, rounds=2)
        assert built.code is record
        assert built.readout_basis == "z"

    with pytest.raises(ValueError, match="requires an X-type logical observable"):
        build_memory_circuit(RotatedSurfaceCode(3), rounds=2, readout_basis="x")

    steane_x = build_memory_circuit(SteaneCode(), rounds=2, readout_basis="x")
    assert steane_x.readout_basis == "x"
    assert steane_x.observables.observables[0].pauli == Pauli(x_qubits=(0, 1, 2))


def test_public_namespace_publishes_the_bands_and_the_registry() -> None:
    import flagquantum.qec as qec

    expected = ("ancilla_bands", "code_names", "get_code", "register_code")
    missing = [name for name in expected if not hasattr(qec, name)]
    assert not missing, f"flagquantum.qec is missing {missing}"
    for name in expected:
        assert name in qec.__all__, f"{name} is not in flagquantum.qec.__all__"
