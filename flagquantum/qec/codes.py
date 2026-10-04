"""Code-independent stabilizer-code descriptions.

A code declares its qubit layout, its checks, its per-basis ancilla bands, and
its logical observables. The frozen repetition profile keeps its own records;
this module describes a code as a value so that circuit generation, detector
layout, and decoding can be derived from it rather than pinned to one instance.
A record can also be built by name through the family's own registry, so a
caller that knows which code it wants does not have to know which class
declares it.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

from .pauli import Pauli


def ancilla_bands(
    checks: tuple[CodeCheck, ...] | list[CodeCheck],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split a code's check ancillas into the X band and the Z band.

    A check's ancilla measures one basis, which its stabilizer's type fixes, so
    the two bands are a function of the checks and are derived here rather than
    declared beside them. That is deliberate: a record that stated its bands as
    well as its checks could state two answers, and the drift would be silent
    because every consumer reads one of them. A record that derives them cannot.

    An ancilla that measures neither basis -- a flag or an idle ancilla -- is in
    neither band, so the two bands need not cover a record's declared ancillas.
    Both bands are returned in ascending qubit order, which is the order a
    detector band and a schedule are read in.

    Args:
        checks: The checks whose ancillas are being split.

    Returns:
        The ``(x_qubits, z_qubits)`` pair, each sorted and without duplicates.
    """

    x_qubits = tuple(
        sorted({check.ancilla_qubit for check in checks if check.stabilizer.x_qubits})
    )
    z_qubits = tuple(
        sorted({check.ancilla_qubit for check in checks if check.stabilizer.z_qubits})
    )
    overlap = set(x_qubits) & set(z_qubits)
    if overlap:
        raise ValueError(
            "ancillas "
            + ", ".join(str(qubit) for qubit in sorted(overlap))
            + " measure a check of each basis, so they belong to neither band: one "
            "ancilla measures one basis"
        )
    return x_qubits, z_qubits


@dataclass(frozen=True)
class CodeCheck:
    """One stabilizer check with its ancilla and its CNOT coupling.

    Each entry of ``cnot_qubits`` is a ``(control, target)`` pair, and which qubit
    is which is fixed by the check type rather than left to the caller. A
    Z-type check couples every data qubit in the stabilizer's support into an
    ancilla prepared in ``|0>``, so the data qubit controls and the ancilla is the target
    the target. An X-type check couples the ancilla out into the same support
    with the ancilla prepared in ``|+>``, the ancilla controls and the data qubit is the target. Both gadgets leave the ancilla's Z-basis readout equal
    to the check's eigenvalue, which is what makes the two symmetric here.
    """

    index: int
    stabilizer: Pauli
    ancilla_qubit: int
    cnot_qubits: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("check index must be non-negative")
        if not isinstance(self.stabilizer, Pauli):
            raise TypeError("check stabilizer must be a Pauli operator")
        if self.stabilizer.x_qubits and self.stabilizer.z_qubits:
            raise ValueError(
                "check stabilizer must be pure X-type or pure Z-type, not a "
                "mixture: a mixture needs a second ancilla and a second CNOT "
                "direction, which this record does not describe"
            )
        if self.ancilla_qubit < 0:
            raise ValueError("check ancilla qubit must be non-negative")
        if not self.cnot_qubits:
            raise ValueError("check must declare at least one CNOT")
        for control, target in self.cnot_qubits:
            if control < 0 or target < 0:
                raise ValueError("check CNOT qubits must be non-negative")
        if self.stabilizer.x_qubits:
            self._validate_x_type()
        else:
            self._validate_z_type()

    def _validate_z_type(self) -> None:
        if any(control == self.ancilla_qubit for control, _ in self.cnot_qubits):
            raise ValueError("check CNOTs must control data qubits, not the ancilla")
        if any(target != self.ancilla_qubit for _, target in self.cnot_qubits):
            raise ValueError("check CNOTs must target the declared ancilla")
        controls = tuple(sorted(control for control, _ in self.cnot_qubits))
        if controls != self.stabilizer.support:
            raise ValueError("check CNOT controls must match the stabilizer support")

    def _validate_x_type(self) -> None:
        if any(target == self.ancilla_qubit for _, target in self.cnot_qubits):
            raise ValueError("check CNOTs must target data qubits, not the ancilla")
        if any(control != self.ancilla_qubit for control, _ in self.cnot_qubits):
            raise ValueError("check CNOTs must be controlled by the declared ancilla")
        targets = tuple(sorted(target for _, target in self.cnot_qubits))
        if targets != self.stabilizer.support:
            raise ValueError("check CNOT targets must match the stabilizer support")


@runtime_checkable
class StabilizerCode(Protocol):
    """A code that declares its qubit layout, checks, and logical observables.

    The ancilla count is stated twice on purpose: once as a total, and once per
    basis. Upstream's code record carries the same three accessors as pure
    virtuals -- ``get_num_ancilla_qubits``, ``get_num_ancilla_x_qubits`` and
    ``get_num_ancilla_z_qubits`` -- and the split is what a syndrome-extraction
    round and a detector band are laid out against, so a total alone does not say
    which ancillas a round has to visit. :func:`ancilla_bands` derives the two
    bands from ``checks``, and the records in this module read their counts from
    it, so a record cannot report a split its own checks contradict.
    """

    @property
    def distance(self) -> int: ...

    @property
    def num_data_qubits(self) -> int: ...

    @property
    def num_ancilla_qubits(self) -> int: ...

    @property
    def num_ancilla_x_qubits(self) -> int: ...

    @property
    def num_ancilla_z_qubits(self) -> int: ...

    @property
    def data_qubits(self) -> tuple[int, ...]: ...

    @property
    def ancilla_qubits(self) -> tuple[int, ...]: ...

    @property
    def checks(self) -> tuple[CodeCheck, ...]: ...

    @property
    def stabilizers(self) -> tuple[Pauli, ...]: ...

    @property
    def logical_observables(self) -> tuple[Pauli, ...]: ...


@dataclass(frozen=True)
class RepetitionCode:
    """The bit-flip repetition code with ``distance`` data qubits.

    Data qubits occupy qubits ``0..distance-1`` and check ancillas occupy qubits
    ``distance..2*distance-2``. Check ``c`` measures ``Z_c Z_{c+1}``, so the code
    detects bit flips. The single declared logical observable is ``Z`` on every
    data qubit, which makes its readout representative the parity of the terminal
    data measurements.
    """

    distance: int = 3

    def __post_init__(self) -> None:
        if isinstance(self.distance, bool) or not isinstance(self.distance, Integral):
            raise TypeError("repetition-code distance must be an integer")
        if self.distance < 2:
            raise ValueError("repetition-code distance must be at least two")

    @property
    def num_data_qubits(self) -> int:
        return self.distance

    @property
    def num_ancilla_qubits(self) -> int:
        return self.distance - 1

    @property
    def num_ancilla_x_qubits(self) -> int:
        return len(ancilla_bands(self.checks)[0])

    @property
    def num_ancilla_z_qubits(self) -> int:
        return len(ancilla_bands(self.checks)[1])

    @property
    def data_qubits(self) -> tuple[int, ...]:
        return tuple(range(self.distance))

    @property
    def ancilla_qubits(self) -> tuple[int, ...]:
        return tuple(range(self.distance, 2 * self.distance - 1))

    @property
    def checks(self) -> tuple[CodeCheck, ...]:
        return tuple(
            CodeCheck(
                index=index,
                stabilizer=Pauli(z_qubits=(index, index + 1)),
                ancilla_qubit=self.distance + index,
                cnot_qubits=(
                    (index, self.distance + index),
                    (index + 1, self.distance + index),
                ),
            )
            for index in range(self.distance - 1)
        )

    @property
    def stabilizers(self) -> tuple[Pauli, ...]:
        return tuple(check.stabilizer for check in self.checks)

    @property
    def logical_observables(self) -> tuple[Pauli, ...]:
        return (Pauli(z_qubits=self.data_qubits),)


def _surface_ancilla_sites(distance: int) -> tuple[tuple[int, int, bool], ...]:
    """Return the rotated-surface-code ancilla sites as ``(a, b, is_x)``.

    Data qubits sit on the lattice ``(i, j)`` with ``0 <= i, j <= distance - 1``
    and an ancilla sits on ``(a, b)`` with ``0 <= a, b <= distance``. An X-type
    ancilla needs an interior column and an odd lattice parity, a Z-type ancilla
    needs an interior row and an even lattice parity, which is the stabilizer
    assignment the rotated layout requires: the parity that would place an
    ancilla outside the patch is what truncates the boundary checks.

    Sites are reported in lattice order so the check and ancilla indices are
    reproducible.
    """

    sites: list[tuple[int, int, bool]] = []
    for a in range(distance + 1):
        for b in range(distance + 1):
            if 1 <= a <= distance - 1 and (a + b) % 2 == 1:
                sites.append((a, b, True))
            elif 1 <= b <= distance - 1 and (a + b) % 2 == 0:
                sites.append((a, b, False))
    return tuple(sites)


@dataclass(frozen=True)
class RotatedSurfaceCode:
    """The rotated surface code of odd ``distance`` read out as Z memory.

    Data qubits occupy qubits ``0..distance**2 - 1``, indexed so that lattice
    site ``(i, j)`` is qubit ``j * distance + i``. Ancillas follow the data qubits
    in lattice order. ``(i, j)`` is a data qubit for ``0 <= i, j <= distance - 1``
    and ``(a, b)`` is an ancilla for ``0 <= a, b <= distance``; an X-type ancilla
    sits on an interior column with odd lattice parity, a Z-type ancilla on an
    interior row with even lattice parity, and its support is the up to four data
    qubits diagonally adjacent to it. That assignment is the rotated layout: a
    check that would fall outside the patch is truncated, which is what leaves
    the boundary checks at weight two.

    The declared logical observable is ``Z`` on the data row ``j == 0``, which is
    one complete row of the patch, and the two logical operators of the patch
    therefore read out as Z memory. The X-type checks are still measured every
    round, because they detect the Z errors this readout is vulnerable to; they
    are simply not deterministic in the initial all-zero state.

    This record describes the checks and the observable. It does not choose a
    decoding strategy, a noise model, or an error-correction threshold.
    """

    distance: int = 3

    def __post_init__(self) -> None:
        if isinstance(self.distance, bool) or not isinstance(self.distance, Integral):
            raise TypeError("rotated-surface-code distance must be an integer")
        if self.distance < 2:
            raise ValueError("rotated-surface-code distance must be at least two")

    @property
    def num_data_qubits(self) -> int:
        return self.distance * self.distance

    @property
    def num_ancilla_qubits(self) -> int:
        return len(_surface_ancilla_sites(self.distance))

    @property
    def num_ancilla_x_qubits(self) -> int:
        return len(ancilla_bands(self.checks)[0])

    @property
    def num_ancilla_z_qubits(self) -> int:
        return len(ancilla_bands(self.checks)[1])

    @property
    def data_qubits(self) -> tuple[int, ...]:
        return tuple(range(self.num_data_qubits))

    @property
    def ancilla_qubits(self) -> tuple[int, ...]:
        first = self.num_data_qubits
        return tuple(first + index for index in range(self.num_ancilla_qubits))

    @property
    def checks(self) -> tuple[CodeCheck, ...]:
        distance = self.distance
        checks: list[CodeCheck] = []
        for index, (a, b, is_x) in enumerate(_surface_ancilla_sites(distance)):
            ancilla = self.num_data_qubits + index
            support = tuple(
                sorted(
                    j * distance + i
                    for i in range(distance)
                    for j in range(distance)
                    if i in (a - 1, a) and j in (b - 1, b)
                )
            )
            if is_x:
                stabilizer = Pauli(x_qubits=support)
                cnot_qubits = tuple((ancilla, wire) for wire in support)
            else:
                stabilizer = Pauli(z_qubits=support)
                cnot_qubits = tuple((wire, ancilla) for wire in support)
            checks.append(
                CodeCheck(
                    index=index,
                    stabilizer=stabilizer,
                    ancilla_qubit=ancilla,
                    cnot_qubits=cnot_qubits,
                )
            )
        return tuple(checks)

    @property
    def stabilizers(self) -> tuple[Pauli, ...]:
        return tuple(check.stabilizer for check in self.checks)

    @property
    def logical_observables(self) -> tuple[Pauli, ...]:
        return (Pauli(z_qubits=tuple(range(self.distance))),)


_STEANE_CHECK_SUPPORTS: tuple[tuple[int, ...], ...] = (
    (3, 4, 5, 6),
    (1, 2, 5, 6),
    (0, 2, 4, 6),
)
_STEANE_LOGICAL_SUPPORT: tuple[int, ...] = (0, 1, 2)


@dataclass(frozen=True)
class SteaneCode:
    """The seven-qubit Steane code, declared by its checks and its logicals.

    Data qubits occupy qubits ``0..6`` and the six check ancillas follow them on
    qubits ``7..12``: checks ``0..2`` are Z-type and checks ``3..5`` are X-type,
    each with the same support, which is what makes the code Calderbank-Shor-
    Steane. The supports are the three non-zero parity constraints of the
    ``[7, 4, 3]`` Hamming code, so each check has weight four and the code has
    distance three.

    Both logical operators are declared, on the same three data qubits: ``Z`` on
    ``(0, 1, 2)`` and ``X`` on ``(0, 1, 2)``. They anticommute, since they overlap
    so an odd number of qubits flips the is a product of checks. This is the
    smallest code this package declares whose two fault families are both
    non-trivial, which is why it is the record the matrix route is exercised on.

    The record states the checks and the observables. It does not choose a
    decoding strategy or a noise model, and it declines to describe a
    preparation, so a memory circuit built from it is Z memory like every other
    record here.
    """

    @property
    def distance(self) -> int:
        return 3

    @property
    def num_data_qubits(self) -> int:
        return 7

    @property
    def num_ancilla_qubits(self) -> int:
        return 6

    @property
    def num_ancilla_x_qubits(self) -> int:
        return len(ancilla_bands(self.checks)[0])

    @property
    def num_ancilla_z_qubits(self) -> int:
        return len(ancilla_bands(self.checks)[1])

    @property
    def data_qubits(self) -> tuple[int, ...]:
        return tuple(range(7))

    @property
    def ancilla_qubits(self) -> tuple[int, ...]:
        return tuple(range(7, 13))

    @property
    def checks(self) -> tuple[CodeCheck, ...]:
        checks: list[CodeCheck] = []
        for index, support in enumerate(_STEANE_CHECK_SUPPORTS):
            ancilla = 7 + index
            checks.append(
                CodeCheck(
                    index=index,
                    stabilizer=Pauli(z_qubits=support),
                    ancilla_qubit=ancilla,
                    cnot_qubits=tuple((wire, ancilla) for wire in support),
                )
            )
        for offset, support in enumerate(_STEANE_CHECK_SUPPORTS):
            index = len(_STEANE_CHECK_SUPPORTS) + offset
            ancilla = 7 + index
            checks.append(
                CodeCheck(
                    index=index,
                    stabilizer=Pauli(x_qubits=support),
                    ancilla_qubit=ancilla,
                    cnot_qubits=tuple((ancilla, wire) for wire in support),
                )
            )
        return tuple(checks)

    @property
    def stabilizers(self) -> tuple[Pauli, ...]:
        return tuple(check.stabilizer for check in self.checks)

    @property
    def logical_observables(self) -> tuple[Pauli, ...]:
        return (
            Pauli(z_qubits=_STEANE_LOGICAL_SUPPORT),
            Pauli(x_qubits=_STEANE_LOGICAL_SUPPORT),
        )


# ---------------------------------------------------------------------------
# The code registry
# ---------------------------------------------------------------------------

#: The members a record has to carry to be registered, read off the protocol
#: rather than restated: the protocol is what a consumer holds a record to, and a
#: second hand-written list here would be a second statement of the same
#: contract, free to drift from the one callers are type-checked against.
_STABILIZER_CODE_MEMBERS: tuple[str, ...] = tuple(
    sorted(name for name in vars(StabilizerCode) if not name.startswith("_"))
)

#: One name per registered record. The value is the class and not an instance,
#: because the options are the caller's: ``get_code`` builds the record, so a
#: registered instance would have frozen a distance nobody chose.
_CODES: dict[str, type[StabilizerCode]] = {}

_CodeT = TypeVar("_CodeT")


def register_code(name: str, *, replace: bool = False) -> Callable[[_CodeT], _CodeT]:
    """Name a stabilizer-code record in the registry.

    Registration is what lets a caller reach a code by name instead of by class,
    which is the shape upstream's own factory has: ``get_code(name, options)``
    builds a code and ``get_available_codes()`` lists the names. The class is
    registered and returned unchanged, so a registered record keeps its name and
    its module and stays directly constructible.

    Args:
        name: The name :func:`get_code` will build this record from. It must be a
            non-empty string with no whitespace in it, so that a name is one
            token and can be printed in a refusal.
        replace: Whether to overwrite an existing registration. Replacing is
            refused by default, for the same reason it is refused for a decoder:
            two records answering to one name is a choice the registry cannot
            make on the caller's behalf, and the second registration is usually a
            typo rather than an intent.

    Returns:
        The decorator that registers a class and returns it unchanged.

    Raises:
        TypeError: The name is not a string, or the registered object does not
            carry every member of the :class:`StabilizerCode` protocol.
        ValueError: The name is empty or carries whitespace, or it is already
            registered and ``replace`` is not set.
    """

    if not isinstance(name, str):
        raise TypeError("a code name must be a string")
    if not name or name != name.strip() or any(char.isspace() for char in name):
        raise ValueError(
            f"a code name must be non-empty and carry no whitespace, not {name!r}"
        )
    if name in _CODES and not replace:
        raise ValueError(
            f"the name {name!r} is already registered against "
            f"{getattr(_CODES[name], '__name__', _CODES[name])!r}; pass "
            "replace=True to overwrite it"
        )

    def decorate(candidate: _CodeT) -> _CodeT:
        missing = [
            member
            for member in _STABILIZER_CODE_MEMBERS
            if not hasattr(candidate, member)
        ]
        if missing:
            raise TypeError(
                f"{getattr(candidate, '__name__', candidate)!r} cannot be registered "
                f"as {name!r}: a stabilizer-code record declares "
                + ", ".join(_STABILIZER_CODE_MEMBERS)
                + f", and {', '.join(missing)} "
                + ("is" if len(missing) == 1 else "are")
                + " missing"
            )
        # The members were just checked, which is the only form of the protocol
        # check a class object admits: isinstance reads an instance, and no
        # instance exists until a caller asks for one.
        _CODES[name] = cast("type[StabilizerCode]", candidate)
        return candidate

    return decorate


def code_names() -> tuple[str, ...]:
    """Return the registered code names, sorted, so a refusal can list them."""

    return tuple(sorted(_CODES))


def get_code(name: str, **options: Any) -> StabilizerCode:
    """Build the stabilizer-code record registered under ``name``.

    The options are the fields the record itself declares, forwarded to its
    constructor, so a name reaches the same record a direct construction would
    and there is no second place a distance or a patch size can be set. A name
    whose record takes no fields is built with no options, which is what makes
    ``get_code("steane")`` and ``get_code("rotated_surface", distance=5)`` the
    same call shape.

    Args:
        name: A name from :func:`code_names`.
        **options: Field values for the record, checked against the record's own
            fields before the constructor runs so an unknown field is a refusal
            that lists the ones it takes rather than a traceback from inside it.
            A value the record refuses is refused by the record, with the
            message the record states.

    Returns:
        The record, ready for :func:`flagquantum.qec.build_memory_circuit`.

    Raises:
        ValueError: The name is not registered.
        TypeError: The name is not a string, or an option is not a field of the
            record the name selects.

    Note:
        Upstream's factory also has an overload that builds a code from a list of
        stabilizers the caller supplies. That overload is not here, because a
        record built from arbitrary stabilizers has to decide each check's
        ancilla and coupling direction on the caller's behalf, and a code whose
        generator matrix is known is already served by
        :meth:`flagquantum.qec.DetectorErrorModel.from_code_matrices`, which
        reads the matrix rather than inventing a gadget for it.
    """

    try:
        record = _CODES[name]
    except KeyError:
        known = ", ".join(code_names())
        raise ValueError(
            f"no code is registered as {name!r}; the registry holds {known}"
        ) from None
    fields = [
        parameter
        for parameter in inspect.signature(record).parameters.values()
        if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
        and parameter.name != "self"
    ]
    accepted = {parameter.name for parameter in fields}
    unknown = sorted(set(options) - accepted)
    if unknown:
        raise TypeError(
            f"{getattr(record, '__name__', record)!r} takes no option "
            + ", ".join(repr(field) for field in unknown)
            + "; it takes "
            + (", ".join(sorted(accepted)) if accepted else "no options")
        )
    return record(**options)


register_code("repetition")(RepetitionCode)
register_code("rotated_surface")(RotatedSurfaceCode)
register_code("steane")(SteaneCode)


__all__ = (
    "CodeCheck",
    "RepetitionCode",
    "RotatedSurfaceCode",
    "SteaneCode",
    "StabilizerCode",
    "ancilla_bands",
    "code_names",
    "get_code",
    "register_code",
)
