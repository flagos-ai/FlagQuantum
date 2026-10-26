"""Code-independent stabilizer-code descriptions.

A code declares its qubit layout, its checks, its per-basis ancilla bands, and
its logical observables. The frozen repetition profile keeps its own records;
this module describes a code as a value so that circuit generation, detector
layout, and decoding can be derived from it rather than pinned to one instance.
A record can also be built by name through the family's own registry, so a
caller that knows which code it wants does not have to know which class
declares it.

Two routes reach such a record. A family declares its own layout from a distance
or an index -- the repetition profile, the rotated surface patch, the ZXXZ
surface patch, the Steane code, the triangular colour patch and the
square-lattice torus all do -- and the matrix route instead takes the four
matrices a Calderbank-Shor-Steane code is already written down as, which is how
a code this package does not declare arrives. The two routes meet:
:func:`triangular_colour_code` and :func:`toric_code` derive a family's matrices
and hand back the same matrix record a caller would otherwise have written out
by hand. Every route satisfies the same protocol, so no consumer can tell which
one it was handed.

A check names one ancilla and the CNOT pairs that couple it to the stabilizer's
support, and the direction of each pair is fixed by the factor that qubit
carries. A pure Z-type check therefore uses one direction alone, a pure X-type
check the other, and a mixed X-and-Z check -- which is what every check of the
ZXXZ surface patch is -- uses both directions through the same ancilla.
"""

from __future__ import annotations

import inspect as inspect
import itertools as itertools
from collections.abc import Callable as Callable
from collections.abc import Sequence as Sequence
from dataclasses import dataclass as dataclass
from numbers import Integral as Integral
from typing import TYPE_CHECKING as TYPE_CHECKING
from typing import Any as Any
from typing import Protocol as Protocol
from typing import TypeVar as TypeVar
from typing import cast as cast
from typing import runtime_checkable as runtime_checkable

from ..gf2 import in_span as in_span
from ..gf2 import reduce_rows as reduce_rows
from ..gf2 import reduce_vector as reduce_vector
from ..pauli import Pauli as Pauli

if TYPE_CHECKING:
    # The matrix route's record, which this module's derived families build and
    # which is not imported at run time: it rests on this module's protocol, so a
    # module-level import here would close a cycle between the two.
    from flagquantum.qec.codes.css import CssCode


from ._css_helpers import _DISTANCE_SEARCH_WEIGHT as _DISTANCE_SEARCH_WEIGHT
from ._css_helpers import BinaryMatrix as BinaryMatrix
from ._css_helpers import _block_width as _block_width
from ._css_helpers import _mask as _mask
from ._css_helpers import _matrix_rows as _matrix_rows
from ._css_helpers import _minimum_logical_weight as _minimum_logical_weight
from ._css_helpers import _odd_overlap as _odd_overlap
from ._css_helpers import _refuse_non_logicals as _refuse_non_logicals
from ._css_helpers import _supports as _supports
from .checks import CodeCheck as CodeCheck
from .checks import StabilizerCode as StabilizerCode
from .checks import ancilla_band_counts as ancilla_band_counts
from .checks import ancilla_bands as ancilla_bands
from .colour import triangular_colour_code as triangular_colour_code
from .repetition import RepetitionCode as RepetitionCode
from .steane import _STEANE_CHECK_SUPPORTS as _STEANE_CHECK_SUPPORTS
from .steane import _STEANE_LOGICAL_SUPPORT as _STEANE_LOGICAL_SUPPORT
from .steane import SteaneCode as SteaneCode
from .surface.rotated import RotatedSurfaceCode as RotatedSurfaceCode
from .surface.rotated import _surface_ancilla_sites as _surface_ancilla_sites
from .toric import toric_code as toric_code

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
    "CssCode",
    "CodeCheck",
    "RepetitionCode",
    "RotatedSurfaceCode",
    "StabilizerCode",
    "SteaneCode",
    "ancilla_band_counts",
    "ancilla_bands",
    "code_names",
    "get_code",
    "register_code",
    "toric_code",
    "triangular_colour_code",
)


def __getattr__(name: str) -> Any:
    """Publish the matrix record from its own module, on first use.

    :class:`~flagquantum.qec.CssCode` is built on this module's protocol, so a
    module-level import of it here would close a cycle between the two modules.
    The name is resolved on first use instead, which is what lets a caller import
    it from either home.
    """

    if name == "CssCode":
        from flagquantum.qec.codes.css import CssCode

        return CssCode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
