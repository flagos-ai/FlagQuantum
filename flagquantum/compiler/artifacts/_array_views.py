# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix id/array_{z,xz} and stride_span_{z,x,xz}.
# https://github.com/quantumlib/Qualtran/tree/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/id

"""Static operand view contexts over the single immutable array owner."""

from ._operand_views import (
    _ALLOWED,
    OperandDomain,
    admit_operand_identity,
    format_operand_word,
)
from .operand_array import OperandArray, _compute_common_type
from .operand_identity import OperandIdentity, OperandKind

_HINT_KINDS = {
    **{
        int(kind): frozenset((kind,))
        for kind in OperandKind
        if kind is not OperandKind.REGISTER
    },
    64: _ALLOWED[OperandDomain.X_OPERAND],
    128: _ALLOWED[OperandDomain.Z_OPERAND],
    192: _ALLOWED[OperandDomain.MIXED_OPERAND],
}


def operand_view(array: OperandArray, common_type: int) -> OperandArray:
    """Apply a truthful source constructor hint while retaining storage."""
    if type(common_type) is not int or common_type not in _HINT_KINDS:
        raise ValueError("Unsupported operand array type hint")
    if any(item.kind not in _HINT_KINDS[common_type] for item in array):
        raise ValueError("Operand array type hint contradicts stored identities")
    return array._view(array.offset, array.stride, len(array), common_type)


def default_operand_view() -> OperandArray:
    """Represent an empty default span with zero stride and QUBIT hint."""
    return OperandArray()._view(0, 0, 0, int(OperandKind.QUANTUM))


def default_owned_operands() -> OperandArray:
    """Represent the owning C++ arrays' empty BOOL hint."""
    return operand_view(OperandArray(), int(OperandKind.BOOLEAN))


def copy_operands(array: OperandArray, *, z_only: bool = False) -> OperandArray:
    """Compact a view and recompute the source owner's common type."""
    if z_only:
        for item in array:
            admit_operand_identity(item, OperandDomain.Z_OPERAND)
    copied = OperandArray(iter(array))
    if z_only and not copied:
        return default_owned_operands()
    return copied


def recompute_operand_view(
    array: OperandArray, *, z_only: bool = False
) -> OperandArray:
    """Recompute metadata while retaining offset, stride and owner."""
    if z_only:
        for item in array:
            admit_operand_identity(item, OperandDomain.Z_OPERAND)
    hint = 0 if z_only and not array else _compute_common_type(array)
    return array._view(array.offset, array.stride, len(array), hint)


def operand_array_is_all(
    array: OperandArray,
    target: OperandKind | OperandDomain,
    *,
    classical_only: bool = False,
) -> bool:
    """Query source scalar or union predicates using truthful static hints."""
    allowed: frozenset[OperandKind]
    if isinstance(target, OperandKind) and target is not OperandKind.REGISTER:
        allowed = frozenset((target,))
    elif isinstance(target, OperandDomain):
        allowed = _ALLOWED[target]
    else:
        raise ValueError("Unsupported operand array target family")
    if classical_only:
        allowed = allowed - {OperandKind.QUANTUM}
    if not array:
        return True
    hinted = _HINT_KINDS[array.common_type]
    if hinted <= allowed:
        return True
    if not hinted & allowed:
        return False
    return all(item.kind in allowed for item in array)


def checked_operand_array(
    array: OperandArray, target: OperandKind | OperandDomain, value_name: str
) -> OperandArray:
    """Check a source typed cast and retain the original view and type hint."""
    if not operand_array_is_all(array, target):
        raise ValueError(f"{value_name} contained operands outside {target.name}")
    return array


def cast_operand_payloads(array: OperandArray, target: OperandKind) -> OperandArray:
    """Project source unchecked ID casts into canonical typed identities.

    Boolean memory views are valid only when the addressed byte is zero or one.
    Raw noncanonical Boolean object representations refuse at this boundary.
    """
    if not isinstance(target, OperandKind) or target is OperandKind.REGISTER:
        raise ValueError("Unsupported operand array cast target")
    values = []
    for item in array:
        value = (
            item.physical_word & 255 if target is OperandKind.BOOLEAN else item.value
        )
        if target is OperandKind.X_BOOLEAN:
            value = int(item.physical_word == (3 << 29) | 1)
        values.append(OperandIdentity(target, value))
    return OperandArray(values)


def format_operand_array(
    array: OperandArray, domain: OperandDomain, *, owned: bool = False
) -> str:
    """Render domain-owned diagnostics with native array and view names."""
    if domain not in (OperandDomain.Z_OPERAND, OperandDomain.MIXED_OPERAND):
        raise ValueError("Operand array diagnostics require a Z or mixed domain")
    contents = []
    for item in array:
        admit_operand_identity(item, domain)
        contents.append(format_operand_word(item.physical_word, domain))
    name = (
        "OperandArray"
        if owned and domain is OperandDomain.MIXED_OPERAND
        else "OperandView"
    )
    return name + "{" + ", ".join(contents) + "}"
