# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Numerical payload conversion follows locked Qualtran v0.7.0 sparse_matrix.py.

"""Convert sparse constructor payloads through existing native table ownership."""

from ..algorithms.primitives.sparse_matrix_oracles import ExplicitEntryOracle
from ..compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from ..compiler.formats.legacy_sparse_matrix_factories import (
    EntryContext,
    IndexContext,
    sparse_matrix_factories,
)
from ..compiler.registers.lookup.qrom_table import QromTable
from ..core.value_math import SymbolicInt
from ..errors import SerializationError


def _angle(value: ResolvedArgument) -> float:
    try:
        import numpy as np
    except ModuleNotFoundError as error:
        raise SerializationError("Sparse numerical payloads require NumPy") from error
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SerializationError("Uniform sparse entries require a real numeric value")
    if not -1 <= value <= 1:
        raise SerializationError(
            "Uniform sparse entry must have a finite real rotation"
        )
    return float(2 * np.arccos(value))


def _table(
    width: SymbolicInt, data: ResolvedArgument, precision: SymbolicInt
) -> QromTable:
    try:
        import numpy as np
    except ModuleNotFoundError as error:
        raise SerializationError("Sparse numerical payloads require NumPy") from error
    if not isinstance(data, (np.ndarray, tuple, list)):
        raise SerializationError("Explicit sparse entries require a matrix payload")
    values = np.asarray(data)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise SerializationError(
            "Explicit sparse data must be square and two-dimensional"
        )
    if np.iscomplexobj(values) or not bool(np.all((values >= 0) & (values <= 1))):
        raise SerializationError("Explicit sparse entries must be real and in [0,1]")
    if type(width) is int and values.shape[0] != 2**width:
        raise SerializationError(
            "Explicit sparse matrix dimension does not match index width"
        )
    if type(precision) is int and precision < 1:
        raise SerializationError("Explicit sparse entry precision must be positive")
    if type(width) is int and width > 0 and type(precision) is int:
        return ExplicitEntryOracle(width, tuple(map(tuple, values)), precision).table
    return QromTable(tuple(values.shape), (width, width), (precision,), ((),))


def _sqrt_angle(value: ResolvedArgument) -> float:
    """Preserve the source sqrt then arccos order and default Ry precision."""
    import numpy as np

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SerializationError(
            "Uniform square-root entries require a real numeric value"
        )
    if not 0 <= value <= 1:
        raise SerializationError(
            "Uniform square-root entry must be finite and in [0,1]"
        )
    return float(2 * np.arccos(np.sqrt(value)))


def sparse_matrix_library_factories(
    *,
    index_context: IndexContext | None = None,
    entry_context: EntryContext | None = None,
    adjoint_factory: ComponentFactory | None = None,
) -> dict[str, ComponentFactory]:
    """Keep source objects outside Compiler and preserve exact native oracle roles."""
    return sparse_matrix_factories(
        entry_angle=_angle,
        entry_table=_table,
        sqrt_entry_angle=_sqrt_angle,
        index_context=index_context,
        entry_context=entry_context,
        adjoint_factory=adjoint_factory,
    )
