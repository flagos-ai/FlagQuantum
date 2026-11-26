"""Stable public namespace over the responsible implementation owners."""

import math as math
import operator as operator
from collections.abc import Iterable as Iterable
from collections.abc import Sequence as Sequence
from dataclasses import dataclass as dataclass
from importlib import import_module as _import_module
from importlib import import_module as import_module
from numbers import Real as Real
from typing import TYPE_CHECKING
from typing import Any as Any
from typing import cast as cast

import torch as torch

from flagquantum.core.measurement import OutputRequest as OutputRequest
from flagquantum.core.measurement import counts as counts
from flagquantum.core.measurement import density_matrix as density_matrix
from flagquantum.core.measurement import expectation as expectation
from flagquantum.core.measurement import lower_outputs as lower_outputs
from flagquantum.core.measurement import probabilities as probabilities
from flagquantum.core.measurement import samples as samples
from flagquantum.core.measurement import vn_entropy as vn_entropy
from flagquantum.core.operators.observable import _FERMION_EXPORTS, _SUBMODULES
from flagquantum.core.operators.observable import I as I
from flagquantum.core.operators.observable import Observable as Observable
from flagquantum.core.operators.observable import X as X
from flagquantum.core.operators.observable import Y as Y
from flagquantum.core.operators.observable import Z as Z
from flagquantum.core.operators.observable import _PauliTerm as _PauliTerm

from ..core._qubit_aliases import OMITTED as OMITTED
from ..core._qubit_aliases import Omitted as Omitted
from ..core._qubit_aliases import warn_qubit_alias as warn_qubit_alias
from ..core.ir import MeasurementNode as MeasurementNode
from ..errors import CapabilityError as CapabilityError

if TYPE_CHECKING:
    from flagquantum.core.measurement import (
        OutputRequest as OutputRequest,
    )
    from flagquantum.core.measurement import (
        counts as counts,
    )
    from flagquantum.core.measurement import (
        density_matrix as density_matrix,
    )
    from flagquantum.core.measurement import (
        expectation as expectation,
    )
    from flagquantum.core.measurement import (
        lower_outputs as lower_outputs,
    )
    from flagquantum.core.measurement import (
        probabilities as probabilities,
    )
    from flagquantum.core.measurement import (
        samples as samples,
    )
    from flagquantum.core.measurement import (
        vn_entropy as vn_entropy,
    )
    from flagquantum.core.operators.observable import (
        I as I,
    )
    from flagquantum.core.operators.observable import (
        Observable as Observable,
    )
    from flagquantum.core.operators.observable import (
        X as X,
    )
    from flagquantum.core.operators.observable import (
        Y as Y,
    )
    from flagquantum.core.operators.observable import (
        Z as Z,
    )
    from flagquantum.core.operators.observable import (
        _PauliTerm as _PauliTerm,
    )

_OWNERS = {
    "I": "flagquantum.core.operators.observable",
    "Observable": "flagquantum.core.operators.observable",
    "X": "flagquantum.core.operators.observable",
    "Y": "flagquantum.core.operators.observable",
    "Z": "flagquantum.core.operators.observable",
    "OutputRequest": "flagquantum.core.measurement",
    "counts": "flagquantum.core.measurement",
    "density_matrix": "flagquantum.core.measurement",
    "expectation": "flagquantum.core.measurement",
    "lower_outputs": "flagquantum.core.measurement",
    "probabilities": "flagquantum.core.measurement",
    "samples": "flagquantum.core.measurement",
    "vn_entropy": "flagquantum.core.measurement",
}

__all__ = (
    "I",
    "Observable",
    "OutputRequest",
    "X",
    "Y",
    "Z",
    "counts",
    "density_matrix",
    "expectation",
    "probabilities",
    "samples",
    *_SUBMODULES,
    *_FERMION_EXPORTS,
    "vn_entropy",
)


def __getattr__(name: str) -> Any:
    if name.startswith("_"):
        raise AttributeError(name)
    if name in _OWNERS:
        value = getattr(_import_module(_OWNERS[name]), name)
        globals()[name] = value
        return value
    if name in _FERMION_EXPORTS:
        value = getattr(_import_module("flagquantum.core.operators.fermion"), name)
        globals()[name] = value
        return value
    if name in _SUBMODULES:
        value = _import_module(f"{__name__}.{name}")
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Resolve published class annotations in this public module without owner side effects.
