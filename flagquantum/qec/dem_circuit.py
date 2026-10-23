"""Historical imports for :mod:`flagquantum.qec.dem.circuit`; Core-owned compatibility boundary."""

from __future__ import annotations

import math as math
from collections.abc import Iterable as Iterable
from collections.abc import Sequence as Sequence
from numbers import Integral as Integral
from numbers import Real as Real
from typing import TYPE_CHECKING as TYPE_CHECKING
from typing import Any as Any

import torch as torch

from flagquantum.core.ir import CircuitIR as CircuitIR
from flagquantum.core.ir import Instruction as Instruction
from flagquantum.core.ir import ensure_circuit_ir as ensure_circuit_ir
from flagquantum.core.operator_schema import canonical_opcode as canonical_opcode
from flagquantum.qec.dem.alternatives import fault_groups as fault_groups
from flagquantum.qec.dem.circuit import _BIT_FLIP as _BIT_FLIP
from flagquantum.qec.dem.circuit import _ERROR_ID_KEY as _ERROR_ID_KEY
from flagquantum.qec.dem.circuit import _FRAME_OPCODE as _FRAME_OPCODE
from flagquantum.qec.dem.circuit import _IDENTITY as _IDENTITY
from flagquantum.qec.dem.circuit import _MASS_TOLERANCE as _MASS_TOLERANCE
from flagquantum.qec.dem.circuit import _MEASURE_OPCODE as _MEASURE_OPCODE
from flagquantum.qec.dem.circuit import _OPERATOR_TOLERANCE as _OPERATOR_TOLERANCE
from flagquantum.qec.dem.circuit import _PROBABILITIES_KEY as _PROBABILITIES_KEY
from flagquantum.qec.dem.circuit import __all__ as __all__
from flagquantum.qec.dem.circuit import _count as _count
from flagquantum.qec.dem.circuit import (
    _declared_probabilities as _declared_probabilities,
)
from flagquantum.qec.dem.circuit import _error_id as _error_id
from flagquantum.qec.dem.circuit import _frame as _frame
from flagquantum.qec.dem.circuit import _frame_support as _frame_support
from flagquantum.qec.dem.circuit import _mechanisms as _mechanisms
from flagquantum.qec.dem.circuit import _operators as _operators
from flagquantum.qec.dem.circuit import (
    _require_every_mechanism_fires as _require_every_mechanism_fires,
)
from flagquantum.qec.dem.circuit import _scale as _scale
from flagquantum.qec.dem.circuit import _support as _support
from flagquantum.qec.dem.circuit import (
    circuit_from_detector_error_model as circuit_from_detector_error_model,
)
from flagquantum.qec.dem.circuit import (
    detector_error_model_from_circuit as detector_error_model_from_circuit,
)
