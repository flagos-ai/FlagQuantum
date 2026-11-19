"""Compatibility alias for :mod:`.codegen.arithmetic`."""

import sys

from .codegen import arithmetic as _implementation
from .codegen.arithmetic import (
    _NORMALIZATION_PROBE_MAX_OPERATIONS as _NORMALIZATION_PROBE_MAX_OPERATIONS,
)
from .codegen.arithmetic import ArithmeticCandidate as ArithmeticCandidate
from .codegen.arithmetic import BoundArithmetic as BoundArithmetic
from .codegen.arithmetic import Schedule as Schedule
from .codegen.arithmetic import ScheduleEvent as ScheduleEvent
from .codegen.arithmetic import _Allocation as _Allocation
from .codegen.arithmetic import _bitwise as _bitwise
from .codegen.arithmetic import _copy as _copy
from .codegen.arithmetic import _emit_arithmetic as _emit_arithmetic
from .codegen.arithmetic import _logical_cost as _logical_cost
from .codegen.arithmetic import (
    _schedule_normalized_arithmetic as _schedule_normalized_arithmetic,
)
from .codegen.arithmetic import _self_adjoint_gates as _self_adjoint_gates
from .codegen.arithmetic import _WireValue as _WireValue
from .codegen.arithmetic import adjoint_value_candidate as adjoint_value_candidate
from .codegen.arithmetic import lower_arithmetic as lower_arithmetic
from .codegen.arithmetic import schedule_arithmetic as schedule_arithmetic

sys.modules[__name__] = _implementation
