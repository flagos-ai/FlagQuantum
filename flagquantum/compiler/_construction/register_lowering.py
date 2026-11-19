"""Compatibility alias for :mod:`.codegen.registers`."""

import sys

from .codegen import registers as _implementation
from .codegen.registers import RegisterCandidate as RegisterCandidate
from .codegen.registers import _array_controlled_x as _array_controlled_x
from .codegen.registers import _bounded_control_group as _bounded_control_group
from .codegen.registers import _modular_exp_instructions as _modular_exp_instructions
from .codegen.registers import _pattern_gate as _pattern_gate
from .codegen.registers import (
    _real_transform_instructions as _real_transform_instructions,
)
from .codegen.registers import _type1_instructions as _type1_instructions
from .codegen.registers import _type2_instructions as _type2_instructions
from .codegen.registers import _uniform_interval as _uniform_interval
from .codegen.registers import schedule_register_program as schedule_register_program

sys.modules[__name__] = _implementation
