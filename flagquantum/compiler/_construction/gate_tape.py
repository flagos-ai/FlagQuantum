"""Compatibility alias for :mod:`.codegen.gate_tape`."""

import sys

from .codegen import gate_tape as _implementation
from .codegen.gate_tape import _ANGLED as _ANGLED
from .codegen.gate_tape import _ARITY as _ARITY
from .codegen.gate_tape import _MAX_OPERATIONS as _MAX_OPERATIONS
from .codegen.gate_tape import GateRow as GateRow
from .codegen.gate_tape import GateTemplate as GateTemplate
from .codegen.gate_tape import check_gate_tape as check_gate_tape
from .codegen.gate_tape import expand_gate_tape as expand_gate_tape
from .codegen.gate_tape import pack_gate_templates as pack_gate_templates

sys.modules[__name__] = _implementation
