"""Compatibility alias for :mod:`.frontend.ports`."""

import sys

from .frontend import ports as _implementation
from .frontend.ports import _annotation_node as _annotation_node
from .frontend.ports import _ClassicalParameter as _ClassicalParameter
from .frontend.ports import _Port as _Port
from .frontend.ports import _resolve as _resolve
from .frontend.ports import classical_parameters as classical_parameters
from .frontend.ports import program_annotation_names as program_annotation_names
from .frontend.ports import program_ports as program_ports
from .frontend.ports import quantum_port as quantum_port

sys.modules[__name__] = _implementation
