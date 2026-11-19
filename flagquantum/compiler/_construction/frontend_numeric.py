"""Compatibility alias for :mod:`.frontend.numeric`."""

import sys

from .frontend import numeric as _implementation
from .frontend.numeric import _OPERATORS as _OPERATORS
from .frontend.numeric import _expression as _expression
from .frontend.numeric import _numeric_input as _numeric_input
from .frontend.numeric import assign_numeric as assign_numeric
from .frontend.numeric import capture_numeric_expression as capture_numeric_expression
from .frontend.numeric import has_numeric as has_numeric
from .frontend.numeric import has_numeric_reads as has_numeric_reads
from .frontend.numeric import needs_numeric as needs_numeric
from .frontend.numeric import reject_numeric_reads as reject_numeric_reads

sys.modules[__name__] = _implementation
