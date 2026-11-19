"""Compatibility alias for :mod:`.frontend.python`."""

import sys

from .frontend import python as _implementation
from .frontend.python import Definition as Definition
from .frontend.python import _annotation_name as _annotation_name
from .frontend.python import _error as _error
from .frontend.python import _Expression as _Expression
from .frontend.python import _QuantumFunction as _QuantumFunction
from .frontend.python import _Source as _Source
from .frontend.python import _source as _source
from .frontend.python import capture as capture
from .frontend.python import capture_named as capture_named
from .frontend.python import qfunc as qfunc

sys.modules[__name__] = _implementation
