"""Compatibility alias for :mod:`.codegen.boolean`."""

import sys

from .codegen import boolean as _implementation
from .codegen.boolean import _MAX_INPUT_BITS as _MAX_INPUT_BITS
from .codegen.boolean import _MAX_OPERATIONS as _MAX_OPERATIONS
from .codegen.boolean import _MAX_PRODUCTS as _MAX_PRODUCTS
from .codegen.boolean import _MAX_WIDTH as _MAX_WIDTH
from .codegen.boolean import _ONE as _ONE
from .codegen.boolean import _ZERO as _ZERO
from .codegen.boolean import BooleanCandidateError as BooleanCandidateError
from .codegen.boolean import BooleanValueCandidate as BooleanValueCandidate
from .codegen.boolean import _Bits as _Bits
from .codegen.boolean import _disjoint_cubes as _disjoint_cubes
from .codegen.boolean import _Expansion as _Expansion
from .codegen.boolean import _ordered_cubes as _ordered_cubes
from .codegen.boolean import _Poly as _Poly
from .codegen.boolean import lower_boolean_value as lower_boolean_value

sys.modules[__name__] = _implementation
