"""Compatibility imports for :mod:`flagquantum.compiler.lowering.operator_lowering`."""

from .lowering.operator_lowering import (
    BACKENDS as BACKENDS,
)
from .lowering.operator_lowering import (
    DEFAULT_LOWERING_REGISTRY as DEFAULT_LOWERING_REGISTRY,
)
from .lowering.operator_lowering import (
    LoweringCapability as LoweringCapability,
)
from .lowering.operator_lowering import (
    OperatorLoweringRegistry as OperatorLoweringRegistry,
)
from .lowering.operator_lowering import (
    UnsupportedLoweringError as UnsupportedLoweringError,
)
from .lowering.operator_lowering import __all__ as __all__
from .lowering.operator_lowering import (
    require_static_gate_program as require_static_gate_program,
)
from .lowering.operator_lowering import (
    validate_lowering as validate_lowering,
)
