"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compiler.frontends._structured.frontend.expressions import (
    _Binding as _Binding,
)
from flagquantum.compiler.frontends._structured.frontend.expressions import (
    _call_name as _call_name,
)
from flagquantum.compiler.frontends._structured.frontend.expressions import (
    _TensorView as _TensorView,
)
from flagquantum.compiler.frontends._structured.frontend.expressions import (
    capture_measurement as capture_measurement,
)
from flagquantum.compiler.frontends._structured.frontend.expressions import (
    dynamic_wires as dynamic_wires,
)
from flagquantum.compiler.frontends._structured.frontend.expressions import (
    emit_constant as emit_constant,
)
from flagquantum.compiler.frontends._structured.frontend.expressions import (
    emit_dimension as emit_dimension,
)
from flagquantum.compiler.frontends._structured.frontend.expressions import (
    emit_expression as emit_expression,
)
from flagquantum.compiler.frontends._structured.frontend.expressions import (
    emit_index as emit_index,
)
from flagquantum.compiler.frontends._structured.frontend.expressions import (
    emit_shape_dimension as emit_shape_dimension,
)
from flagquantum.compiler.frontends._structured.frontend.expressions import (
    emit_tensor_extract as emit_tensor_extract,
)
from flagquantum.compiler.frontends._structured.frontend.expressions import (
    literal_non_negative_int as literal_non_negative_int,
)
from flagquantum.compiler.frontends._structured.frontend.expressions import (
    reject_inline_measurement as reject_inline_measurement,
)
from flagquantum.compiler.frontends._structured.frontend.expressions import (
    resolve_tensor_view as resolve_tensor_view,
)
from flagquantum.compiler.frontends._structured.frontend.expressions import (
    static_wires as static_wires,
)
from flagquantum.compiler.frontends._structured.frontend.expressions import (
    wires_node as wires_node,
)

_implementation = _import_module(
    "flagquantum.compiler.frontends._structured.frontend.expressions"
)
_sys.modules[__name__] = _implementation
