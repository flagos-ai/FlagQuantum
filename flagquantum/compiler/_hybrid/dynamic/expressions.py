"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compiler.frontends._structured.dynamic.expressions import (
    _FALSE_CLAUSES as _FALSE_CLAUSES,
)
from flagquantum.compiler.frontends._structured.dynamic.expressions import (
    _TRUE_CLAUSES as _TRUE_CLAUSES,
)
from flagquantum.compiler.frontends._structured.dynamic.expressions import (
    _ClassicalPredicate as _ClassicalPredicate,
)
from flagquantum.compiler.frontends._structured.dynamic.expressions import (
    _ConditionalValue as _ConditionalValue,
)
from flagquantum.compiler.frontends._structured.dynamic.expressions import (
    _MeasurementRef as _MeasurementRef,
)
from flagquantum.compiler.frontends._structured.dynamic.expressions import (
    and_clauses as and_clauses,
)
from flagquantum.compiler.frontends._structured.dynamic.expressions import (
    as_predicate as as_predicate,
)
from flagquantum.compiler.frontends._structured.dynamic.expressions import (
    canonicalize_clauses as canonicalize_clauses,
)
from flagquantum.compiler.frontends._structured.dynamic.expressions import (
    compare_predicate as compare_predicate,
)
from flagquantum.compiler.frontends._structured.dynamic.expressions import (
    compare_values as compare_values,
)
from flagquantum.compiler.frontends._structured.dynamic.expressions import (
    conditional_product as conditional_product,
)
from flagquantum.compiler.frontends._structured.dynamic.expressions import (
    conjoin_predicates as conjoin_predicates,
)
from flagquantum.compiler.frontends._structured.dynamic.expressions import (
    disjoin_predicates as disjoin_predicates,
)
from flagquantum.compiler.frontends._structured.dynamic.expressions import (
    map_conditional_values as map_conditional_values,
)
from flagquantum.compiler.frontends._structured.dynamic.expressions import (
    merge_conditional_values as merge_conditional_values,
)
from flagquantum.compiler.frontends._structured.dynamic.expressions import (
    negate_predicate as negate_predicate,
)
from flagquantum.compiler.frontends._structured.dynamic.expressions import (
    not_clauses as not_clauses,
)
from flagquantum.compiler.frontends._structured.dynamic.expressions import (
    or_clauses as or_clauses,
)
from flagquantum.compiler.frontends._structured.dynamic.expressions import (
    predicate_clauses as predicate_clauses,
)
from flagquantum.compiler.frontends._structured.dynamic.expressions import (
    same_runtime_value as same_runtime_value,
)

_implementation = _import_module(
    "flagquantum.compiler.frontends._structured.dynamic.expressions"
)
_sys.modules[__name__] = _implementation
