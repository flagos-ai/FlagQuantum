"""Historical import facade for the relocated implementation."""

from importlib import import_module as _import_module

from flagquantum.algorithms.optimization._research_schemas import (
    ResearchSchemaDocument as ResearchSchemaDocument,
)
from flagquantum.algorithms.optimization._research_schemas import __all__ as __all__
from flagquantum.algorithms.optimization._research_schemas import (
    describe_research_schema as describe_research_schema,
)
from flagquantum.algorithms.optimization._research_schemas import (
    list_research_schemas as list_research_schemas,
)

_implementation = _import_module(
    "flagquantum.algorithms.optimization._research_schemas"
)
