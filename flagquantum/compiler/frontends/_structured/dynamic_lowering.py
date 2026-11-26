"""Retained private entry points for bounded measurement-feedback lowering."""

from flagquantum._import_compat import alias_module as _alias_module
from flagquantum.compiler.frontends._structured.dynamic.program import (
    LoweredDynamicProgram,
    lower_dynamic_program,
)

__all__ = ("LoweredDynamicProgram", "lower_dynamic_program")

_alias_module("flagquantum.compiler._hybrid.dynamic_lowering", __name__)
