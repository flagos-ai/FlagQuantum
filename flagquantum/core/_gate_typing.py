"""Runtime base for the statically declared built-in gate surface.

The companion stub declares the methods installed from OperatorSchema.
This runtime class deliberately stays empty so the schema remains the only
implementation source.
"""


class _BuiltinGateMethods:
    """Typing base whose runtime behavior is installed on Circuit."""


__all__ = ("_BuiltinGateMethods",)
