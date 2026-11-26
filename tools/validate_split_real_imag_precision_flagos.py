"""Compatibility command; implementation lives in its owning tools domain."""

import runpy as _runpy
import sys as _sys
from importlib import import_module as _import_module
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
if __name__ == "__main__":
    _runpy.run_module(
        "tools.validation.validate_split_real_imag_precision_flagos",
        run_name="__main__",
    )
else:
    _implementation = _import_module(
        "tools.validation.validate_split_real_imag_precision_flagos"
    )
    _sys.modules[__name__] = _implementation

    def __getattr__(name):
        return getattr(_implementation, name)
