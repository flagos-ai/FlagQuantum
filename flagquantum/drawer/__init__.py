"""
FlagQuantum circuit drawing module
"""

from .ir_adapter import DrawableCircuit, to_drawable_circuit
from .style import available_styles, use_style
from .text_drawer import TextDrawer, draw_text

# mpl is an optional dependency; text mode works fine without matplotlib
try:
    from .mpl_drawer import MPLDrawer, draw_mpl

    _has_mpl = True
except ImportError:
    _has_mpl = False
    draw_mpl = None  # type: ignore
    MPLDrawer = None  # type: ignore


def draw(program, format="text", **kwargs):
    """
    Draw a circuit diagram

    Args:
        program: FlagQuantum Circuit, CircuitIR, or device object
        format: "text" or "mpl"
        **kwargs: Additional arguments passed to the specific drawer

    Returns:
        For text mode: returns a string; for mpl mode: returns (fig, ax)
    Raises:
        ImportError: When format="mpl" is used but matplotlib is not installed
    """
    if format == "mpl":
        if not _has_mpl:
            raise ImportError(
                "matplotlib is required for `format='mpl'`. "
                "Install it with: pip install flagquantum[viz]"
            )
        return draw_mpl(program, **kwargs)
    return draw_text(program, **kwargs)


__all__ = [
    "DrawableCircuit",
    "TextDrawer",
    "draw_text",
    "to_drawable_circuit",
    "available_styles",
    "use_style",
    "draw",
]

if _has_mpl:
    __all__ += ["MPLDrawer", "draw_mpl"]
