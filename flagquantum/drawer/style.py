"""
Drawing style management
"""

_has_mpl = True
try:
    import matplotlib.pyplot as plt
except (ModuleNotFoundError, ImportError):
    _has_mpl = False
    plt = None


def _needs_mpl(func):
    def wrapper():
        if not _has_mpl:
            raise ImportError(
                "The drawer style module requires matplotlib. "
                "You can install matplotlib via: pip install matplotlib"
            )
        func()

    return wrapper


@_needs_mpl
def _black_white():
    """Black and white style - suitable for printing"""
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["patch.facecolor"] = "white"
    plt.rcParams["patch.edgecolor"] = "black"
    plt.rcParams["patch.linewidth"] = 3.0
    plt.rcParams["patch.force_edgecolor"] = True
    plt.rcParams["lines.color"] = "black"
    plt.rcParams["text.color"] = "black"
    plt.rcParams["lines.linewidth"] = 1.5
    plt.rcParams["path.sketch"] = None


@_needs_mpl
def _black_white_dark():
    """Black and white dark style"""
    almost_black = "#151515"
    plt.rcParams["savefig.facecolor"] = almost_black
    plt.rcParams["figure.facecolor"] = almost_black
    plt.rcParams["axes.facecolor"] = almost_black
    plt.rcParams["patch.edgecolor"] = "white"
    plt.rcParams["patch.facecolor"] = almost_black
    plt.rcParams["patch.force_edgecolor"] = True
    plt.rcParams["lines.color"] = "white"
    plt.rcParams["text.color"] = "white"
    plt.rcParams["path.sketch"] = None


@_needs_mpl
def _sketch():
    """Hand-drawn sketch style"""
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#D6F5E2"
    plt.rcParams["patch.facecolor"] = "#FFEED4"
    plt.rcParams["patch.edgecolor"] = "black"
    plt.rcParams["patch.linewidth"] = 3.0
    plt.rcParams["patch.force_edgecolor"] = True
    plt.rcParams["lines.color"] = "black"
    plt.rcParams["text.color"] = "black"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["path.sketch"] = (1, 100, 2)


@_needs_mpl
def _flagquantum():
    """FlagQuantum signature style"""
    almost_black = "#151515"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#FFB5F1"
    plt.rcParams["patch.facecolor"] = "#D5F0FD"
    plt.rcParams["patch.edgecolor"] = almost_black
    plt.rcParams["patch.linewidth"] = 2.0
    plt.rcParams["patch.force_edgecolor"] = True
    plt.rcParams["lines.color"] = "black"
    plt.rcParams["text.color"] = "black"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["path.sketch"] = None


@_needs_mpl
def _flagquantum_sketch():
    """FlagQuantum hand-drawn sketch style"""
    _flagquantum()
    plt.rcParams["path.sketch"] = (1, 250, 1)


@_needs_mpl
def _sketch_dark():
    """Hand-drawn dark sketch style"""
    almost_black = "#151515"
    plt.rcParams["figure.facecolor"] = almost_black
    plt.rcParams["savefig.facecolor"] = almost_black
    plt.rcParams["axes.facecolor"] = "#EBAAC1"
    plt.rcParams["patch.facecolor"] = "#B0B5DC"
    plt.rcParams["patch.edgecolor"] = "white"
    plt.rcParams["patch.linewidth"] = 3.0
    plt.rcParams["patch.force_edgecolor"] = True
    plt.rcParams["lines.color"] = "white"
    plt.rcParams["text.color"] = "white"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["path.sketch"] = (1, 100, 2)


@_needs_mpl
def _solarized_light():
    """Solarized light theme"""
    plt.rcParams["savefig.facecolor"] = "#fdf6e3"
    plt.rcParams["figure.facecolor"] = "#fdf6e3"
    plt.rcParams["axes.facecolor"] = "#eee8d5"
    plt.rcParams["patch.edgecolor"] = "#93a1a1"
    plt.rcParams["patch.linewidth"] = 3.0
    plt.rcParams["patch.facecolor"] = "#eee8d5"
    plt.rcParams["lines.color"] = "#657b83"
    plt.rcParams["text.color"] = "#586e75"
    plt.rcParams["patch.force_edgecolor"] = True
    plt.rcParams["path.sketch"] = None


@_needs_mpl
def _solarized_dark():
    """Solarized dark theme"""
    plt.rcParams["savefig.facecolor"] = "#002b36"
    plt.rcParams["figure.facecolor"] = "#002b36"
    plt.rcParams["axes.facecolor"] = "#002b36"
    plt.rcParams["patch.edgecolor"] = "#268bd2"
    plt.rcParams["patch.linewidth"] = 3.0
    plt.rcParams["patch.facecolor"] = "#073642"
    plt.rcParams["lines.color"] = "#839496"
    plt.rcParams["text.color"] = "#2aa198"
    plt.rcParams["patch.force_edgecolor"] = True
    plt.rcParams["path.sketch"] = None


# Style mapping table
_STYLES_MAP = {
    "black_white": _black_white,
    "black_white_dark": _black_white_dark,
    "sketch": _sketch,
    "flagquantum": _flagquantum,
    "flagquantum_sketch": _flagquantum_sketch,
    "sketch_dark": _sketch_dark,
    "solarized_light": _solarized_light,
    "solarized_dark": _solarized_dark,
    "default": _needs_mpl(lambda: plt.style.use("default")),
}

_current_style = _black_white


def available_styles():
    """Get all available styles"""
    return tuple(_STYLES_MAP.keys())


def use_style(style: str):
    """Set the global drawing style"""
    global _current_style
    if style in _STYLES_MAP:
        _current_style = _STYLES_MAP[style]
    else:
        raise ValueError(f"Unknown style: {style}. Available: {available_styles()}")


def _apply_style(style: str = None):
    """Apply style (for internal use)"""
    if style is None:
        _current_style()
    elif style in _STYLES_MAP:
        _STYLES_MAP[style]()
    else:
        raise ValueError(f"Unknown style: {style}")
