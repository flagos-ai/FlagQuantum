# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FlagQuantum circuit drawing module
"""

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


def draw(qdev, format="text", **kwargs):
    """
    Draw a circuit diagram

    Args:
        qdev: FlagQuantum device object (contains op_history and n_wires)
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
        return draw_mpl(qdev, **kwargs)
    return draw_text(qdev, **kwargs)


__all__ = ["TextDrawer", "draw_text", "available_styles", "use_style", "draw"]

if _has_mpl:
    __all__ += ["MPLDrawer", "draw_mpl"]
