"""
FlagQuantum 电路绘图模块
"""

from .style import available_styles, use_style
from .text_drawer import TextDrawer, draw_text

# mpl 是可选依赖，未安装 matplotlib 时不影响 text 模式
try:
    from .mpl_drawer import MPLDrawer, draw_mpl

    _has_mpl = True
except ImportError:
    _has_mpl = False
    draw_mpl = None  # type: ignore
    MPLDrawer = None  # type: ignore


def draw(qdev, format="text", **kwargs):
    """
    绘制电路图

    Args:
        qdev: FlagQuantum 设备对象（包含 op_history 和 n_wires）
        format: "text" 或 "mpl"
        **kwargs: 其他参数传递给具体的绘图器

    Returns:
        text模式返回字符串，mpl模式返回 (fig, ax)
    Raises:
        ImportError: 当 format="mpl" 但 matplotlib 未安装时
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
