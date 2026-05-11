"""
FlagQuantum 电路绘图模块
"""

from .mpl_drawer import draw_mpl
from .style import available_styles, use_style
from .text_drawer import draw_text


def draw(qdev, format="text", **kwargs):
    """
    绘制电路图

    Args:
        qdev: FlagQuantum 设备对象（包含 op_history 和 n_wires）
        format: "text" 或 "mpl"
        **kwargs: 其他参数传递给具体的绘图器

    Returns:
        text模式返回字符串，mpl模式返回 (fig, ax)
    """
    if format == "mpl":
        return draw_mpl(qdev, **kwargs)
    else:
        return draw_text(qdev, **kwargs)


__all__ = ["draw_text", "draw_mpl", "available_styles", "use_style", "draw"]
