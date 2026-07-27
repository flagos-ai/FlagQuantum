"""Generate dependency-free SVG figures for end-to-end VQE training runs."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Callable

WIDTH, HEIGHT = 1800, 980
COLORS = {2: "#3478C5", 4: "#D07124", 8: "#54319B"}
LIGHT_COLORS = {2: "#9CC5EA", 4: "#F2BE8B", 8: "#B9A8E5"}


def _header(title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#F8FAFC"/>',
        '<rect x="24" y="24" width="1752" height="932" rx="20" fill="#FFFFFF" stroke="#E2E8F0"/>',
        '<rect x="24" y="24" width="8" height="104" rx="4" fill="#54319B"/>',
        '<style>text{font-family:Inter,"PingFang SC","Microsoft YaHei",Arial,sans-serif;fill:#334155}.tag{font-size:12px;font-weight:700;fill:#54319B;letter-spacing:1px}.title{font-size:29px;font-weight:750;fill:#172B4D}.sub{font-size:13px;fill:#64748B}.panel{font-size:17px;font-weight:700;fill:#172B4D}.axis{font-size:11px;fill:#64748B}.legend{font-size:12px;font-weight:600}.note{font-size:13px;font-weight:650;fill:#30245C}</style>',
        '<text x="58" y="55" class="tag">FLAGQUANTUM · END-TO-END VQE TRAINING</text>',
        f'<text x="58" y="91" class="title">{html.escape(title)}</text>',
        f'<text x="58" y="119" class="sub">{html.escape(subtitle)}</text>',
    ]


def _scale(value: float, low: float, high: float, start: float, end: float) -> float:
    if high <= low:
        return (start + end) / 2
    return start + (value - low) / (high - low) * (end - start)


def _line(
    values: list[tuple[float, float]],
    *,
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
    left: float,
    top: float,
    width: float,
    height: float,
    color: str,
    dashed: bool,
    stroke_width: float = 2.4,
) -> str:
    sampled = values[:: max(1, len(values) // 180)]
    if values and sampled[-1] != values[-1]:
        sampled.append(values[-1])
    points = " ".join(
        f"{_scale(x, x_low, x_high, left, left+width):.1f},{_scale(y, y_low, y_high, top+height, top):.1f}"
        for x, y in sampled
    )
    dash = ' stroke-dasharray="8 6"' if dashed else ""
    return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{stroke_width}"{dash}/>'


def _markers(
    values: list[tuple[float, float]],
    *,
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
    left: float,
    top: float,
    width: float,
    height: float,
    color: str,
    circular: bool,
) -> str:
    indices = sorted(
        {0, len(values) // 4, len(values) // 2, 3 * len(values) // 4, len(values) - 1}
    )
    shapes = []
    for index in indices:
        x, y = values[index]
        px = _scale(x, x_low, x_high, left, left + width)
        py = _scale(y, y_low, y_high, top + height, top)
        if circular:
            shapes.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.2" fill="#FFFFFF" stroke="{color}" stroke-width="2.4"/>'
            )
        else:
            shapes.append(
                f'<rect x="{px-4:.1f}" y="{py-4:.1f}" width="8" height="8" fill="#FFFFFF" stroke="{color}" stroke-width="2.4"/>'
            )
    return "".join(shapes)


def _panel_base(
    parts: list[str], *, x: int, y: int, w: int, h: int, title: str
) -> tuple[float, float, float, float]:
    parts.extend(
        [
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="#FFFFFF" stroke="#E2E8F0"/>',
            f'<text x="{x+22}" y="{y+31}" class="panel">{html.escape(title)}</text>',
        ]
    )
    left, top, width, height = x + 62, y + 52, w - 88, h - 100
    for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
        gy = top + height * ratio
        parts.append(
            f'<line x1="{left}" y1="{gy:.1f}" x2="{left+width}" y2="{gy:.1f}" stroke="#E8EDF3"/>'
        )
    parts.append(
        f'<line x1="{left}" y1="{top+height}" x2="{left+width}" y2="{top+height}" stroke="#94A3B8"/>'
    )
    return left, top, width, height


def _trajectory_figure(
    *,
    path: Path,
    payload: dict,
    title: str,
    x_values: Callable[[dict], list[float]],
    x_label: str,
    takeaway: str,
) -> None:
    runs = payload["runs"]
    wires = sorted({run["n_wires"] for run in runs})
    parts = _header(
        title,
        "A800 · Adam 500 steps · 相同初始化 · PyTorch 浅色实线方点 / JAX 深色虚线圆点 · 对数能量误差",
    )
    for index, n_wires in enumerate(wires):
        x, y, w, h = 58 + index * 575, 176, 540, 650
        left, top, plot_w, plot_h = _panel_base(
            parts, x=x, y=y, w=w, h=h, title=f"{n_wires} QUBITS"
        )
        selected = [run for run in runs if run["n_wires"] == n_wires]
        all_x = [value for run in selected for value in x_values(run)]
        x_low, x_high = 0.0, max(all_x)
        y_low, y_high = -6.0, 1.0
        tolerance_y = _scale(-3.0, y_low, y_high, top + plot_h, top)
        parts.append(
            f'<line x1="{left}" y1="{tolerance_y:.1f}" x2="{left+plot_w}" y2="{tolerance_y:.1f}" stroke="#B83A3A" stroke-width="1.5" stroke-dasharray="4 4"/>'
        )
        parts.append(
            f'<text x="{left+plot_w-4}" y="{tolerance_y-7:.1f}" text-anchor="end" class="axis" fill="#B83A3A">收敛阈值 1e-3</text>'
        )
        for run in selected:
            xs = x_values(run)
            ys = [math.log10(max(value, 1e-6)) for value in run["energy_errors"]]
            points = list(zip(xs, ys))
            is_jax = run["backend"] == "jax"
            curve_color = (
                COLORS[run["layers"]] if is_jax else LIGHT_COLORS[run["layers"]]
            )
            parts.append(
                _line(
                    points,
                    x_low=x_low,
                    x_high=x_high,
                    y_low=y_low,
                    y_high=y_high,
                    left=left,
                    top=top,
                    width=plot_w,
                    height=plot_h,
                    color=curve_color,
                    dashed=is_jax,
                    stroke_width=3.0 if is_jax else 4.5,
                )
            )
            parts.append(
                _markers(
                    points,
                    x_low=x_low,
                    x_high=x_high,
                    y_low=y_low,
                    y_high=y_high,
                    left=left,
                    top=top,
                    width=plot_w,
                    height=plot_h,
                    color=curve_color,
                    circular=is_jax,
                )
            )
        for exponent in range(-6, 2):
            yy = _scale(exponent, y_low, y_high, top + plot_h, top)
            parts.append(
                f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" class="axis">1e{exponent}</text>'
            )
        for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
            xx = left + plot_w * ratio
            tick = x_low + (x_high - x_low) * ratio
            parts.append(
                f'<text x="{xx:.1f}" y="{top+plot_h+18:.1f}" text-anchor="middle" class="axis">{tick:.0f}</text>'
            )
        parts.append(
            f'<text x="{left+plot_w/2}" y="{y+h-4}" text-anchor="middle" class="axis">{html.escape(x_label)}</text>'
        )
    legend_y = 858
    for index, depth in enumerate((2, 4, 8)):
        lx = 570 + index * 150
        parts.append(
            f'<line x1="{lx}" y1="{legend_y}" x2="{lx+34}" y2="{legend_y}" stroke="{COLORS[depth]}" stroke-width="4"/>'
        )
        parts.append(
            f'<text x="{lx+43}" y="{legend_y+4}" class="legend">Depth {depth}</text>'
        )
    parts.extend(
        [
            '<line x1="1045" y1="858" x2="1079" y2="858" stroke="#94A3B8" stroke-width="5"/><rect x="1058" y="854" width="8" height="8" fill="#FFFFFF" stroke="#64748B" stroke-width="2"/>',
            '<text x="1088" y="862" class="legend">PyTorch · 浅色实线 / 方点</text>',
            '<line x1="1280" y1="858" x2="1314" y2="858" stroke="#334155" stroke-width="3" stroke-dasharray="8 6"/><circle cx="1297" cy="858" r="4" fill="#FFFFFF" stroke="#334155" stroke-width="2"/>',
            '<text x="1323" y="862" class="legend">JAX · 深色虚线 / 圆点</text>',
            '<line x1="58" y1="895" x2="1742" y2="895" stroke="#E2E8F0"/>',
            f'<circle cx="74" cy="925" r="5" fill="#54319B"/><text x="90" y="930" class="note">结论：{html.escape(takeaway)}</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def _crossover_figure(path: Path, payload: dict) -> None:
    runs = payload["runs"]
    comparisons = {
        (row["n_wires"], row["layers"]): row for row in payload["comparisons"]
    }
    parts = _header(
        "累计训练耗时与 JAX JIT 回本点",
        "A800 · Adam 500 steps · 墙钟时间包含 forward、backward、optimizer、同步以及 JAX 首次编译",
    )
    for row, n_wires in enumerate((4, 8, 12)):
        for column, depth in enumerate((2, 4, 8)):
            x, y, w, h = 58 + column * 575, 168 + row * 238, 540, 210
            cross = comparisons[(n_wires, depth)]["cumulative_time_crossover_step"]
            heading = (
                f"{n_wires}Q · Depth {depth} · 回本 {cross if cross else '>500'} 步"
            )
            left, top, plot_w, plot_h = _panel_base(
                parts, x=x, y=y, w=w, h=h, title=heading
            )
            selected = [
                run
                for run in runs
                if run["n_wires"] == n_wires and run["layers"] == depth
            ]
            high = max(run["total_wall_seconds"] for run in selected)
            for run in selected:
                values = list(
                    zip(range(1, run["steps"] + 1), run["cumulative_wall_seconds"])
                )
                parts.append(
                    _line(
                        values,
                        x_low=1,
                        x_high=run["steps"],
                        y_low=0,
                        y_high=high,
                        left=left,
                        top=top,
                        width=plot_w,
                        height=plot_h,
                        color="#3478C5" if run["backend"] == "pytorch" else "#D07124",
                        dashed=False,
                    )
                )
            if cross:
                cx = _scale(cross, 1, 500, left, left + plot_w)
                parts.append(
                    f'<line x1="{cx:.1f}" y1="{top}" x2="{cx:.1f}" y2="{top+plot_h}" stroke="#54319B" stroke-dasharray="4 4"/>'
                )
            parts.append(
                f'<text x="{left-8}" y="{top+4}" text-anchor="end" class="axis">{high:.0f}s</text><text x="{left-8}" y="{top+plot_h+4}" text-anchor="end" class="axis">0</text>'
            )
            for step in (1, 250, 500):
                xx = _scale(step, 1, 500, left, left + plot_w)
                parts.append(
                    f'<text x="{xx:.1f}" y="{top+plot_h+16:.1f}" text-anchor="middle" class="axis">{step}</text>'
                )
    parts.extend(
        [
            '<line x1="640" y1="906" x2="678" y2="906" stroke="#3478C5" stroke-width="4"/><text x="688" y="910" class="legend">PyTorch 累计耗时</text>',
            '<line x1="860" y1="906" x2="898" y2="906" stroke="#D07124" stroke-width="4"/><text x="908" y="910" class="legend">JAX 累计耗时</text>',
            '<line x1="58" y1="928" x2="1742" y2="928" stroke="#E2E8F0"/>',
            '<circle cx="74" cy="946" r="5" fill="#54319B"/><text x="90" y="951" class="note">结论：全部配置在 108–157 步回本；回本只代表累计速度反超，不代表 VQE 已达到基态。</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _trajectory_figure(
        path=args.output_dir / "01_energy_error_vs_steps.svg",
        payload=payload,
        title="VQE 收敛轨迹：能量误差 × 迭代步数",
        x_values=lambda run: list(range(1, run["steps"] + 1)),
        x_label="优化步数",
        takeaway="后端不改变按步收敛；浅层 Ansatz 在 8–12 qubit 出现明显表达与优化瓶颈。",
    )
    _trajectory_figure(
        path=args.output_dir / "02_energy_error_vs_wall_time.svg",
        payload=payload,
        title="VQE 有效求解速度：能量误差 × 墙钟时间",
        x_values=lambda run: run["cumulative_wall_seconds"],
        x_label="累计墙钟时间（秒）",
        takeaway="JAX 先承担 JIT 冷启动，再以更快稳态推进；只有达到目标误差才构成有效加速。",
    )
    _crossover_figure(args.output_dir / "03_cumulative_time_crossover.svg", payload)


if __name__ == "__main__":
    main()
