"""Generate dependency-free presentation SVGs for the JAX JIT benchmark."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Callable

WIDTH, HEIGHT = 1080, 620
LEFT, RIGHT, TOP, BOTTOM = 105, 45, 86, 92
PLOT_W, PLOT_H = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM
COLORS = {"pytorch": "#3478C5", "jax": "#D07124", "accent": "#54319B"}


def _base(title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<style>text{font-family:Inter,"PingFang SC","Microsoft YaHei",Arial,sans-serif;fill:#243447}.title{font-size:25px;font-weight:700;fill:#172B4D}.sub{font-size:13px;fill:#64748B}.axis{font-size:12px;fill:#64748B}.label{font-size:13px}.value{font-size:12px;font-weight:700}</style>',
        f'<text x="{LEFT}" y="38" class="title">{html.escape(title)}</text>',
        f'<text x="{LEFT}" y="62" class="sub">{html.escape(subtitle)}</text>',
    ]


def _axes(
    parts: list[str],
    wires: list[int],
    y_ticks: list[tuple[float, str]],
    y_map: Callable[[float], float],
    y_label: str,
) -> list[float]:
    xs = [LEFT + index * PLOT_W / max(1, len(wires) - 1) for index in range(len(wires))]
    for value, label in y_ticks:
        y = y_map(value)
        parts.append(
            f'<line x1="{LEFT}" y1="{y:.1f}" x2="{WIDTH-RIGHT}" y2="{y:.1f}" stroke="#D8E0EA" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{LEFT-12}" y="{y+4:.1f}" text-anchor="end" class="axis">{html.escape(label)}</text>'
        )
    parts.append(
        f'<line x1="{LEFT}" y1="{TOP+PLOT_H}" x2="{WIDTH-RIGHT}" y2="{TOP+PLOT_H}" stroke="#94A3B8"/>'
    )
    for x, wire in zip(xs, wires):
        parts.append(
            f'<text x="{x:.1f}" y="{TOP+PLOT_H+28}" text-anchor="middle" class="label">{wire}</text>'
        )
    parts.append(
        f'<text x="{LEFT+PLOT_W/2:.1f}" y="{HEIGHT-27}" text-anchor="middle" class="label">Qubits（state dimension = 2ⁿ）</text>'
    )
    parts.append(
        f'<text transform="translate(28 {TOP+PLOT_H/2:.1f}) rotate(-90)" text-anchor="middle" class="label">{html.escape(y_label)}</text>'
    )
    return xs


def _line_chart(
    path: Path,
    *,
    rows: list[dict],
    title: str,
    subtitle: str,
    series: list[tuple[str, str, str, float]],
    y_label: str,
) -> None:
    wires = [int(row["n_wires"]) for row in rows]
    all_values = [
        float(row[key]) * factor for _, key, _, factor in series for row in rows
    ]
    positive = [value for value in all_values if value > 0]
    low_power = math.floor(math.log10(min(positive)))
    high_power = math.ceil(math.log10(max(positive)))
    if high_power == low_power:
        high_power += 1
    low, high = 10**low_power, 10**high_power

    def y_map(value: float) -> float:
        ratio = (math.log10(value) - math.log10(low)) / (
            math.log10(high) - math.log10(low)
        )
        return TOP + PLOT_H * (1 - ratio)

    ticks = [(10**power, f"10^{power}") for power in range(low_power, high_power + 1)]
    parts = _base(title, subtitle)
    xs = _axes(parts, wires, ticks, y_map, y_label)
    for index, (name, key, color, factor) in enumerate(series):
        points = [(x, y_map(float(row[key]) * factor)) for x, row in zip(xs, rows)]
        parts.append(
            '<polyline points="'
            + " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            + f'" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        for x, y in points:
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" stroke="#FFF" stroke-width="2"/>'
            )
        legend_x = WIDTH - RIGHT - 330 + index * 165
        parts.append(
            f'<line x1="{legend_x}" y1="60" x2="{legend_x+25}" y2="60" stroke="{color}" stroke-width="4"/>'
        )
        parts.append(
            f'<text x="{legend_x+32}" y="65" class="axis">{html.escape(name)}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _bar_chart(
    path: Path,
    *,
    rows: list[dict],
    key: str,
    title: str,
    subtitle: str,
    y_label: str,
    color: str,
    formatter: Callable[[float], str],
) -> None:
    wires = [int(row["n_wires"]) for row in rows]
    values = [float(row[key] if row[key] is not None else 100_000) for row in rows]
    high = max(values) * 1.18 or 1.0

    def y_map(value: float) -> float:
        return TOP + PLOT_H * (1 - value / high)

    ticks = [(high * index / 4, formatter(high * index / 4)) for index in range(5)]
    parts = _base(title, subtitle)
    xs = _axes(parts, wires, ticks, y_map, y_label)
    bar_width = min(95, PLOT_W / max(1, len(wires)) * 0.48)
    for x, value in zip(xs, values):
        y = y_map(value)
        height = TOP + PLOT_H - y
        parts.append(
            f'<rect x="{x-bar_width/2:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{height:.1f}" rx="5" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{max(TOP+14,y-9):.1f}" text-anchor="middle" class="value">{html.escape(formatter(value))}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = payload["measurements"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = (
        payload["environment"].get("cuda_device_name")
        or payload["environment"]["device"]
    )
    common = f"{device} · single_device_fast_path · loss + gradient · median of fresh-process runs"

    _line_chart(
        args.output_dir / "01_cold_start_by_qubits.svg",
        rows=rows,
        title="JAX JIT 首次编译成本随模拟规模增长",
        subtitle=common,
        series=[
            ("PyTorch first", "pytorch_first_seconds_median", COLORS["pytorch"], 1.0),
            ("JAX build+first", "jax_cold_total_seconds_median", COLORS["jax"], 1.0),
        ],
        y_label="冷启动耗时（秒，对数轴）",
    )
    _line_chart(
        args.output_dir / "02_steady_state_by_qubits.svg",
        rows=rows,
        title="编译完成后，JAX 加速重复模拟",
        subtitle=common,
        series=[
            (
                "PyTorch native",
                "pytorch_steady_seconds_median",
                COLORS["pytorch"],
                1000.0,
            ),
            ("JAX JIT", "jax_steady_seconds_median", COLORS["jax"], 1000.0),
        ],
        y_label="稳态损失+梯度耗时（毫秒，对数轴）",
    )
    _bar_chart(
        args.output_dir / "03_steady_speedup.svg",
        rows=rows,
        key="steady_speedup_jax_over_pytorch",
        title="JAX JIT 稳态加速比",
        subtitle=common + " · 大于 1× 表示 JAX 更快",
        y_label="PyTorch / JAX（×）",
        color=COLORS["accent"],
        formatter=lambda value: f"{value:.1f}×",
    )
    _bar_chart(
        args.output_dir / "04_break_even_calls.svg",
        rows=rows,
        key="break_even_calls",
        title="编译成本何时回本？",
        subtitle=common + " · 持久化编译缓存已关闭",
        y_label="累计耗时回本所需调用次数",
        color=COLORS["jax"],
        formatter=lambda value: ">100k" if value >= 100_000 else str(int(round(value))),
    )


if __name__ == "__main__":
    main()
