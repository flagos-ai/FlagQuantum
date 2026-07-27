"""Plot readable TFIM VQE loss trajectories by qubit scale and continuation phase."""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

import torch

WIDTH, HEIGHT = 1680, 920
QUBITS = (8, 12, 16, 20)
DEPTHS = (2, 4, 8, 12, 16, 20)
COLORS = {
    2: "#3478C5",
    4: "#2E9D8F",
    8: "#7A9E35",
    12: "#D99A2B",
    16: "#D06161",
    20: "#6546A6",
}


def _state(checkpoint_dir: Path, backend: str, qubits: int, depth: int) -> dict:
    path = checkpoint_dir / f"adam_{backend}_q{qubits}_d{depth}.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


def _header(title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#F8FAFC"/>',
        '<rect x="24" y="24" width="1632" height="872" rx="20" fill="#FFFFFF" stroke="#E2E8F0"/>',
        '<rect x="24" y="24" width="8" height="104" rx="4" fill="#54319B"/>',
        '<style>text{font-family:Inter,"PingFang SC","Microsoft YaHei",Arial,sans-serif;fill:#334155}.tag{font-size:12px;font-weight:700;fill:#54319B;letter-spacing:1px}.title{font-size:29px;font-weight:750;fill:#172B4D}.sub{font-size:13px;fill:#64748B}.panel{font-size:17px;font-weight:700;fill:#172B4D}.axis{font-size:12px;fill:#64748B}.legend{font-size:12px;font-weight:650}.note{font-size:14px;font-weight:650;fill:#30245C}</style>',
        '<text x="58" y="55" class="tag">FLAGQUANTUM · TFIM VQE · LOSS TRAJECTORY</text>',
        f'<text x="58" y="91" class="title">{html.escape(title)}</text>',
        f'<text x="58" y="119" class="sub">{html.escape(subtitle)}</text>',
    ]


def _scale(value: float, low: float, high: float, start: float, end: float) -> float:
    return start + (value - low) / (high - low) * (end - start)


def _polyline(
    values: list[float],
    *,
    max_steps: int,
    left: float,
    top: float,
    width: float,
    height: float,
    color: str,
    dashed: bool = False,
    start_step: int = 1,
) -> str:
    stride = max(1, len(values) // 280)
    indices = list(range(0, len(values), stride))
    if indices[-1] != len(values) - 1:
        indices.append(len(values) - 1)
    points = []
    for index in indices:
        step = start_step + index
        value = math.log10(max(float(values[index]), 1e-6))
        x = _scale(step, 1, max_steps, left, left + width)
        y = _scale(value, -6, 1, top + height, top)
        points.append(f"{x:.1f},{y:.1f}")
    dash = ' stroke-dasharray="8 6"' if dashed else ""
    return f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.8"{dash}/>'


def _panel(
    parts: list[str], *, x: int, y: int, w: int, h: int, title: str, max_steps: int
) -> tuple[float, float, float, float]:
    parts.extend(
        [
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="#FFFFFF" stroke="#E2E8F0"/>',
            f'<text x="{x+22}" y="{y+32}" class="panel">{html.escape(title)}</text>',
        ]
    )
    left, top, plot_w, plot_h = x + 76, y + 58, w - 108, h - 112
    for exponent in range(-6, 2):
        py = _scale(exponent, -6, 1, top + plot_h, top)
        parts.append(
            f'<line x1="{left}" y1="{py}" x2="{left+plot_w}" y2="{py}" stroke="#E8EDF3"/>'
        )
        parts.append(
            f'<text x="{left-12}" y="{py+4}" text-anchor="end" class="axis">1e{exponent}</text>'
        )
    for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
        step = 1 + round((max_steps - 1) * ratio)
        px = left + plot_w * ratio
        parts.append(
            f'<text x="{px}" y="{top+plot_h+22}" text-anchor="middle" class="axis">{step}</text>'
        )
    threshold = _scale(-3, -6, 1, top + plot_h, top)
    parts.append(
        f'<line x1="{left}" y1="{threshold}" x2="{left+plot_w}" y2="{threshold}" stroke="#B83A3A" stroke-width="1.6" stroke-dasharray="5 4"/>'
    )
    parts.append(
        f'<text x="{left+plot_w-4}" y="{threshold-7}" text-anchor="end" class="axis" fill="#B83A3A">阈值 1e-3</text>'
    )
    parts.append(
        f'<text x="{left+plot_w/2}" y="{y+h-14}" text-anchor="middle" class="axis">Optimizer steps</text>'
    )
    return left, top, plot_w, plot_h


def _finish(parts: list[str], takeaway: str, *, legend_y: int = 790) -> None:
    for index, depth in enumerate(DEPTHS):
        x = 380 + index * 155
        parts.append(
            f'<line x1="{x}" y1="{legend_y}" x2="{x+35}" y2="{legend_y}" stroke="{COLORS[depth]}" stroke-width="4"/>'
        )
        parts.append(
            f'<text x="{x+44}" y="{legend_y+4}" class="legend">Depth {depth}</text>'
        )
    parts.extend(
        [
            '<line x1="58" y1="834" x2="1622" y2="834" stroke="#E2E8F0"/>',
            '<circle cx="74" cy="862" r="5" fill="#54319B"/>',
            f'<text x="90" y="867" class="note">结论：{html.escape(takeaway)}</text>',
            "</svg>",
        ]
    )


def _qubit_figure(path: Path, checkpoint_dir: Path, qubits: int) -> None:
    parts = _header(
        f"{qubits} Qubits：不同电路深度的VQE Loss下降轨迹",
        "前1000步冻结视图 · Adam lr=0.01 · 纵轴为基态能量误差 E(θ)-E₀（对数刻度）",
    )
    for column, backend in enumerate(("pytorch", "jax")):
        x = 58 + column * 800
        left, top, plot_w, plot_h = _panel(
            parts,
            x=x,
            y=165,
            w=764,
            h=575,
            title="PyTorch Native" if backend == "pytorch" else "JAX JIT",
            max_steps=1000,
        )
        for depth in DEPTHS:
            state = _state(checkpoint_dir, backend, qubits, depth)
            values = state["energy_errors"][:1000]
            parts.append(
                _polyline(
                    values,
                    max_steps=1000,
                    left=left,
                    top=top,
                    width=plot_w,
                    height=plot_h,
                    color=COLORS[depth],
                )
            )
    messages = {
        8: "Depth 8开始进入收敛区；Depth 2/4接近阈值但下降明显变慢。",
        12: "Depth 16/20稳定收敛，Depth 12逼近阈值，浅层配置进入高误差平台。",
        16: "Depth 16形成有效窗口；Depth 20曾达标但存在末期波动，浅层仍受表达与优化限制。",
        20: "1000步内所有深度均未稳定达标；Depth 20最接近，Depth 2–8形成明显平台。",
    }
    _finish(parts, messages[qubits])
    path.write_text("\n".join(parts), encoding="utf-8")


def _continuation_figure(path: Path, checkpoint_dir: Path) -> None:
    selected = ((8, 2), (8, 4), (12, 12), (16, 20), (20, 12), (20, 20))
    parts = _header(
        "续训是否有效：1000步之后的Loss轨迹",
        "竖线为首轮1000步边界 · PyTorch浅色实线 / JAX深色虚线 · 续训阶段部分配置降至lr=0.003",
    )
    for index, (qubits, depth) in enumerate(selected):
        column, row = index % 3, index // 3
        x, y, w, h = 52 + column * 535, 165 + row * 300, 505, 270
        states = {
            backend: _state(checkpoint_dir, backend, qubits, depth)
            for backend in ("pytorch", "jax")
        }
        max_steps = max(int(state["completed_steps"]) for state in states.values())
        left, top, plot_w, plot_h = _panel(
            parts,
            x=x,
            y=y,
            w=w,
            h=h,
            title=f"{qubits}Q · Depth {depth} · latest {max_steps} steps",
            max_steps=max_steps,
        )
        boundary = _scale(1000, 1, max_steps, left, left + plot_w)
        parts.append(
            f'<line x1="{boundary}" y1="{top}" x2="{boundary}" y2="{top+plot_h}" stroke="#D07124" stroke-width="2" stroke-dasharray="4 4"/>'
        )
        for backend, color, dashed in (
            ("pytorch", "#8FBCE6", False),
            ("jax", "#54319B", True),
        ):
            parts.append(
                _polyline(
                    states[backend]["energy_errors"],
                    max_steps=max_steps,
                    left=left,
                    top=top,
                    width=plot_w,
                    height=plot_h,
                    color=color,
                    dashed=dashed,
                )
            )
    parts.extend(
        [
            '<line x1="565" y1="780" x2="605" y2="780" stroke="#8FBCE6" stroke-width="4"/><text x="616" y="784" class="legend">PyTorch</text>',
            '<line x1="760" y1="780" x2="800" y2="780" stroke="#54319B" stroke-width="3" stroke-dasharray="8 6"/><text x="811" y="784" class="legend">JAX</text>',
            '<line x1="58" y1="834" x2="1622" y2="834" stroke="#E2E8F0"/>',
            '<circle cx="74" cy="862" r="5" fill="#54319B"/>',
            '<text x="90" y="867" class="note">结论：续训能改善接近阈值的配置，但浅层平台和20-qubit优化瓶颈不能仅靠增加步数消除。</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, qubits in enumerate(QUBITS, start=1):
        _qubit_figure(
            args.output_dir / f"0{index}_q{qubits}_loss_trajectories.svg",
            args.checkpoint_dir,
            qubits,
        )
    _continuation_figure(
        args.output_dir / "05_selected_continuation_trajectories.svg",
        args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
