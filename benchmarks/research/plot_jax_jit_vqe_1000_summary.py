"""Plot the frozen first-1000-step TFIM VQE summary from training checkpoints."""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path
from typing import Any, Callable

import torch

WIDTH, HEIGHT = 1680, 900
QUBITS = (8, 12, 16, 20)
DEPTHS = (2, 4, 8, 12, 16, 20)


def _load(checkpoint_dir: Path) -> dict[tuple[int, int], dict[str, dict[str, Any]]]:
    matrix = {}
    for qubits in QUBITS:
        for depth in DEPTHS:
            pair = {}
            for backend in ("pytorch", "jax"):
                path = checkpoint_dir / f"adam_{backend}_q{qubits}_d{depth}.pt"
                if not path.exists():
                    continue
                state = torch.load(path, map_location="cpu", weights_only=False)
                if len(state["energy_errors"]) < 1000:
                    continue
                errors = [float(value) for value in state["energy_errors"][:1000]]
                wall = [
                    float(value) for value in state["cumulative_wall_seconds"][:1000]
                ]
                pair[backend] = {
                    "errors": errors,
                    "wall": wall,
                    "final_error": errors[-1],
                    "first_converged": next(
                        (
                            index + 1
                            for index, value in enumerate(errors)
                            if value <= 1e-3
                        ),
                        None,
                    ),
                }
            matrix[(qubits, depth)] = pair
    return matrix


def _header(title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#F8FAFC"/>',
        '<rect x="24" y="24" width="1632" height="852" rx="20" fill="#FFFFFF" stroke="#E2E8F0"/>',
        '<rect x="24" y="24" width="8" height="104" rx="4" fill="#54319B"/>',
        '<style>text{font-family:Inter,"PingFang SC","Microsoft YaHei",Arial,sans-serif;fill:#334155}.tag{font-size:12px;font-weight:700;fill:#54319B;letter-spacing:1px}.title{font-size:29px;font-weight:750;fill:#172B4D}.sub{font-size:13px;fill:#64748B}.axis{font-size:13px;font-weight:600;fill:#475569}.cell{font-size:18px;font-weight:750}.detail{font-size:11px;fill:#334155}.panel{font-size:17px;font-weight:700;fill:#172B4D}.note{font-size:14px;font-weight:650;fill:#30245C}</style>',
        '<text x="58" y="55" class="tag">FLAGQUANTUM · TFIM VQE · FROZEN 1000-STEP VIEW</text>',
        f'<text x="58" y="91" class="title">{html.escape(title)}</text>',
        f'<text x="58" y="119" class="sub">{html.escape(subtitle)}</text>',
    ]


def _finish(parts: list[str], takeaway: str) -> None:
    parts.extend(
        [
            '<line x1="58" y1="824" x2="1622" y2="824" stroke="#E2E8F0"/>',
            '<circle cx="74" cy="850" r="5" fill="#54319B"/>',
            f'<text x="90" y="855" class="note">结论：{html.escape(takeaway)}</text>',
            "</svg>",
        ]
    )


def _mix(start: tuple[int, int, int], end: tuple[int, int, int], ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    rgb = [round(a + (b - a) * ratio) for a, b in zip(start, end)]
    return "#" + "".join(f"{value:02X}" for value in rgb)


def _heatmap(
    *,
    path: Path,
    matrix: dict,
    title: str,
    subtitle: str,
    value: Callable[[dict], float | None],
    label: Callable[[dict], str],
    detail: Callable[[dict], str],
    color: Callable[[float, float, float], str],
    takeaway: str,
) -> None:
    left, top, cell_w, cell_h = 245, 205, 218, 128
    numeric = [value(pair) for pair in matrix.values()]
    numeric = [item for item in numeric if item is not None]
    low, high = min(numeric), max(numeric)
    parts = _header(title, subtitle)
    for column, depth in enumerate(DEPTHS):
        x = left + column * cell_w + cell_w / 2
        parts.append(
            f'<text x="{x}" y="174" text-anchor="middle" class="axis">DEPTH {depth}</text>'
        )
    for row, qubits in enumerate(QUBITS):
        y = top + row * cell_h
        parts.append(
            f'<text x="{left-28}" y="{y+69}" text-anchor="end" class="axis">{qubits} QUBITS</text>'
        )
        for column, depth in enumerate(DEPTHS):
            pair = matrix[(qubits, depth)]
            number = value(pair)
            fill = "#EEF2F6" if number is None else color(number, low, high)
            x = left + column * cell_w
            parts.append(
                f'<rect x="{x+5}" y="{y+5}" width="{cell_w-10}" height="{cell_h-10}" rx="13" fill="{fill}" stroke="#FFFFFF" stroke-width="3"/>'
            )
            parts.append(
                f'<text x="{x+cell_w/2}" y="{y+57}" text-anchor="middle" class="cell">{html.escape(label(pair))}</text>'
            )
            parts.append(
                f'<text x="{x+cell_w/2}" y="{y+82}" text-anchor="middle" class="detail">{html.escape(detail(pair))}</text>'
            )
    _finish(parts, takeaway)
    path.write_text("\n".join(parts), encoding="utf-8")


def _scale(value: float, low: float, high: float, start: float, end: float) -> float:
    return start + (value - low) / (high - low) * (end - start)


def _line_chart(path: Path, matrix: dict) -> None:
    parts = _header(
        "1000步最终能量误差：有效深度窗口",
        "Adam lr=0.01 · 实线/方点 PyTorch · 虚线/圆点 JAX · 红色虚线为1e-3收敛阈值",
    )
    for index, qubits in enumerate(QUBITS):
        column, row = index % 2, index // 2
        x, y, w, h = 70 + column * 800, 165 + row * 315, 740, 275
        left, top, plot_w, plot_h = x + 78, y + 48, w - 112, h - 90
        parts.extend(
            [
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="#FFFFFF" stroke="#E2E8F0"/>',
                f'<text x="{x+20}" y="{y+30}" class="panel">{qubits} QUBITS</text>',
            ]
        )
        for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
            gy = top + plot_h * ratio
            parts.append(
                f'<line x1="{left}" y1="{gy}" x2="{left+plot_w}" y2="{gy}" stroke="#E8EDF3"/>'
            )
        threshold_y = _scale(-3, -4.5, 0, top + plot_h, top)
        parts.append(
            f'<line x1="{left}" y1="{threshold_y}" x2="{left+plot_w}" y2="{threshold_y}" stroke="#B83A3A" stroke-dasharray="5 4"/>'
        )
        for backend, color, dash, marker in (
            ("pytorch", "#8FBCE6", "", "square"),
            ("jax", "#54319B", ' stroke-dasharray="8 6"', "circle"),
        ):
            points = []
            for depth in DEPTHS:
                error = matrix[(qubits, depth)][backend]["final_error"]
                px = _scale(depth, 2, 20, left, left + plot_w)
                py = _scale(math.log10(max(error, 1e-5)), -4.5, 0, top + plot_h, top)
                points.append(f"{px:.1f},{py:.1f}")
                if marker == "circle":
                    parts.append(
                        f'<circle cx="{px}" cy="{py}" r="4" fill="#FFFFFF" stroke="{color}" stroke-width="2.5"/>'
                    )
                else:
                    parts.append(
                        f'<rect x="{px-4}" y="{py-4}" width="8" height="8" fill="#FFFFFF" stroke="{color}" stroke-width="2.5"/>'
                    )
            parts.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"{dash}/>'
            )
        for depth in DEPTHS:
            px = _scale(depth, 2, 20, left, left + plot_w)
            parts.append(
                f'<text x="{px}" y="{top+plot_h+19}" text-anchor="middle" class="axis">{depth}</text>'
            )
        for exponent in (-4, -3, -2, -1, 0):
            py = _scale(exponent, -4.5, 0, top + plot_h, top)
            parts.append(
                f'<text x="{left-10}" y="{py+4}" text-anchor="end" class="axis">1e{exponent}</text>'
            )
    _finish(parts, "存在有效深度窗口：浅层表达不足，深层也不保证误差单调下降。")
    path.write_text("\n".join(parts), encoding="utf-8")


def _parity(path: Path, matrix: dict) -> None:
    parts = _header(
        "PyTorch与JAX的1000步收敛一致性",
        "每个点对应一个Qubits×Depth配置 · 横轴PyTorch最终误差 · 纵轴JAX最终误差 · 双对数坐标",
    )
    left, top, size = 310, 180, 600
    parts.append(
        f'<rect x="{left}" y="{top}" width="{size}" height="{size}" rx="12" fill="#F8FAFC" stroke="#E2E8F0"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{top+size}" x2="{left+size}" y2="{top}" stroke="#94A3B8" stroke-width="2" stroke-dasharray="7 6"/>'
    )
    palette = {8: "#3478C5", 12: "#2E7D4F", 16: "#D07124", 20: "#54319B"}
    for (qubits, depth), pair in matrix.items():
        xerr, yerr = pair["pytorch"]["final_error"], pair["jax"]["final_error"]
        px = _scale(math.log10(max(xerr, 1e-5)), -5, 0, left, left + size)
        py = _scale(math.log10(max(yerr, 1e-5)), -5, 0, top + size, top)
        parts.append(
            f'<circle cx="{px}" cy="{py}" r="7" fill="{palette[qubits]}" fill-opacity=".82" stroke="#FFFFFF" stroke-width="2"/>'
        )
    for exponent in range(-5, 1):
        px = _scale(exponent, -5, 0, left, left + size)
        py = _scale(exponent, -5, 0, top + size, top)
        parts.append(
            f'<text x="{px}" y="{top+size+25}" text-anchor="middle" class="axis">1e{exponent}</text>'
        )
        parts.append(
            f'<text x="{left-14}" y="{py+4}" text-anchor="end" class="axis">1e{exponent}</text>'
        )
    parts.append(
        f'<text x="{left+size/2}" y="{top+size+58}" text-anchor="middle" class="axis">PyTorch 最终误差</text>'
    )
    for index, qubits in enumerate(QUBITS):
        ly = 260 + index * 72
        parts.append(
            f'<circle cx="1080" cy="{ly}" r="8" fill="{palette[qubits]}"/><text x="1103" y="{ly+5}" class="panel">{qubits} QUBITS</text>'
        )
    parts.append(
        '<text x="1030" y="610" class="panel">点越接近对角线</text><text x="1030" y="640" class="axis">表示两个后端的按步优化结果越一致</text>'
    )
    _finish(parts, "所有配置紧贴y=x，JAX主要改变训练速度，不改变VQE按步收敛语义。")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    matrix = _load(args.checkpoint_dir)
    if any(set(pair) != {"pytorch", "jax"} for pair in matrix.values()):
        raise RuntimeError("all 24 configurations need complete first-1000-step traces")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def status(pair: dict) -> float:
        final = all(pair[b]["final_error"] <= 1e-3 for b in ("pytorch", "jax"))
        ever = all(pair[b]["first_converged"] is not None for b in ("pytorch", "jax"))
        return 2.0 if final else (1.0 if ever else 0.0)

    status_names = {2.0: "稳定收敛", 1.0: "曾经达标", 0.0: "未收敛"}
    status_colors = {2.0: "#79B98C", 1.0: "#F2B66D", 0.0: "#E99A9A"}
    _heatmap(
        path=args.output_dir / "01_convergence_phase_map.svg",
        matrix=matrix,
        title="1000步VQE收敛相图",
        subtitle="Adam lr=0.01 · 收敛阈值1e-3 · 稳定收敛要求PyTorch/JAX第1000步均保持达标",
        value=status,
        label=lambda pair: status_names[status(pair)],
        detail=lambda pair: f'PT {pair["pytorch"]["final_error"]:.1e} · JAX {pair["jax"]["final_error"]:.1e}',
        color=lambda number, low, high: status_colors[number],
        takeaway="最小有效深度随qubit数上升；20 qubit在当前深度与1000步预算内尚未稳定收敛。",
    )
    _line_chart(args.output_dir / "02_final_error_vs_depth.svg", matrix)
    _parity(args.output_dir / "03_backend_parity.svg", matrix)

    def speedup(pair: dict) -> float:
        return pair["pytorch"]["wall"][999] / pair["jax"]["wall"][999]

    _heatmap(
        path=args.output_dir / "04_end_to_end_speedup_map.svg",
        matrix=matrix,
        title="1000步端到端JAX JIT加速比",
        subtitle="PyTorch累计墙钟 / JAX累计墙钟 · 包含JAX首次编译 · 数值越大表示JAX端到端收益越高",
        value=speedup,
        label=lambda pair: f"{speedup(pair):.2f}×",
        detail=lambda pair: f'PT {pair["pytorch"]["wall"][999]:.0f}s · JAX {pair["jax"]["wall"][999]:.0f}s',
        color=lambda number, low, high: _mix(
            (232, 241, 255), (46, 125, 79), (number - low) / (high - low)
        ),
        takeaway="JAX在8–16 qubit保持约4–6倍收益；20 qubit下降至约1.1倍。",
    )

    def crossover(pair: dict) -> float | None:
        for index, (torch_time, jax_time) in enumerate(
            zip(pair["pytorch"]["wall"], pair["jax"]["wall"]), start=1
        ):
            if jax_time <= torch_time:
                return float(index)
        return None

    _heatmap(
        path=args.output_dir / "05_jit_break_even_map.svg",
        matrix=matrix,
        title="JAX JIT累计耗时回本步数",
        subtitle="累计JAX墙钟首次低于PyTorch所需的同形状训练步数 · 数值越大表示冷编译越难回本",
        value=crossover,
        label=lambda pair: (
            ">1000" if crossover(pair) is None else str(int(crossover(pair) or 0))
        ),
        detail=lambda pair: "optimizer steps",
        color=lambda number, low, high: _mix(
            (234, 247, 238), (84, 49, 155), (number - low) / (high - low)
        ),
        takeaway="小中规模通常约百步回本；20 qubit需数百步，深层配置逼近完整训练后期。",
    )


if __name__ == "__main__":
    main()
