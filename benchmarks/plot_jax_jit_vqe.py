"""Generate SVG decision maps for the multi-depth TFIM VQE JAX benchmark."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Callable

WIDTH, HEIGHT = 1440, 730
LEFT, TOP = 220, 230
CELL_W, CELL_H = 230, 120


def _mix(start: tuple[int, int, int], end: tuple[int, int, int], ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    values = [round(a + (b - a) * ratio) for a, b in zip(start, end)]
    return "#" + "".join(f"{value:02X}" for value in values)


def _heatmap(
    *,
    path: Path,
    rows: list[dict],
    title: str,
    subtitle: str,
    value: Callable[[dict], float],
    label: Callable[[dict], str],
    detail: Callable[[dict], str],
    color: Callable[[float, float, float], str],
    legend: str,
    takeaway: str,
) -> None:
    wires = sorted({int(row["n_wires"]) for row in rows})
    depths = sorted({int(row["layers"]) for row in rows})
    indexed = {(int(row["layers"]), int(row["n_wires"])): row for row in rows}
    values = [value(row) for row in rows]
    low, high = min(values), max(values)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#F8FAFC"/>',
        '<rect x="24" y="24" width="1392" height="682" rx="18" fill="#FFFFFF" stroke="#E2E8F0"/>',
        '<rect x="24" y="24" width="7" height="98" rx="3.5" fill="#54319B"/>',
        '<style>text{font-family:Inter,"PingFang SC","Microsoft YaHei",Arial,sans-serif;fill:#243447}.title{font-size:27px;font-weight:750;fill:#172B4D}.sub{font-size:13px;fill:#64748B}.axis{font-size:14px;font-weight:650;fill:#334155}.cell{font-size:20px;font-weight:750}.detail{font-size:11px;font-weight:500;fill:#334155}.hint{font-size:12px;fill:#475569}.tag{font-size:11px;font-weight:700;fill:#54319B;letter-spacing:.7px}.takeaway{font-size:13px;font-weight:650;fill:#30245C}</style>',
        '<text x="54" y="52" class="tag">FLAGQUANTUM · PERFORMANCE STUDY</text>',
        f'<text x="54" y="84" class="title">{html.escape(title)}</text>',
        f'<text x="54" y="111" class="sub">{html.escape(subtitle)}</text>',
        '<rect x="54" y="136" width="1332" height="38" rx="8" fill="#F8FAFC" stroke="#E2E8F0"/>',
        f'<text x="74" y="160" class="hint">{html.escape(legend)}</text>',
        f'<rect x="{LEFT}" y="{TOP-12}" width="{len(wires)*CELL_W}" height="{len(depths)*CELL_H+12}" rx="12" fill="#F8FAFC"/>',
    ]
    for column, wire in enumerate(wires):
        x = LEFT + column * CELL_W + CELL_W / 2
        parts.append(
            f'<text x="{x:.1f}" y="{TOP-27}" text-anchor="middle" class="axis">{wire} QUBITS</text>'
        )
    for row_index, depth in enumerate(depths):
        y = TOP + row_index * CELL_H + CELL_H / 2
        parts.append(
            f'<text x="{LEFT-24}" y="{y+5:.1f}" text-anchor="end" class="axis">DEPTH {depth}</text>'
        )
        for column, wire in enumerate(wires):
            item = indexed.get((depth, wire))
            x = LEFT + column * CELL_W
            top = TOP + row_index * CELL_H
            if item is None:
                fill, text, secondary = "#F1F5F9", "N/A", "无数据"
            else:
                numeric = value(item)
                fill, text, secondary = (
                    color(numeric, low, high),
                    label(item),
                    detail(item),
                )
            parts.append(
                f'<rect x="{x+5}" y="{top+5}" width="{CELL_W-10}" height="{CELL_H-10}" rx="10" fill="{fill}" stroke="#FFFFFF" stroke-width="3"/>'
            )
            parts.append(
                f'<text x="{x+CELL_W/2:.1f}" y="{top+CELL_H/2-1:.1f}" text-anchor="middle" class="cell">{html.escape(text)}</text>'
            )
            parts.append(
                f'<text x="{x+CELL_W/2:.1f}" y="{top+CELL_H/2+21:.1f}" text-anchor="middle" class="detail">{html.escape(secondary)}</text>'
            )
    footer_y = TOP + len(depths) * CELL_H + 31
    parts.append(
        f'<line x1="54" y1="{footer_y}" x2="1386" y2="{footer_y}" stroke="#E2E8F0"/>'
    )
    parts.append(f'<circle cx="70" cy="{footer_y+27}" r="5" fill="#54319B"/>')
    parts.append(
        f'<text x="85" y="{footer_y+32}" class="takeaway">结论：{html.escape(takeaway)}</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _ratio(
    value: float, low: float, high: float, *, logarithmic: bool = False
) -> float:
    if high <= low:
        return 0.5
    if logarithmic:
        value, low, high = (
            math.log10(max(value, 1e-12)),
            math.log10(max(low, 1e-12)),
            math.log10(max(high, 1e-12)),
        )
    return (value - low) / (high - low)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    if any(payload["workload"]["task"] != "tfim_vqe" for payload in payloads):
        raise ValueError("every input must be a tfim_vqe benchmark payload")
    payload = payloads[0]
    rows = [row for item in payloads for row in item["measurements"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = (
        payload["environment"].get("cuda_device_name")
        or payload["environment"]["device"]
    )
    common = (
        f"{device} · TFIM energy + gradient · Statevector · single_device_fast_path"
    )

    _heatmap(
        path=args.output_dir / "01_vqe_jax_benefit_map.svg",
        rows=rows,
        title="JAX JIT 收益区间：多深度横场 Ising VQE",
        subtitle=common,
        value=lambda row: float(row["steady_speedup_jax_over_pytorch"]),
        label=lambda row: f'{row["steady_speedup_jax_over_pytorch"]:.1f}×',
        detail=lambda row: f'JAX {row["jax_steady_seconds_median"] * 1000:.1f} ms / step',
        color=lambda value, low, high: _mix(
            (234, 247, 238), (46, 125, 79), _ratio(value, low, high)
        ),
        legend="单元格为 PyTorch/JAX 稳态耗时比；>1× 表示 JAX 更快，颜色越深收益越高。",
        takeaway="4–16 qubit 是清晰收益区；20 qubit 的稳态优势显著收窄。",
    )
    _heatmap(
        path=args.output_dir / "02_vqe_compile_cost_map.svg",
        rows=rows,
        title="JAX JIT 首次编译成本",
        subtitle=common + " · fresh process · persistent cache disabled",
        value=lambda row: float(row["jax_cold_total_seconds_median"]),
        label=lambda row: f'{row["jax_cold_total_seconds_median"]:.1f}s',
        detail=lambda row: "首次完整编译",
        color=lambda value, low, high: _mix(
            (255, 243, 227), (184, 58, 58), _ratio(value, low, high)
        ),
        legend="单元格为 kernel build + 首次 energy/gradient 调用；颜色越深等待越长。",
        takeaway="编译成本随规模和深度上升，深电路的首次等待不可忽略。",
    )
    _heatmap(
        path=args.output_dir / "03_vqe_break_even_map.svg",
        rows=rows,
        title="JAX JIT 编译回本边界",
        subtitle=common,
        value=lambda row: float(row["break_even_calls"] or 100_000),
        label=lambda row: (
            ">100k" if row["break_even_calls"] is None else str(row["break_even_calls"])
        ),
        detail=lambda row: "同形状 VQE steps",
        color=lambda value, low, high: _mix(
            (232, 241, 255), (84, 49, 155), _ratio(value, low, high, logarithmic=True)
        ),
        legend="单元格为累计耗时首次低于 PyTorch 所需的同形状调用数；颜色越深越难回本。",
        takeaway="4–16 qubit 通常约 130–190 步回本；20 qubit 上升至约 500 步。",
    )

    def decision(row: dict) -> float:
        speedup = float(row["steady_speedup_jax_over_pytorch"])
        calls = row["break_even_calls"]
        if speedup >= 2.0 and calls is not None and calls <= 500:
            return 2.0
        if speedup > 1.0 and calls is not None and calls <= 5000:
            return 1.0
        return 0.0

    names = {2.0: "JAX优先", 1.0: "条件选择", 0.0: "PyTorch优先"}
    palette = {2.0: "#79B98C", 1.0: "#F2B66D", 0.0: "#8FBCE6"}
    _heatmap(
        path=args.output_dir / "04_vqe_backend_decision_map.svg",
        rows=rows,
        title="Backend 选择建议：收益与编译成本共同决策",
        subtitle=common + " · development heuristic, not release policy",
        value=decision,
        label=lambda row: names[decision(row)],
        detail=lambda row: (
            f'{row["steady_speedup_jax_over_pytorch"]:.1f}× · {row["break_even_calls"]}步回本'
        ),
        color=lambda value, low, high: palette[value],
        legend="JAX优先：≥2×且≤500步回本；条件选择：>1×且≤5000步回本；否则优先PyTorch。",
        takeaway="后端选择必须同时考虑稳态收益、编译成本、迭代次数与计算图稳定性。",
    )


if __name__ == "__main__":
    main()
