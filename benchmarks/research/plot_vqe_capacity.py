"""Render VQE capacity probe artifacts as an SVG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = {}
    for path in args.input_dir.glob("vqe_capacity_*q_d32_*_a800.json"):
        payload = json.loads(path.read_text())
        records[(int(payload["n_wires"]), payload["backend"])] = payload

    qubits = (20, 22, 24, 26)
    width, height = 900, 560
    left, right, top, bottom = 92, 35, 108, 76
    plot_w, plot_h = width - left - right, height - top - bottom
    y_max = 85.0

    def x(index: int) -> float:
        return left + plot_w * index / (len(qubits) - 1)

    def y(value: float) -> float:
        return top + plot_h * (1.0 - value / y_max)

    grid = []
    for tick in range(6):
        value = y_max * tick / 5
        axis_y = y(value)
        grid.append(
            f'<line x1="{left}" y1="{axis_y:.2f}" x2="{left + plot_w}" '
            f'y2="{axis_y:.2f}" stroke="#d9dee8"/>'
            f'<text x="{left - 12}" y="{axis_y + 5:.2f}" text-anchor="end" '
            f'font-size="14">{value:.0f}</text>'
        )
    for index, qubit in enumerate(qubits):
        grid.append(
            f'<text x="{x(index):.2f}" y="{top + plot_h + 28}" '
            f'text-anchor="middle" font-size="14">{qubit}</text>'
        )

    shapes = []
    for backend, color in (("triton", "#2468b4"), ("eager", "#d24b40")):
        points = []
        for index, qubit in enumerate(qubits):
            record = records[(qubit, backend)]
            gib = int(record["peak_memory_bytes"]) / 2**30
            points.append(f"{x(index):.2f},{y(gib):.2f}")
            if record["status"] == "oom":
                shapes.append(
                    f'<path d="M {x(index) - 7:.2f} {y(gib) - 7:.2f} '
                    f'L {x(index) + 7:.2f} {y(gib) + 7:.2f} '
                    f'M {x(index) - 7:.2f} {y(gib) + 7:.2f} '
                    f'L {x(index) + 7:.2f} {y(gib) - 7:.2f}" '
                    f'stroke="{color}" stroke-width="3"/>'
                    f'<text x="{x(index):.2f}" y="{y(gib) - 13:.2f}" '
                    f'text-anchor="middle" font-size="13" fill="{color}">OOM</text>'
                )
            else:
                shapes.append(
                    f'<circle cx="{x(index):.2f}" cy="{y(gib):.2f}" r="5" '
                    f'fill="{color}"/>'
                )
        shapes.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{color}" stroke-width="3"/>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<g font-family="sans-serif">
<text x="{width / 2}" y="27" text-anchor="middle" font-size="20" font-weight="600">FlagQuantum VQE capacity scan (32 RX/RZ layers)</text>
{''.join(grid)}
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>
{''.join(shapes)}
<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-size="16">Number of Qubits</text>
<text x="22" y="{top + plot_h / 2}" text-anchor="middle" font-size="16" transform="rotate(-90 22 {top + plot_h / 2})">Peak Allocated GPU Memory (GiB)</text>
<line x1="110" y1="55" x2="145" y2="55" stroke="#2468b4" stroke-width="3"/><text x="154" y="60" font-size="14">FlagQuantum IR + Triton fusion</text>
<line x1="110" y1="78" x2="145" y2="78" stroke="#d24b40" stroke-width="3"/><text x="154" y="83" font-size="14">FlagQuantum PyTorch eager</text>
</g></svg>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
