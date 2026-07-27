"""Plot per-iteration Triton/eager VQE energy deviation from a result JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    triton = payload["triton"]["energies"]
    eager = payload["eager"]["energies"]
    deviations = [abs(left - right) for left, right in zip(triton, eager)]
    iterations = len(deviations)
    maximum = max(deviations)
    maximum_iteration = deviations.index(maximum) + 1
    mean = sum(deviations) / iterations
    final = deviations[-1]

    width, height = 900, 560
    left, right, top, bottom = 110, 35, 58, 76
    plot_w, plot_h = width - left - right, height - top - bottom
    y_max = maximum * 1.12

    def x(index: int) -> float:
        return left + plot_w * index / max(1, iterations - 1)

    def y(value: float) -> float:
        return top + plot_h * (1.0 - value / y_max)

    points = " ".join(
        f"{x(index):.2f},{y(value):.2f}"
        for index, value in enumerate(deviations)
    )
    grid = []
    for tick in range(6):
        value = y_max * tick / 5
        axis_y = y(value)
        grid.append(
            f'<line x1="{left}" y1="{axis_y:.2f}" x2="{left + plot_w}" '
            f'y2="{axis_y:.2f}" stroke="#d9dee8"/>'
            f'<text x="{left - 12}" y="{axis_y + 5:.2f}" text-anchor="end" '
            f'font-size="14">{value:.1e}</text>'
        )
    for tick in range(6):
        iteration = 1 + round((iterations - 1) * tick / 5)
        axis_x = x(iteration - 1)
        grid.append(
            f'<text x="{axis_x:.2f}" y="{top + plot_h + 28}" '
            f'text-anchor="middle" font-size="14">{iteration}</text>'
        )
    marker_x, marker_y = x(maximum_iteration - 1), y(maximum)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<g font-family="sans-serif">
<text x="{width / 2}" y="27" text-anchor="middle" font-size="20" font-weight="600">FlagQuantum VQE: IR+Triton vs PyTorch eager deviation</text>
<text x="{width / 2}" y="48" text-anchor="middle" font-size="14" fill="#445">precision: complex64 statevector / float32 parameter reduction</text>
{''.join(grid)}
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>
<polyline points="{points}" fill="none" stroke="#27866f" stroke-width="2"/>
<circle cx="{marker_x:.2f}" cy="{marker_y:.2f}" r="5" fill="#b33a3a"/>
<text x="{marker_x:.2f}" y="{marker_y - 10:.2f}" text-anchor="middle" font-size="13" fill="#b33a3a">max {maximum:.2e} @ {maximum_iteration}</text>
<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-size="16">Number of Iterations</text>
<text x="23" y="{top + plot_h / 2}" text-anchor="middle" font-size="16" transform="rotate(-90 23 {top + plot_h / 2})">Absolute Energy Deviation |E_Triton - E_eager|</text>
<rect x="{left + 15}" y="{top + 12}" width="260" height="70" rx="5" fill="white" fill-opacity="0.9" stroke="#bbc3cc"/>
<text x="{left + 28}" y="{top + 34}" font-size="14">final deviation: {final:.2e}</text>
<text x="{left + 28}" y="{top + 55}" font-size="14">mean deviation: {mean:.2e}</text>
<text x="{left + 28}" y="{top + 76}" font-size="14">max deviation: {maximum:.2e}</text>
</g></svg>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
