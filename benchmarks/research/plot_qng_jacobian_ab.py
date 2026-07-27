"""Plot scalar-versus-vectorized exact-QNG A/B results on one accelerator."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def total(path: Path) -> tuple[float, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return sum(row["wall_time_seconds"] for row in payload["records"]), payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n4-scalar", type=Path, required=True)
    parser.add_argument("--n4-vectorized", type=Path, required=True)
    parser.add_argument("--n6-scalar", type=Path, required=True)
    parser.add_argument("--n6-vectorized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = (
        ("N=4, depth=2", args.n4_scalar, args.n4_vectorized),
        ("N=6, depth=3", args.n6_scalar, args.n6_vectorized),
    )
    scalar, vectorized, speedups = [], [], []
    for _, scalar_path, vectorized_path in cases:
        scalar_total, scalar_payload = total(scalar_path)
        vectorized_total, vectorized_payload = total(vectorized_path)
        scalar.append(scalar_total / len(scalar_payload["records"]))
        vectorized.append(vectorized_total / len(vectorized_payload["records"]))
        speedups.append(scalar[-1] / vectorized[-1])
    x = range(len(cases))
    fig, axis = plt.subplots(figsize=(8.2, 5.0))
    axis.bar([i - 0.18 for i in x], scalar, width=0.36, label="Scalar-output Jacobian")
    bars = axis.bar([i + 0.18 for i in x], vectorized, width=0.36, label="Vectorized reverse Jacobian")
    for bar, speedup in zip(bars, speedups):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{speedup:.2f}x", ha="center", va="bottom")
    axis.set_xticks(list(x), [case[0] for case in cases])
    axis.set_ylabel("Mean exact-QNG step time (s)")
    axis.set_title("A800 exact-QNG baseline optimization")
    axis.grid(alpha=0.2, axis="y")
    axis.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".svg"))
    fig.savefig(args.output.with_suffix(".png"), dpi=180)


if __name__ == "__main__":
    main()
