"""Compare Bell Z-basis counts with the ideal distribution, including replay provenance."""

import argparse
import json
from pathlib import Path


def main(path):
    record = json.loads(Path(path).read_text())
    if record.get("status") != "completed":
        raise ValueError("A completed result is required")
    counts = record["counts"]
    if not counts or any(
        k not in ("00", "01", "10", "11") or type(v) is not int or v < 0
        for k, v in counts.items()
    ):
        raise ValueError("Expected nonnegative integer counts for two-bit outcomes")
    shots = sum(counts.values())
    if shots <= 0 or shots != record["shots"]:
        raise ValueError("Counts total must match shots")
    observed = [counts.get(k, 0) / shots for k in ("00", "01", "10", "11")]
    ideal = [0.5, 0.0, 0.0, 0.5]
    return {
        "source": record["source"],
        "task_id": record["task_id"],
        "shots": shots,
        "ideal": ideal,
        "observed": observed,
        "total_variation_distance": sum(abs(a - b) for a, b in zip(ideal, observed))
        / 2,
        "odd_parity_probability": observed[1] + observed[2],
        "interpretation": "Z-basis distribution comparison only; not state fidelity or entanglement certification",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    print(json.dumps(main(parser.parse_args().result), indent=2))
