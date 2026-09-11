"""Validate the workshop's local numerical results without remote submissions."""

import json
from pathlib import Path
from bell import main as bell
from train import main as train
from compare import main as compare


def main():
    circuit = bell()
    training = train()
    if training["final_loss"] >= 0.01:
        raise AssertionError("Training did not reach the workshop target")
    replay = compare(
        Path(__file__).resolve().parents[1] / "reference_results/quafu_bell_replay.json"
    )
    if replay["source"] != "historical_replay":
        raise AssertionError("Reference data must retain its replay label")
    report = {
        "bell": circuit,
        "training_final_loss": training["final_loss"],
        "replay": replay,
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
