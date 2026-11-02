"""Verify retained producer hashes and recompute optimizer trajectory errors."""

import hashlib
import json
import subprocess
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[2]
receipt = json.loads((DIRECTORY / "optimizer-capture.json").read_text())
publication = json.loads((DIRECTORY / "publication.json").read_text())
for name, expected in publication.items():
    assert hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest() == expected
assert receipt["source_dirty"] is False
for relative, field in (
    ("tests/unit/test_training_compilation_contract_fake.py", "fake_sha256"),
    (
        "artifacts/development/torch_training_contract_20261102/reproduce_optimizer_capture.py",
        "producer_sha256",
    ),
):
    source = subprocess.check_output(
        ["git", "show", f"{receipt['source_commit']}:{relative}"], cwd=ROOT, timeout=20
    )
    assert hashlib.sha256(source).hexdigest() == receipt[field]
for result in receipt["results"]:
    for kind in ("parameters", "losses"):
        error = max(
            abs(a - b)
            for a, b in zip(
                result[f"eager_{kind}"], result[f"captured_{kind}"], strict=True
            )
        )
        assert (
            error
            == result[
                "parameter_max_error" if kind == "parameters" else "loss_max_error"
            ]
        )
    assert result["graph_count"] > 0
assert receipt["results"][0]["status"] == "conformant"
assert receipt["results"][1]["status"] == "unsupported_precision"
print("Producer hashes, log/receipt hashes and retained SGD/Adam errors verified")
