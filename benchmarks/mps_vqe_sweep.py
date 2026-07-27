"""Run a batch-one MPS VQE qubit/bond/depth sweep."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _integers(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=_integers, default=(32, 64, 128, 256))
    parser.add_argument("--bonds", type=_integers, default=(16, 32, 64, 128))
    parser.add_argument("--depths", type=_integers, default=(4, 8, 12))
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--cutoff", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runner = Path(__file__).with_name("mps_vqe_triton_runtime.py")
    temporary = args.output.with_suffix(".case.json")
    cases = []
    for qubits in args.qubits:
        for bond in args.bonds:
            for depth in args.depths:
                command = [
                    sys.executable,
                    str(runner),
                    "--n-wires", str(qubits),
                    "--layers", str(depth),
                    "--batch-size", "1",
                    "--max-bond", str(bond),
                    "--cutoff", str(args.cutoff),
                    "--iterations", str(args.iterations),
                    "--backend", "both",
                    "--json-output", str(temporary),
                ]
                print(
                    f"case qubits={qubits} bond={bond} depth={depth}", flush=True
                )
                subprocess.run(command, check=True)
                cases.append(json.loads(temporary.read_text(encoding="utf-8")))
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps({"batch_size": 1, "cases": cases}, indent=2) + "\n",
                    encoding="utf-8",
                )
    temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
