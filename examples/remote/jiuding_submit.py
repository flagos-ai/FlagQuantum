"""Submit the Bell example from an existing Jiuding workspace."""

import argparse
import json
from pathlib import Path

from flagquantum.remote.compute.jiuding import JiudingClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Image from client.images()")
    parser.add_argument(
        "--receipt", required=True, type=Path, help="New receipt path on shared storage"
    )
    parser.add_argument("--workspace", help="Defaults to current platform pod")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    client = JiudingClient(workspace=args.workspace)
    receipt = client.submit(
        root / "examples/remote/jiuding_bell.py",
        image=args.image,
        receipt=args.receipt,
        pythonpath=root,
    )
    print(json.dumps(receipt), flush=True)
    value = client.result(receipt, timeout=180)
    print(json.dumps({"jobs": client.status(receipt), "result": value}), flush=True)


if __name__ == "__main__":
    main()
