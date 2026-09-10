"""Explicit Jiuding submission or receipt-based status/result/cancel."""

import argparse
import json
import sys
from pathlib import Path
from flagquantum.remote.compute.jiuding import JiudingClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("submit", "status", "result", "cancel"))
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--image")
    parser.add_argument("--workspace")
    parser.add_argument("--gpus", type=int, choices=(0, 1), default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    client = JiudingClient(workspace=args.workspace)
    if args.action == "submit":
        if not args.image:
            parser.error("submit requires --image")
        root = Path(__file__).resolve().parents[3]
        script = (
            root / "examples/remote/jiuding_bell_gpu.py"
            if args.gpus
            else Path(__file__).with_name("bell.py")
        )
        receipt = client.submit(
            script,
            image=args.image,
            receipt=args.receipt,
            python=args.python,
            pythonpath=root,
            gpus=args.gpus,
            cpus=4 if args.gpus else 2,
            memory_gib=8 if args.gpus else 2,
        )
        print(json.dumps(receipt, indent=2))
        return
    receipt = json.loads(args.receipt.read_text())
    if args.action == "result":
        value = client.result(receipt, timeout=args.timeout)
    elif args.action == "cancel":
        client.cancel(receipt)
        value = client.status(receipt)
    else:
        value = client.status(receipt)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
