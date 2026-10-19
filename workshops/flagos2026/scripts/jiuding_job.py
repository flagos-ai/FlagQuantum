"""Explicit Jiuding submission or receipt-based status/result/cancel."""

import argparse
import json
from pathlib import Path

from flagquantum.remote.compute.jiuding import JiudingClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("submit", "status", "result", "cancel"))
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--image")
    parser.add_argument("--workspace")
    parser.add_argument(
        "--target",
        help="Jiuding compute target, for example jiuding:cpu or jiuding:gpu/A100",
    )
    parser.add_argument("--python", default="python3")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    client = JiudingClient(workspace=args.workspace)
    if args.action == "submit":
        if not args.image:
            parser.error("submit requires --image")
        if not args.target:
            parser.error("submit requires --target")
        root = Path(__file__).resolve().parents[3]
        gpu = args.target.partition(":")[2].partition("/")[0].lower() == "gpu"
        script = (
            root / "examples/remote/jiuding_bell_gpu.py"
            if gpu
            else Path(__file__).with_name("bell.py")
        )
        receipt = client.submit(
            script,
            target=args.target,
            image=args.image,
            receipt=args.receipt,
            python=args.python,
            pythonpath=root,
            cpus=4 if gpu else 2,
            memory_gib=8 if gpu else 2,
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
