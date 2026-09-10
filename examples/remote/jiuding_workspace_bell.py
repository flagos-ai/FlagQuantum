"""Verify a Jiuding workspace by running a Bell circuit."""

import argparse
import json
import time

import torch

import flagquantum as fq
from flagquantum.remote.compute import JiudingClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", help="auto-selected when exactly one is visible")
    parser.add_argument("--target", default="jiuding:cpu")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--list-workspaces",
        action="store_true",
        help="list visible workspaces without running a circuit",
    )
    args = parser.parse_args()
    client = JiudingClient(workspace=args.workspace)
    if args.list_workspaces:
        print(json.dumps(client.list_workspaces(), indent=2))
        return

    circuit = fq.Circuit(2).h(0).cx(0, 1)
    expected = torch.tensor([[2**-0.5, 0, 0, 2**-0.5]], dtype=torch.complex64)

    with client:
        workspace = client.workspace()["name"]
        for index in range(args.repeats):
            started = time.perf_counter()
            result = client.run_statevector(circuit, target=args.target)
            torch.testing.assert_close(result.to_statevector(), expected)
            print(
                {
                    "run": index + 1,
                    "workspace": workspace,
                    "total_seconds": time.perf_counter() - started,
                    "execute_seconds": result.runtime["elapsed_seconds"],
                    "device": result.runtime["device"],
                }
            )


if __name__ == "__main__":
    main()
