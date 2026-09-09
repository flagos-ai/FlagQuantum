"""Run repeated Bell circuits through one warm Jiuding A100 executor."""

import argparse
import time

import torch

import flagquantum as fq
from flagquantum.remote.compute.jiuding import JiudingClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    expected = torch.tensor([[2**-0.5, 0, 0, 2**-0.5]], dtype=torch.complex64)

    with JiudingClient(workspace=args.workspace) as client:
        for index in range(args.repeats):
            started = time.perf_counter()
            result = client.run_statevector(circuit, target=args.target)
            torch.testing.assert_close(result.to_statevector(), expected)
            print(
                {
                    "run": index + 1,
                    "total_seconds": time.perf_counter() - started,
                    "execute_seconds": result.runtime["elapsed_seconds"],
                    "device": result.runtime["device"],
                }
            )


if __name__ == "__main__":
    main()
