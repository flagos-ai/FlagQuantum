"""Submit a Bell circuit as a Jiuding batch job without writing a job script."""

import argparse
import json

import flagquantum as fq
from flagquantum.remote.compute.jiuding import JiudingClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--workspace")
    parser.add_argument("--target", default="jiuding:gpu")
    args = parser.parse_args()

    client = JiudingClient(workspace=args.workspace)
    receipt = client.submit_program(
        fq.Circuit(2).h(0).cx(0, 1),
        target=args.target,
        image=args.image,
        outputs=fq.counts(),
        shots=1024,
        cpus=4,
        memory_gib=8,
    )
    print(json.dumps({"job_id": receipt["jobId"]}))

    result = client.result(receipt, timeout=600)
    print(result.counts)


if __name__ == "__main__":
    main()
