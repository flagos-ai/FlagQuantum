"""Submit a Bell circuit as a Jiuding batch job without writing a job script."""

import argparse
import json

import flagquantum as fq
from flagquantum.remote.compute import JiudingClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", help="runtime image for a new Job")
    mode.add_argument("--restore-job", metavar="JOB_ID", help="resume an existing Job")
    parser.add_argument("--target", default="jiuding:gpu", help="new Jobs only")
    parser.add_argument(
        "--detach",
        action="store_true",
        help="print the new Job ID without waiting for its result",
    )
    args = parser.parse_args()
    if args.detach and args.restore_job:
        parser.error("--detach is only valid with --image")

    client = JiudingClient(workspace=args.workspace)
    if args.restore_job:
        receipt = client.restore_receipt(args.restore_job)
    else:
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
    if args.detach:
        return

    result = client.result(receipt, timeout=600)
    print(result.counts)


if __name__ == "__main__":
    main()
