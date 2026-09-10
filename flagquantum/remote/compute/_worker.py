"""Execute one shared Python entrypoint and atomically write its JSON result."""

import argparse
import json
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script")
    parser.add_argument("output")
    parser.add_argument("run_id")
    parser.add_argument("--gpus", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    script, output, run_id = args.script, args.output, args.run_id
    worker = {"requested_gpus": args.gpus}
    if args.gpus:
        from flagquantum.compute.registry import get_platform_runtime

        devices = get_platform_runtime("cuda").discover()
        if len(devices) != args.gpus:
            raise RuntimeError(
                "Requested one GPU but worker does not see exactly one CUDA device"
            )
        worker.update(
            visible_cuda_devices=len(devices),
            cuda_device_name=devices[0].name,
        )
    sys.path.insert(0, str(Path(script).parent))
    entrypoint = runpy.run_path(script)["main"]
    value = entrypoint()
    result = Path(output)
    temporary = result.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"run_id": run_id, "value": value, "worker": worker}, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(result)


if __name__ == "__main__":
    main()
