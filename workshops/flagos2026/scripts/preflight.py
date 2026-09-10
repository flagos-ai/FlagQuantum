"""Read-only checks. Does not authenticate or submit remote work."""

import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path
import torch
import flagquantum as fq
from bell import main as bell


def main():
    try:
        plugin = importlib.metadata.version("flagquantum-compiler-qsteed")
    except importlib.metadata.PackageNotFoundError:
        plugin = None
    report = {
        "python": sys.version.split()[0],
        "flagquantum": fq.__version__,
        "source": fq.__file__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "visible_gpus": torch.cuda.device_count(),
        "notebook_dependencies": {
            name: importlib.util.find_spec(name) is not None
            for name in ("jupyterlab", "matplotlib", "ipykernel")
        },
        "quafu_token_present": bool(os.getenv("QUAFU_API_TOKEN")),
        "qsteed_plugin": plugin,
        "jiuding_credentials_present": bool(
            os.getenv("JIUDING_AK") and os.getenv("JIUDING_SK")
        )
        or all(
            Path("/etc/accesskey", name).is_file() for name in ("user-ak", "user-sk")
        ),
        "cpu_check": bell(),
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
