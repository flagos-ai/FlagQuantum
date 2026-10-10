"""Read-only checks. Does not authenticate or submit remote work."""

import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from bell import main as bell

import flagquantum as fq


def source_commit():
    recorded = os.getenv("FLAGQUANTUM_SOURCE_COMMIT")
    if recorded:
        return recorded
    source = Path(fq.__file__).resolve()
    try:
        return subprocess.run(
            ["git", "-C", str(source.parent.parent), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def main():
    try:
        plugin = importlib.metadata.version("flagquantum-compiler-qsteed")
    except importlib.metadata.PackageNotFoundError:
        plugin = None
    report = {
        "python": sys.version.split()[0],
        "flagquantum": fq.__version__,
        "source": fq.__file__,
        "source_commit": source_commit(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "visible_gpus": torch.cuda.device_count(),
        "notebook_dependencies": {
            name: importlib.util.find_spec(name) is not None
            for name in ("jupyterlab", "matplotlib", "ipykernel")
        },
        "quafu_api_key_present": bool(os.getenv("QUAFU_API_KEY")),
        "quafu_task_server_url_present": bool(os.getenv("QUAFU_TASK_SERVER_URL")),
        "quafu_legacy_token_present": bool(os.getenv("QUAFU_API_TOKEN")),
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
