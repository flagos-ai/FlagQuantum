# flagquantum/run_tests.py
#!/usr/bin/env python
import os
import shutil
import subprocess
import sys


def has_gpu():
    """Unified check for available accelerator."""
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=True
            )
            if result.returncode == 0:
                print("[OK] NVIDIA GPU detected")
                return True
        except (FileNotFoundError, subprocess.SubprocessError):
            pass

    if shutil.which("hy-smi"):
        try:
            result = subprocess.run(
                ["hy-smi", "-u"], capture_output=True, text=True, check=True
            )
            if result.returncode == 0:
                print("[OK] Hygon DCU detected")
                return True
        except (FileNotFoundError, subprocess.SubprocessError):
            pass

    if shutil.which("mthreads-gmi"):
        try:
            result = subprocess.run(
                ["mthreads-gmi", "-L"], capture_output=True, text=True, check=True
            )
            if result.returncode == 0:
                print("[OK] Mthreads GPU detected")
                os.environ["HAS_MTHREADS"] = "1"
                return True
        except (FileNotFoundError, subprocess.SubprocessError):
            pass

    print(
        "[INFO] No supported accelerator "
        "(NVIDIA GPU / Hygon DCU / Mthreads GPU) detected"
    )
    return False


def main():
    args = sys.argv[1:]

    if has_gpu():
        print("[INFO] GPU detected, running single-process tests first")
        commands = [[
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-v",
            "--ignore=tests/distributed",
        ] + args]
        print("[INFO] Running torchrun tests for distributed package")
        commands.append([
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "-m",
            "pytest",
            "tests/distributed",
            "-v",
        ] + args)
        
    else:
        print("[INFO] No GPU detected, running single process tests")
        commands = [[
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-v",
            "--ignore=tests/distributed",
        ] + args]

    for cmd in commands:
        print(f"Running command: {cmd}")
        result = subprocess.run(cmd, check=True)
        if result.returncode != 0:
            sys.exit(result.returncode)
    sys.exit(0)


if __name__ == "__main__":
    main()
