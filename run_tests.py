# flagquantum/run_tests.py
#!/usr/bin/env python
import os
import shutil
import subprocess
import sys


def has_gpu():
    """Unified check for available accelerator (NVIDIA GPU or Hygon DCU)"""
    # First try to detect NVIDIA GPU
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=True
            )
            if result.returncode == 0:
                print("✓ NVIDIA GPU detected")
                return True
        except (FileNotFoundError, subprocess.SubprocessError):
            pass  # Not NVIDIA, continue to try other options

    # Then try to detect Hygon DCU
    if shutil.which("hy-smi"):
        try:
            result = subprocess.run(
                ["hy-smi", "-u"], capture_output=True, text=True, check=True
            )
            if result.returncode == 0:
                print("✓ Hygon DCU detected")
                return True
        except (FileNotFoundError, subprocess.SubprocessError):
            pass

    # Then try to detect Mthreads GPU
    if shutil.which("mthreads-gmi"):
        try:
            result = subprocess.run(
                ["mthreads-gmi", "-L"], capture_output=True, text=True, check=True
            )
            if result.returncode == 0:
                print("✓ Mthreads GPU detected")
                os.environ["HAS_MTHREADS"] = "1"
                return True
        except (FileNotFoundError, subprocess.SubprocessError):
            pass

    # No accelerator detected
    print("✗ No supported accelerator (NVIDIA GPU / Hygon DCU / Mthreads GPU) detected")
    return False


def main():
    args = sys.argv[1:]

    if has_gpu():
        print("✓ GPU detected, running distributed tests")
        cmd = ["torchrun", "--nproc_per_node=2", "-m", "pytest", "tests/", "-v"] + args
        print(f"Running command: {cmd}")
        result = subprocess.run(cmd, check=True)
    else:
        print("✗ No GPU detected, running single process tests")
        cmd = ["pytest", "tests/", "-v", "--ignore=tests/distributed"] + args
        result = subprocess.run(cmd, check=True)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
