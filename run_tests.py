#!/usr/bin/env python
import shutil
import subprocess
import sys


def has_gpu():
    """检测是否有 NVIDIA GPU"""
    if shutil.which("nvidia-smi") is None:
        return False

    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, shell=True
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


def main():
    args = sys.argv[1:]

    if has_gpu():
        print("✓ GPU detected, running distributed tests")
        # 使用 shell=True 让命令在 shell 中执行
        cmd = f"torchrun --nproc_per_node=2 -m pytest tests/ -v {' '.join(args)}"
        print(f"Running command: {cmd}")
        result = subprocess.run(cmd, shell=True)
    else:
        print("✗ No GPU detected, running single process tests")
        cmd = ["pytest", "tests/", "-v", "--ignore=tests/distributed"] + args
        result = subprocess.run(cmd)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
