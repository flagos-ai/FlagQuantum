#!/usr/bin/env python
import shutil
import subprocess
import sys


def has_hygon_dcu():
    """检测是否有海光 DCU"""
    # 1. 检查 hy-smi 工具是否存在
    if shutil.which("hy-smi") is None:
        return False

    try:
        # 2. 运行 hy-smi 并检查返回码，确认驱动是否正常工作
        result = subprocess.run(
            ["hy-smi", "-u"],  # -u 参数用于查看使用率，可用来测试
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.SubprocessError):
        return False

def has_gpu():
    """统一检测是否有可用的加速卡 (NVIDIA GPU 或 海光 DCU)"""
    # 先尝试检测 NVIDIA GPU
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=True
            )
            if result.returncode == 0:
                print("✓ NVIDIA GPU detected")
                return True
        except (FileNotFoundError, subprocess.SubprocessError):
            pass  # 不是 NVIDIA，继续尝试其他

    # 再尝试检测海光 DCU
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

    # 都没有检测到
    print("✗ No supported accelerator (NVIDIA GPU / Hygon DCU) detected")
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
