"""Read-only GPU inventory probe (transient CI diagnostic)."""
import subprocess


def _run(args):
    proc = subprocess.run(args, capture_output=True, text=True, timeout=120)
    return ((proc.stdout or "") + (proc.stderr or "")).strip()


def test_prl_gpu_probe():
    listed = _run(["nvidia-smi", "-L"])
    detail = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,driver_version",
        "--format=csv",
    ])
    block = "=== PRL-GPU-PROBE-BEGIN ===\n" + listed + "\n" + detail + "\n=== PRL-GPU-PROBE-END ==="
    print(block)
    try:
        with open("hardware-run/prl-gpu-probe.txt", "w", encoding="utf-8") as handle:
            handle.write(block + "\n")
    except OSError:
        pass
    assert "GPU 0:" in listed, "PRL GPU probe output:\n" + block
