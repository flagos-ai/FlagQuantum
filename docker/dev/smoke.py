"""Exercise the installed development stack without provider credentials."""

import argparse
from importlib.metadata import version

import numpy as np
import torch

import flagquantum as fq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu", action="store_true", help="Require real CUDA execution"
    )
    parser.add_argument(
        "--qsteed", action="store_true", help="Test the isolated compiler environment"
    )
    args = parser.parse_args()
    if args.qsteed and args.gpu:
        parser.error("The isolated compiler environment uses CPU PyTorch")
    device = "cuda" if args.gpu else "cpu"
    if args.gpu and not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access a CUDA device")
    if not args.qsteed:
        import jax
        import jax.numpy as jnp

        jax_device = jax.devices("gpu" if args.gpu else "cpu")[0]
        with jax.default_device(jax_device):
            np.testing.assert_allclose(jax.jit(lambda x: x @ x)(jnp.eye(2)), np.eye(2))
        print(f"JAX executed on {jax_device.platform}")
    x = torch.tensor([2.0], device=device, requires_grad=True)
    x.square().sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([4.0], device=device))
    circuit = fq.Circuit(3).h(0).cx(0, 2)
    if args.qsteed:
        compiled = fq.compile(
            circuit,
            compiler="qsteed",
            target={
                "basis_gates": ("h", "x", "rx", "ry", "rz", "cx"),
                "coupling_map": ((0, 1), (1, 2)),
            },
        )
        if not compiled.instructions:
            raise RuntimeError("QSteed returned an empty circuit")
    fq.run(circuit)
    packages = ("qsteed", "flagquantum-compiler-qsteed") if args.qsteed else ("jax",)
    for package in ("flagquantum", "torch", *packages):
        print(f"{package}={version(package)}")
    print(f"Development stack smoke passed: torch={device}, qsteed={args.qsteed}")


if __name__ == "__main__":
    main()
