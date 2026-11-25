"""Compare every product site without allocating a 2**64 amplitude vector."""

import torch

import flagquantum as fq
from examples.synthesis.tutorials.basics.mcx.main import build_program


def reference_circuit() -> fq.Circuit:
    # All original controls start at zero: this independent oracle is identity.
    return fq.Circuit(64, dtype=torch.complex128)


def run_example() -> None:
    program = build_program()
    actual = fq.run(program)
    expected = fq.run(
        reference_circuit().to_ir(),
        options=fq.ExecutionOptions(
            mode="mps",
            backend="torch",
            device="cpu",
            precision="complex128",
            allow_approximate=False,
            allow_backend_fallback=False,
            memory_limit_bytes=2048,
        ),
    )
    actual_sites = getattr(actual.native(), "tensors")
    expected_sites = getattr(expected.native(), "tensors")
    assert len(actual_sites) == len(expected_sites) == 64
    for left, right in zip(actual_sites, expected_sites, strict=True):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=0)
    assert dict(program.probabilities(actual, registers=("target",))) == {(0,): 1.0}
    print("Exact native MPS agrees with the independent product-state Circuit.")


if __name__ == "__main__":
    run_example()
