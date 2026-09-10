"""Run the shortest local statevector simulation."""

import torch

import flagquantum as fq


def main() -> None:
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    result = fq.run(circuit)

    expected = torch.tensor(
        [[2**-0.5, 0.0, 0.0, 2**-0.5]], dtype=torch.complex64
    )
    torch.testing.assert_close(result.to_statevector(), expected)
    print(result.to_statevector())


if __name__ == "__main__":
    main()
