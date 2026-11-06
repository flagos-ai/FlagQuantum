"""Convert recorded CPU bits with the existing optional Stim extra."""

import torch

from flagquantum.qec.adapters.stim import convert_stim_measurements


def main() -> None:
    text = "R 0\nX 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    raw = torch.tensor([[1], [0]], dtype=torch.bool)
    result = convert_stim_measurements(text, raw)
    print(result.detectors.tolist())
    print(result.observables.tolist())


if __name__ == "__main__":
    main()
