"""Bell-state forward experiment; main() is also the Jiuding entrypoint."""

import argparse
import json
import torch
import flagquantum as fq


def main(device="cpu"):
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    result = fq.run(
        circuit, options=fq.ExecutionOptions(device=device, mode="statevector")
    )
    state = result.to_statevector().reshape(-1)
    if state.device.type != torch.device(device).type:
        raise RuntimeError("Unexpected execution device")
    expected = torch.tensor(
        [2**-0.5, 0, 0, 2**-0.5], dtype=state.dtype, device=state.device
    )
    torch.testing.assert_close(state, expected, atol=1e-6, rtol=1e-6)
    return {
        "experiment": "bell",
        "device": str(state.device),
        "dtype": str(state.dtype),
        "probabilities": state.abs().square().detach().cpu().tolist(),
        "max_abs_error": float((state - expected).abs().max().detach().cpu()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    print(json.dumps(main(parser.parse_args().device), indent=2))
