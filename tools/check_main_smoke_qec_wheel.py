"""Exercise the installed wheel's default predecoder after an isolated install.

The smoke job copies this checker and the public-facing example out of the
checkout and runs the checker with the clean environment's Python in isolated mode.
"""

from __future__ import annotations

import json
import runpy
import sys
from contextlib import redirect_stdout
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch


def main() -> None:
    assert sys.flags.isolated
    assert find_spec("pymatching") is None
    try:
        version("pymatching")
    except PackageNotFoundError:
        pass
    else:
        raise AssertionError("PyMatching distribution is installed")

    import flagquantum as fq
    from flagquantum.qec.decoders import _sparse_blossom
    from flagquantum.qec.decoders.sparse_blossom import SparseBlossomDecoder

    prefix = Path(sys.prefix).resolve()
    for module in (fq, _sparse_blossom):
        assert Path(module.__file__).resolve().is_relative_to(prefix), module.__file__

    example = Path(__file__).with_name("ai_predecoder.py")
    decode_batch = SparseBlossomDecoder.decode_batch
    batch_sizes: list[int] = []

    def record_batch(
        decoder: SparseBlossomDecoder, events: torch.Tensor
    ) -> torch.Tensor:
        batch_sizes.append(events.shape[0])
        return decode_batch(decoder, events)

    # Original and residual syndromes each use two full batches and one tail.
    batch_size = 128
    tail = 13
    shots = 2 * batch_size + tail
    with TemporaryDirectory(prefix="fq-predecoder-") as directory:
        for model in ("mlp", "cnn"):
            batch_sizes.clear()
            checkpoint = Path(directory) / f"{model}.pt"
            args = [
                str(example),
                "--model",
                model,
                "--steps",
                "1",
                "--shots",
                str(shots),
                "--checkpoint",
                str(checkpoint),
            ]
            with (
                patch.object(SparseBlossomDecoder, "decode_batch", record_batch),
                patch.object(sys, "argv", args),
                redirect_stdout(StringIO()) as output,
            ):
                runpy.run_path(str(example), run_name="__main__")
            report = json.loads(output.getvalue())
            assert report["model"] == model
            assert report["backend"] == "sparse_blossom"
            assert report["shots"] == shots
            assert batch_sizes == [batch_size] * 4 + [tail] * 2, batch_sizes
            for path in ("baseline", "hybrid"):
                failures = report[f"{path}_failures"]
                assert type(failures) is int and 0 <= failures <= shots
                assert report[f"{path}_ler"] == failures / shots
            print(f"{model}: {shots} shots with sparse_blossom")


if __name__ == "__main__":
    main()
