"""Sample explicit Clifford measurements and write six compatible record formats."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit


def main() -> None:
    circuit = (
        DynamicCircuit(3)
        .h(0)
        .measure(0, classical_bit=0)
        .conditional("x", 1, classical_bit=0)
        .reset(0)
        .conditional("x", 0, classical_bit=0, equals=0)
        .measure(0, classical_bit=0)
        .conditional("x", 2, classical_bit=0)
        .measure(1)
        .measure(2)
    )
    sampler = circuit.compile_sampler(seed=7)
    records = sampler.sample(shots=32)
    assert tuple(records.shape) == (32, 4)
    assert all(row in ([0, 1, 0, 1], [1, 0, 1, 0]) for row in records.tolist())
    packed = sampler.sample(shots=32, bit_packed=True)
    assert tuple(packed.shape) == (32, 1)
    assert set(packed[:, 0].tolist()) <= {0x05, 0x0A}
    with TemporaryDirectory() as directory:
        path = Path(directory) / "measurements.b8"
        sampler.sample_write(shots=32, filepath=path, format="b8")
        data = path.read_bytes()
        assert len(data) == 32
        assert set(data) <= {0x05, 0x0A}
        print("records", tuple(records.shape))
        print("packed", tuple(packed.shape))
        print("b8 bytes", len(data))

    program = (
        DynamicCircuit(1)
        .x(0)
        .measure(0, classical_bit=0)
        .reset(0)
        .measure(0, classical_bit=0)
    )
    native = fq.Circuit.from_ir(program.to_ir())
    expected: dict[Literal["01", "b8", "r8", "hits", "dets", "ptb64"], bytes] = {
        "01": b"10\n" * 64,
        "b8": b"\x01" * 64,
        "r8": b"\x00\x01" * 64,
        "hits": b"0\n" * 64,
        "dets": b"shot M0\n" * 64,
        "ptb64": b"\xff" * 8 + b"\x00" * 8,
    }
    with TemporaryDirectory() as directory:
        for format, reference in expected.items():
            path = Path(directory) / f"records.{format}"
            native.compile_sampler(seed=7).sample_write(
                64, filepath=path, format=format
            )
            assert path.read_bytes() == reference
            print(format, path.stat().st_size)


if __name__ == "__main__":
    main()
