"""Sample explicit Clifford measurements and verify their b8 file."""

from pathlib import Path
from tempfile import TemporaryDirectory

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


if __name__ == "__main__":
    main()
