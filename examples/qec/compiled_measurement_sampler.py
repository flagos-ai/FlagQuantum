"""Sample explicit Clifford measurements and verify their b8 file."""

from pathlib import Path
from tempfile import TemporaryDirectory

import stim

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
    )
    sampler = circuit.compile_sampler(seed=7)
    records = sampler.sample(shots=32)
    assert tuple(records.shape) == (32, 2)
    assert bool((records[:, 0] ^ records[:, 1]).all())
    packed = sampler.sample(shots=32, bit_packed=True)
    assert tuple(packed.shape) == (32, 1)
    assert set(packed[:, 0].tolist()) <= {1, 2}
    with TemporaryDirectory() as directory:
        path = Path(directory) / "measurements.b8"
        sampler.sample_write(shots=32, filepath=path, format="b8")
        readback = stim.read_shot_data_file(path=path, format="b8", num_measurements=2)
        assert readback.shape == (32, 2)
        assert (readback[:, 0] ^ readback[:, 1]).all()
        assert path.read_bytes() == bytes(a + 2 * b for a, b in readback.tolist())
        print("records", tuple(records.shape))
        print("packed", tuple(packed.shape))
        print("b8 bytes", path.stat().st_size)


if __name__ == "__main__":
    main()
