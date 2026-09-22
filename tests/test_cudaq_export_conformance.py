from __future__ import annotations

import random
from importlib import import_module

import pytest
import torch

from flagquantum import Circuit
from flagquantum.ecosystem.cudaq import to_cudaq

cudaq = pytest.importorskip("cudaq")
np = import_module("numpy")
pytestmark = [pytest.mark.integration, pytest.mark.cudaq]
cudaq.set_target("qpp-cpu")


def _state(kernel: object, n_wires: int) -> torch.Tensor:
    raw = np.asarray(cudaq.get_state(kernel))
    reordered = raw.reshape((2,) * n_wires).transpose(tuple(reversed(range(n_wires))))
    return torch.as_tensor(reordered.reshape(-1).copy(), dtype=torch.complex128)


@pytest.mark.parametrize("seed", range(5))
def test_seeded_cudaq_statevector_conformance(seed: int) -> None:
    rng = random.Random(seed)
    circuit = Circuit(3, dtype=torch.complex128)
    for _ in range(10):
        gate = rng.choice(
            ("x", "y", "z", "h", "s", "t", "rx", "ry", "rz", "cx", "cz", "swap")
        )
        if gate in {"rx", "ry", "rz"}:
            getattr(circuit, gate)(rng.randrange(3), theta=rng.uniform(-2.0, 2.0))
        elif gate in {"cx", "cz", "swap"}:
            left, right = rng.sample(range(3), 2)
            getattr(circuit, gate)(left, right)
        else:
            getattr(circuit, gate)(rng.randrange(3))
    expected = circuit.state()[0]
    torch.testing.assert_close(
        _state(to_cudaq(circuit), 3), expected, atol=1e-6, rtol=1e-6
    )
