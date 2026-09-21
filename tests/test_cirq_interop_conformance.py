from __future__ import annotations

import math
import random
from importlib import import_module

import pytest
import torch

from flagquantum import Circuit
from flagquantum.ecosystem.cirq import from_cirq, to_cirq

cirq = pytest.importorskip("cirq")
np = import_module("numpy")
pytestmark = [pytest.mark.integration, pytest.mark.cirq]

_SEEDS = (173, 419, 733, 1087, 1429, 1871)
_CATALOG = (
    ("x", 1),
    ("y", 1),
    ("z", 1),
    ("h", 1),
    ("s", 1),
    ("t", 1),
    ("rx", 1),
    ("ry", 1),
    ("rz", 1),
    ("cx", 2),
    ("cz", 2),
    ("swap", 2),
    ("ccx", 3),
    ("cswap", 3),
)


def _cirq_gate(name: str, angle: float | None):
    fixed = {
        "x": cirq.X,
        "y": cirq.Y,
        "z": cirq.Z,
        "h": cirq.H,
        "s": cirq.S,
        "t": cirq.T,
        "cx": cirq.CNOT,
        "cz": cirq.CZ,
        "swap": cirq.SWAP,
        "ccx": cirq.CCX,
        "cswap": cirq.CSWAP,
    }
    if name == "rx":
        return cirq.rx(angle)
    if name == "ry":
        return cirq.ry(angle)
    if name == "rz":
        return cirq.rz(angle)
    return fixed[name]


@pytest.mark.parametrize("seed", _SEEDS)
def test_seeded_cirq_differential_round_trip(seed: int) -> None:
    rng = random.Random(seed)
    n_wires = 3 + seed % 3
    flagquantum = Circuit(n_wires, dtype=torch.complex128)
    moments = []
    for wire in range(n_wires):
        angle = rng.uniform(-math.pi, math.pi)
        flagquantum = flagquantum.ry(wire, theta=angle)
        moments.append(cirq.Moment([cirq.ry(angle)(cirq.LineQubit(wire))]))
    for _ in range(14):
        name, arity = rng.choice(_CATALOG)
        wires = tuple(rng.sample(range(n_wires), arity))
        angle = rng.uniform(-math.pi, math.pi) if name in {"rx", "ry", "rz"} else None
        params = {"theta": angle} if angle is not None else {}
        flagquantum = flagquantum.gate(name, wires, params=params)
        gate = _cirq_gate(name, angle)
        moments.append(
            cirq.Moment([gate.on(*(cirq.LineQubit(wire) for wire in wires))])
        )
    native = cirq.Circuit(moments)

    expected = flagquantum.state()[0]
    native_state = torch.as_tensor(
        cirq.Simulator(dtype=np.complex128)
        .simulate(native, qubit_order=cirq.LineQubit.range(n_wires))
        .final_state_vector.copy()
    )
    torch.testing.assert_close(native_state, expected, atol=1e-10, rtol=1e-10)

    imported = from_cirq(native)
    imported_state = Circuit.from_ir(imported, dtype=torch.complex128).state()[0]
    torch.testing.assert_close(imported_state, expected, atol=1e-10, rtol=1e-10)

    exported = to_cirq(imported)
    exported_state = torch.as_tensor(
        cirq.Simulator(dtype=np.complex128)
        .simulate(exported, qubit_order=cirq.LineQubit.range(n_wires))
        .final_state_vector.copy()
    )
    torch.testing.assert_close(exported_state, expected, atol=1e-10, rtol=1e-10)
