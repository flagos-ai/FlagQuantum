from __future__ import annotations

import math
import random
from importlib import import_module

import pytest
import torch

from flagquantum import Circuit
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.ecosystem.braket import from_braket, to_braket

braket = pytest.importorskip("braket.circuits")
np = import_module("numpy")
pytestmark = [pytest.mark.integration, pytest.mark.braket]

_SEEDS = (181, 431, 743, 1091, 1433, 1877)
_CATALOG = tuple(
    (name, schema.arity)
    for name, schema in OPERATOR_SCHEMAS.items()
    if name
    not in {
        "u1",
        "u2",
        "crx",
        "cry",
        "crz",
        "bit_flip",
        "phase_flip",
        "depolarizing",
        "amplitude_damping",
    }
)


@pytest.mark.parametrize("seed", _SEEDS)
def test_seeded_braket_differential_round_trip(seed: int) -> None:
    rng = random.Random(seed)
    n_wires = 3 + seed % 3
    native = Circuit(n_wires, dtype=torch.complex128)
    for wire in range(n_wires):
        native = native.ry(wire, theta=rng.uniform(-math.pi, math.pi))
    for _ in range(18):
        name, arity = rng.choice(_CATALOG)
        schema = OPERATOR_SCHEMAS[name]
        wires = tuple(rng.sample(range(n_wires), arity))
        params = {
            parameter: rng.uniform(-math.pi, math.pi) for parameter in schema.parameters
        }
        native = native.gate(name, wires, params=params)

    expected = native.state()[0]
    external = to_braket(native.to_ir())
    external_state = torch.as_tensor(
        np.asarray(external.to_unitary(), dtype=np.complex128)[:, 0].copy()
    )
    torch.testing.assert_close(external_state, expected, atol=1e-10, rtol=1e-10)

    imported = from_braket(external)
    imported_state = Circuit.from_ir(imported, dtype=torch.complex128).state()[0]
    torch.testing.assert_close(imported_state, expected, atol=1e-10, rtol=1e-10)

    exported = to_braket(imported)
    exported_state = torch.as_tensor(
        np.asarray(exported.to_unitary(), dtype=np.complex128)[:, 0].copy()
    )
    torch.testing.assert_close(exported_state, expected, atol=1e-10, rtol=1e-10)
